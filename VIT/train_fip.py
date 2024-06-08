# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import copy
from datetime import timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_loader import get_loader
from utils.dist_util import get_world_size

from transformers import AutoModelForImageClassification

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model, global_step=None):
    model_to_save = model.module if hasattr(model, 'module') else model
    if global_step is not None:
        model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % str(global_step))
    else:
        model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    model.save_pretrained(model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):

    # Prepare model
    if args.checkpoint:
        model = AutoModelForImageClassification.from_pretrained(args.pretrained_dir, num_labels=args.num_classes)
    else:
        model = AutoModelForImageClassification.from_pretrained(args.model_type, num_labels=args.num_classes,
                                                            ignore_mismatched_sizes=True)
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step, val_txt = ""):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("\n")
    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x).logits
            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating %s" %val_txt+ "... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results: %s" % val_txt)
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalars('test/accuracy', {val_txt: accuracy}, global_step)
    return accuracy


def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.output_dir+"/logs", exist_ok=True)

        writer = SummaryWriter(log_dir=os.path.join(args.output_dir+"/logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args, cmin = (args.task_n-1)*10, cmax = args.task_n*10, relabel = False)
    old_loaders = [get_loader(args, cmin = i*10, cmax = (i+1)*10, relabel = False) for i in range(args.task_n)]

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    logger.info("\n")
    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0

    accuracies = [[] for _ in range(args.task_n)]
    for i in range(args.task_n):
        txt = "cifar_"+str(i*10)+":"+str((i+1)*10-1)
        accuracy = valid(args, model, writer, old_loaders[i][1], global_step, val_txt = txt)
        accuracies[i].append(accuracy)

    if args.use_fip:
        fip_batch_size = (args.task_n-1)*10*50
        batch_loader, _ = get_loader(args, cmin=0, cmax=(args.task_n-1)*10, relabel = False, data_size = fip_batch_size)
        old_model = copy.deepcopy(model)
        f_softmax = torch.nn.Softmax(dim=1)
        batch_loader_iterator = iter(batch_loader)

    # Train!
    logger.info("\n")
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])


        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            output = model(x).logits
            loss_fct = torch.nn.CrossEntropyLoss()
            CEloss = loss_fct(output, y)

            if args.use_fip:
                try:
                    x_old, y_old = next(batch_loader_iterator)
                except StopIteration:
                    batch_loader_iterator = iter(batch_loader)
                    x_old, y_old = next(batch_loader_iterator)
                x_old = x_old.to(args.device)

                output_ori = f_softmax(old_model(x).logits)
                output_d2 = f_softmax(model(x_old).logits)
                output_newDest = f_softmax(old_model(x_old).logits)
                output_ori = output_ori.detach()
                output_newDest = output_newDest.detach()
                epsAdd = max(1e-10, torch.min(output_ori)*1e-3)
                loss1 = torch.sum(-torch.log(torch.sum(torch.sqrt(output*output_ori+epsAdd), axis=1)))
                loss2 = torch.sum(-torch.log(torch.sum(torch.sqrt(output_d2*output_newDest+epsAdd), axis=1)))
            else:
                loss2 = 0
            loss = CEloss + loss2

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    for i in range(args.task_n):
                        txt = "cifar_"+str(i*10)+":"+str((i+1)*10-1)
                        accuracy = valid(args, model, writer, old_loaders[i][1], global_step, val_txt = txt)
                        accuracies[i].append(accuracy)

                    if best_acc < accuracy:
                        save_model(args, model)
                        best_acc = accuracy
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


    timesteps = [i for i in range(0, args.num_steps+1, args.eval_every)]
    for i in range(args.task_n):
        txt = "cifar_"+str(i*10)+":"+str((i+1)*10-1)
        plt.plot(timesteps, accuracies[i], label=txt)

    plt.title("Accuracy plot")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "accuracy_plot.png"))

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--num_classes", required=True, type=int,
                        help="Number of output classes")
    parser.add_argument("--task_n", required=True, type=int,
                        help="Number of the task currently training")
    parser.add_argument("--model_type", type=str, default="google/vit-base-patch16-224",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--fip_batch_size", default=512, type=int,
                            help="Total batch size for fip training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=40, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=240, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=120, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--checkpoint', action='store_true',
                        help="Whether to use custom pre-trained model or imagenet21k pretrained")
    parser.add_argument('--use_fip', action='store_true',
                            help="Whether to use fip training or normal training")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1)))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)


if __name__ == "__main__":
    main()
