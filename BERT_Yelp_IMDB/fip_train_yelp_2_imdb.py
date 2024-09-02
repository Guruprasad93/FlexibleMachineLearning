import copy
import random
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from datasets import ClassLabel, load_dataset, load_from_disk, load_metric
from IPython.display import HTML, display
from torch import nn
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

parser = ArgumentParser('sparseSubmanifold')
parser.add_argument('--sparsity', type=int, default=0)
parser.add_argument('--num_train_epochs', type=int, default=10)
parser.add_argument('--dataset', type=str, default='wikipedia')
parser.add_argument('--subdataset', type=str, default='20220301.en')
parser.add_argument('--token_dir', type=str, default='/central/groups/mthomson/Guru/grow_functional_brains/SOTA/HF_datasets/wikipedia-tokenized-data')
parser.add_argument('--taskName', type=str, default='wikipedia')
args = parser.parse_args()


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))


#show_random_elements(datasets_yelp["train"])

def main():

    # ----------------- YELP DATASET ----------------- #

    # IMPORT MODEL AND TOKENIZER FOR YELP DATASET
    model_checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    # PREPARE YELP DATASET
    datasets_yelp = load_dataset('yelp_review_full', 'yelp_review_full', cache_dir='/central/groups/mthomson/Guru/grow_functional_brains/SOTA/HF_datasets')
    key = 'text'
    tokenized_datasets_yelp = datasets_yelp.map(lambda examples: tokenizer(examples[key],
                                                        max_length=512,
                                                        padding="max_length",
                                                        truncation=True),
                                                        batched=True)

    tokenized_datasets_yelp.save_to_disk('../HF_datasets/yelpRate-token-data')
    tokenized_datasets_yelp = load_from_disk('../HF_datasets/yelpRate-token-data/')


    # ----------------- IMDB DATASET ----------------- #

    # LOAD IMDB DATASET
    datasets_imdb = load_dataset('imdb', 'plain_text', cache_dir='/central/groups/mthomson/Guru/grow_functional_brains/SOTA/HF_datasets')

    #show_random_elements(datasets_imdb["train"])

    # TOKENIZING IMDB DATASET
    model_checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    key = 'text'
    tokenized_datasets = datasets_imdb.map(lambda examples: tokenizer(examples[key],
                                                        max_length=512,
                                                        padding="max_length",
                                                        truncation=True),
                                                        batched=True)

    tokenized_datasets.save_to_disk('../HF_datasets/imdbRate-token-data')
    tokenized_datasets_imdb = load_from_disk('../HF_datasets/imdbRate-token-data')


    # ----------------- UPDATE LABELS OF IMDB DATASET ----------------- #
    def update_label(example):
        example['label'] = example['label'] + 5
        return example

    # After updating labels of classes for IMDB Dataset

    tokenized_datasets_imdb['train'] = tokenized_datasets_imdb['train'].map(update_label)
    tokenized_datasets_imdb['test'] = tokenized_datasets_imdb['test'].map(update_label)
    tokenized_datasets_imdb.save_to_disk('../HF_datasets/imdbRate-token-data2')
    tokenized_datasets_imdb = load_from_disk('../HF_datasets/imdbRate-token-data2')



    # LOAD BERT MODEL TO PERFORM BOTH TASKS (YELP AND IMDB)

    num_labels = len(np.unique(datasets_yelp['train']['label'])) + len(np.unique(datasets_imdb['train']['label']))
    print("TOtal # of labels = ", num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    #model = AutoModelForSequenceClassification.from_pretrained('../FIP_transformers/bert-base-uncased-finetune-glue-SST2/checkpoint-4000/', num_labels=num_labels)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    # --------------- TRAIN NETWORK ON YELP DATASET (EARLY STOPPER) --------------- #

    model_name = model_checkpoint.split("/")[-1]
    training_args = TrainingArguments(
        f"{model_name}-finetune-yelpRate",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=8
    )


    trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets_yelp['train'],
                eval_dataset=tokenized_datasets_yelp['test'],
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                )

    trainer.train()

    # If NETWOrK IS ALREADY TRAINED ON YELP, LOAD the TRAINED YELP BERT MODEL

    # model = AutoModelForSequenceClassification.from_pretrained('../FIP_transformers/bert-base-uncased-finetune-yelpRate/checkpoint-155500/', num_labels=num_labels)

    # ------------ TRAINING HYPERPARAMETERS FOR LLM TRAINING ON IMDB ------------ #

    model_name = model_checkpoint.split("/")[-1]
    training_args = TrainingArguments(
        f"{model_name}-FIP-yelp2IMDB",
        evaluation_strategy = "no",
        save_steps=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=8
    )


    print("CREATING COPY OF MODEL")
    model_ori = copy.deepcopy(model) # MODEL already trained on YELP
    model_ori.to('cuda:0')
    # global numSteps_taken
    # numSteps_taken = [0]

    # Subsample small # of examples from the yelp dataset.
    small_train_yelp = tokenized_datasets_yelp['train'].shuffle(seed=42).select(range(2000))
    small_test_yelp = tokenized_datasets_yelp['test'].shuffle(seed=42).select(range(2000))
    print("YELP small dataset", small_train_yelp)


    # Defining trainer1 to get the data loader for subsampled YELP
    trainer1 = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_yelp,
        eval_dataset=small_test_yelp,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        )

    prevData_loader = trainer1.get_train_dataloader()

    # --------- CUSTOM TRAINER FOR FIP TUNING BERT ON SECOND DATASET TO PREVENT CF -------- #

    class CustomTrainer(Trainer):

        def compute_loss(self, model, inputs, return_outputs=False):
            """
            How the loss is computed by Trainer. By default, all models return the loss in the first element.
            Subclass and override for custom behavior.
            """
            
            f_softmax = nn.Softmax(dim=1)

            inputs_prev = next(iter(prevData_loader))
            inputs_prev = inputs_prev.to("cuda:0")

            outputs = model(**inputs_prev) # PREVIOUS DATA (YELP DATA)
            outputs = f_softmax(outputs.logits)

            outputs_ori = model_ori(**inputs_prev)
            outputs_ori = f_softmax(outputs_ori.logits).detach()

            epsAdd = max(1e-10, torch.min(outputs_ori)*1e-3)
            bcloss = -torch.log(torch.sum(torch.sqrt(outputs*outputs_ori+epsAdd), axis=1))

            loss = torch.sum(bcloss)
            
            op_current = model(**inputs) # NEW DATA (IMDB DATA - inputs)

            lambda1= 1
            loss = loss + lambda1*torch.sum(op_current["loss"])

            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            return (loss, outputs) if return_outputs else loss


    # ----------------- FIP TRAINING BERT ON IMDB DATASET ----------------- #

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets_imdb['train'],
        eval_dataset=tokenized_datasets_imdb['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        )


    trainer.train()


if __name__ == '__main__':
    main()


