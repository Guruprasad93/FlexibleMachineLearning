# import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

from transformers import AutoConfig, AutoTokenizer
from transformers import BertForSequenceClassification

# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl", load_in_8bit=True, device_map="auto")
model_id = 'google-bert/bert-base-uncased'
config = AutoConfig.from_pretrained(model_id)

config.num_labels = 7 # For Yelp (5) + IMDB (2)
model = BertForSequenceClassification.from_pretrained('google-bert/bert-base-uncased', config=config).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_id)

imdb = load_dataset("imdb")
imdb_train = load_dataset("imdb", split="train")
imdb_test = load_dataset("imdb", split="test")

yelp = load_dataset("yelp_review_full")
yelp_train = load_dataset("yelp_review_full", split="train")
yelp_test = load_dataset("yelp_review_full", split="test[:10%]")

breakpoint()

def preprocess_yelp(examples):
    tokenized = tokenizer(examples['text'], truncation=True, padding=True)
    return tokenized

def preprocess_imdb(examples):
    tokenized = tokenizer(examples['text'], truncation=True, padding=True)
    tokenized['label_2'] = [ele+5 for ele in examples['label']]
    return tokenized
 

# yelp_tokenized_dataset = yelp_dataset.map(preprocess, batched=True,  remove_columns=["text"])
yelp_tokenized_dataset = yelp.map(preprocess_yelp, batched=True,  remove_columns=["text"])
imdb_tokenized_dataset = imdb.map(preprocess_imdb, batched=True,  remove_columns=["text", "label"])
imdb_tokenized_dataset.rename_column("label_2", "label")

yelp_train_dataset=yelp_tokenized_dataset['train']
yelp_eval_dataset=yelp_tokenized_dataset['test'].shard(num_shards=2, index=0)
yelp_test_dataset=yelp_tokenized_dataset['test'].shard(num_shards=2, index=1)

# Extract the number of classess and their names
num_labels = yelp['train'].features['label'].num_classes
class_names = yelp["train"].features["label"].names
print(f"number of labels: {num_labels}")
print(f"the labels: {class_names}")

# Create an id2label mapping
# We will need this for our classifier.
id2label = {i: label for i, label in enumerate(list(range(config.num_labels)))}

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")


# use the same Training args for all models
training_args = TrainingArguments(
    output_dir='bert-yelp',
    evaluation_strategy='steps',
    learning_rate=5e-5,
    num_train_epochs=1,
    max_steps=1,
    per_device_train_batch_size=64,
)

def get_trainer(model, train_dataset, eval_dataset):
      return Trainer(
          model=model,
          args=training_args,
          train_dataset=train_dataset,
          eval_dataset=eval_dataset,
          data_collator=data_collator,
      )

full_finetuning_trainer = get_trainer(
    model=AutoModelForSequenceClassification.from_pretrained(model_id, id2label=id2label),
    train_dataset=yelp_train_dataset,
    eval_dataset=yelp_eval_dataset,
)

full_finetuning_trainer.train()

model_ft = full_finetuning_trainer.model
# --------- PEFT TRAINING 


# model = AutoModelForSequenceClassification.from_pretrained(model_id, id2label=id2label)

imdb_train_dataset=imdb_tokenized_dataset['train']
imdb_eval_dataset=imdb_tokenized_dataset['test'].shard(num_shards=2, index=0)
imdb_test_dataset=imdb_tokenized_dataset['test'].shard(num_shards=2, index=1)


peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
peft_model = get_peft_model(model_ft, peft_config)

print('PEFT Model')
peft_model.print_trainable_parameters()


peft_lora_finetuning_trainer = get_trainer(model=peft_model, train_dataset=imdb_train_dataset, eval_dataset=imdb_eval_dataset) 

peft_lora_finetuning_trainer.train()
peft_lora_finetuning_trainer.evaluate()
