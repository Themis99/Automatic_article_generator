from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
from datasets import load_metric
import pandas as pd
import nltk
import numpy as np
import gc
gc.collect()
torch.cuda.empty_cache()

import wandb
wandb.login()



class PegasusDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])  # torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels['input_ids'])  # len(self.labels)

def prepare_data(model_name, 
                 train_texts, train_labels, 
                 val_texts=None, val_labels=None, 
                 test_texts=None, test_labels=None):
  """
  Prepare input data for model fine-tuning
  """
  tokenizer = PegasusTokenizer.from_pretrained(model_name)

  prepare_val = False if val_texts is None or val_labels is None else True
  prepare_test = False if test_texts is None or test_labels is None else True

  def tokenize_data(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True)
    decodings = tokenizer(labels, truncation=True, padding=True)
    dataset_tokenized = PegasusDataset(encodings, decodings)
    return dataset_tokenized

  train_dataset = tokenize_data(train_texts, train_labels)
  val_dataset = tokenize_data(val_texts, val_labels) if prepare_val else None
  test_dataset = tokenize_data(test_texts, test_labels) if prepare_test else None

  return train_dataset, val_dataset, test_dataset, tokenizer

def flatten(t):
    return [item for sublist in t for item in sublist]

def str_enforce(t):
  new = []
  for item in t:
    new.append(str(item))
  return new

def load_dataset(train_path_inputs,train_path_targets,test_path_inputs,test_path_targets):
  
  # train_dataset
  train_inputs = pd.read_csv(train_path_inputs,header = None)
  train_inputs = train_inputs.drop(0,axis =1)
  train_targets = pd.read_csv(train_path_targets,header= None)
  train_targets = train_targets.drop(0,axis =1)

  train_inputs = train_inputs.values.tolist()
  train_targets = train_targets.values.tolist()

  train_inputs = flatten(train_inputs)
  train_targets = flatten(train_targets)

  train_inputs = str_enforce(train_inputs)
  train_targets = str_enforce(train_targets)


  # test_dataset
  test_inputs = pd.read_csv(test_path_inputs,header = None)
  test_inputs = test_inputs.drop(0,axis =1)
  test_targets = pd.read_csv(test_path_targets,header= None)
  test_targets = test_targets.drop(0,axis =1)

  test_inputs = test_inputs.values.tolist()
  test_targets = test_targets.values.tolist()

  test_inputs = flatten(test_inputs)
  test_targets = flatten(test_targets)

  test_inputs = str_enforce(test_inputs)
  test_targets = str_enforce(test_targets)

  return train_inputs,train_targets,test_inputs,test_targets


def prepare_fine_tuning(model_name, tokenizer, train_dataset, val_dataset=None, compute_metrics=None, freeze_encoder=False, output_dir='./results'):
  """
  Prepare configurations and base model for fine-tuning
  """
  torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
  #print(torch_device)
  model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

  model.config.length_penalty = 1     # lenght penarty to force model output longer sequences
  model.config.max_length = 360       # max_lenght to force model output longer sequences
  model.config.min_length = 360       # min_lenght to force model output longer sequences

  if freeze_encoder:
    for param in model.model.encoder.parameters():
      param.requires_grad = False

  if val_dataset is not None:
    training_args = Seq2SeqTrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=100,           # total number of training epochs
      per_device_train_batch_size=1,   # batch size per device during training, can increase if memory allows
      per_device_eval_batch_size=1,    # batch size for evaluation, can increase if memory allows
      save_steps=500,                  # number of updates steps before checkpoint saves
      save_total_limit=5,              # limit the total amount of checkpoints and deletes the older checkpoints
      evaluation_strategy='steps',     # evaluation strategy to adopt during training
      eval_steps=100,                  # number of update steps before evaluation
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
      logging_steps=10,           # directory for storing logs
      report_to="wandb"

    )

    trainer = Seq2SeqTrainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      eval_dataset=val_dataset,            # evaluation dataset
      tokenizer=tokenizer,

    )

  else:
    training_args = Seq2SeqTrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=100,           # total number of training epochs
      per_device_train_batch_size=1,   # batch size per device during training, can increase if memory allows
      save_steps=500,                  # number of updates steps before checkpoint saves
      save_total_limit=5,              # limit the total amount of checkpoints and deletes the older checkpoints
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
      logging_steps=10,
      

    )

    trainer = Seq2SeqTrainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      tokenizer=tokenizer,

    )

  return trainer

if __name__=='__main__':
  # load datset
  PATH_train_inputs = '/home/efthimios/dataset/train/train_inputs.txt'
  PATH_train_targets = '/home/efthimios/dataset/train/train_targets.txt'
  PATH_test_inputs = '/home/efthimios/dataset/test/test_inputs.txt'
  PATH_test_targets = '/home/efthimios/dataset/test/test_targets.txt'

  train_texts,train_labels,test_texts,test_labels = load_dataset(PATH_train_inputs,PATH_train_targets,PATH_test_inputs,PATH_test_targets)
  #sample_train = 100
  #sample_test = 100

  #train_texts = train_texts[:sample_train]
  #train_labels = train_labels[:sample_train]
  #test_texts = test_texts[:sample_test]
  #test_labels = test_labels[:sample_test]

  # use Pegasus Large model as base for fine-tuning
  model_name = 'google/pegasus-multi_news'
  train_dataset, _, test_dataset, tokenizer = prepare_data(model_name, train_texts, train_labels, test_texts=test_texts, test_labels=test_labels)
  trainer = prepare_fine_tuning(model_name, tokenizer, train_dataset,val_dataset = test_dataset,freeze_encoder=True)
  trainer.train()
  trainer.evaluate(test_dataset)
  wandb.finish()
