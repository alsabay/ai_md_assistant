######################################################
# Pre-processing of medical_dialog dataset from hf hub.
# Stores formatted, tokenized, chunked 
# training data to s3 bucket.
# Derived from @philschmid hugginface-llama-2-samples
# on a different hf dataset
######################################################

import sagemaker
import boto3
from random import randint
from itertools import chain
from functools import partial
from datasets import load_dataset
from random import randrange
import json
import pandas as pd
from transformers import AutoTokenizer

#sagemaker_session_bucket='mlpipes-sm'                                # us-west-2
sagemaker_session_bucket='mlpipes-md-assistant'                  # us-east-1
#sagemaker_session_bucket='md-assistant-us-east-2'                  # us-east-2
role_name = 'Sagemaker-mle'
dataset_name = 'medical_dialog'
dataset_lang = 'en'
model_id = 'meta-llama/Llama-2-13b-chat-hf'
# empty list to save remainder from batches to use in next batch
remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
# sess = sagemaker.Session()

# fetch tokenizer pad_token
def fetch_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

tokenizer = fetch_tokenizer(model_id)

# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it not exists
def init_sagemaker(role, session_bucket):
    if session_bucket is None and sess is not None:
        # set to default bucket if a bucket name is not given
        session_bucket = sess.default_bucket()
    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        iam = boto3.client('iam')
        role = iam.get_role(RoleName=role_name)['Role']['Arn']

    sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)
    return (sess, role)

# load dataset and remove un-used fields
def load_and_extract(dataset_name, dataset_lang):
    dataset = load_dataset(dataset_name, dataset_lang)
    dataset = dataset['train'].remove_columns(['file_name', 'dialogue_id', 'dialogue_url'])
    return dataset

# function to format samples to llama-2-chat-hf format
# which is:
# <s>[INST] <<SYS>>
# System prompt
# <</SYS>>
# User prompt [/INST] Model answer </s>
def format_dialogue(sample):
    instruction = f"[INST]{sample['dialogue_turns']['utterance'][0]}[/INST]"
    response = f"{sample['dialogue_turns']['utterance'][1]}"
    # join all the parts together
    prompt = "\n".join([i for i in [instruction, response] if i is not None])
    return '<s>' + prompt + '</s>'

# template dataset to add prompt to each sample
def template_dataset(sample):
    sample["text"] = f"{format_dialogue(sample)}{tokenizer.eos_token}"
    return sample

# chunk and tokenize
def chunk(sample, chunk_length=2048):
    # define global remainder variable to save remainder from batches to use in next batch
    global remainder
    # Concatenate all texts and add remainder from previous batch
    concatenated_examples = {k: list(chain(*sample[k])) for k in sample.keys()}
    concatenated_examples = {k: remainder[k] + concatenated_examples[k] for k in concatenated_examples.keys()}
    # get total number of tokens for batch
    batch_total_length = len(concatenated_examples[list(sample.keys())[0]])

    # get max number of chunks for batch
    if batch_total_length >= chunk_length:
        batch_chunk_length = (batch_total_length // chunk_length) * chunk_length

    # Split by chunks of max_len.
    result = {
        k: [t[i : i + chunk_length] for i in range(0, batch_chunk_length, chunk_length)]
        for k, t in concatenated_examples.items()
    }
    # add remainder to global variable for next batch
    remainder = {k: concatenated_examples[k][batch_chunk_length:] for k in concatenated_examples.keys()}
    # prepare labels
    result["labels"] = result["input_ids"].copy()
    return result

def process_data():
    sm_session, _ = init_sagemaker(role_name, sagemaker_session_bucket)
    ds = load_and_extract(dataset_name, dataset_lang)
    ds = ds.map(template_dataset)
    print(ds[randint(0, len(ds))]["text"])  # print random sample
    lm_dataset = ds.map(
        lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(ds.features)
            ).map(partial(chunk, chunk_length=2048),
            batched=True,
        )
    print(f"Total number of samples: {len(lm_dataset)}")
    # save train_dataset to s3
    training_input_path = f's3://{sm_session.default_bucket()}/processed/llama/md_dialouge/train'
    lm_dataset.save_to_disk(training_input_path)
    print("uploaded data to:")
    print(f"training dataset to: {training_input_path}")

if __name__ == '__main__':
    process_data()
