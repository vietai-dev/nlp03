import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable annoying tensorflow warnings

import evaluate
import numpy as np
import nltk
import torch
import math

from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    get_scheduler
)
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def show_samples(dataset, num_samples=5, seed=42):
    sample = dataset["train"].shuffle(seed=seed).select(range(num_samples))
    for example in sample:
        print(f"\n>> Title: {example['title']}")
        print(f">> Abstract: {example['abstract']}")
        print(f">> News: {example['article']}")
        print(len(example['article']))


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


if __name__ == '__main__':


    ### ==== LOAD DATA ==== ###
    # load Vietnamese Text Summarization in this repo: https://huggingface.co/datasets/ithieund/VietNews-Abs-Sum
    DATA_FILES = {
        'train': 'processed/train_desegmented.jsonl',
        'validation': 'processed/valid_desegmented.jsonl',
        'test': 'processed/test_desegmented.jsonl'
    }
    raw_dataset = load_dataset("ithieund/VietNews-Abs-Sum", data_files=DATA_FILES) # load desegmented parts

    # Let's take a look at the dataset
    # dataset contains 4 columns ['guid', 'title', 'abstract', 'article']
    # ratio of train:valid:test is 100k:22k:22k
    # print(raw_dataset)
    # show_samples(raw_dataset)

    # TODO: preprocessing - deduplication, filtering characters, text normalization

    ### ==== LOAD MODEL ==== ###
    # Choose T5 family such as mT5, ViT5, mBART-50
    MODEL_NAME = 'google/mt5-small'
    MODEL_NAME = 'VietAI/vit5-base'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    ### === TOKENIZE DATA & PREPARE INPUT/OUTPUT === ###
    # Let's see output of tokenizer
    tmp_inputs = tokenizer(raw_dataset['train'][0]['abstract'])
    tmp_inputs_ids = tokenizer.convert_ids_to_tokens(tmp_inputs.input_ids)
    # print(tmp_inputs)
    # print(tmp_inputs_ids)

    # Let's tokenize input & output for model
    MAX_INPUT_LENGTH = 512
    MAX_OUTPUT_LENGTH = 50
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples['article'],
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
        )
        labels = tokenizer(
            examples['abstract'],
            max_length=MAX_OUTPUT_LENGTH,
            truncation=True,
        )
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    # Then use .map() to tokenize (train, valid, test)
    tokenized_dataset = raw_dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=raw_dataset['train'].column_names, # remove these columns to make it work with map fn
        num_proc=os.cpu_count() # count number of cpus
    )
    # print(tokenized_dataset['train'][0])

    # Then prepare batches
    # use DataCollatorForSeq2Seq designed for seq2seq problem
    # might use other DataCollator for other tasks
    # such as DataCollatorForTokenClassification, DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
    # find more https://huggingface.co/docs/transformers/main_classes/data_collator#data-collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100, # ignore these labels when models compute loss_fn
        pad_to_multiple_of=8, # maximize the usage of tensors
    )
    # see output of DataCollator
    # labels contains -100 value
    tmp_features = [tokenized_dataset["train"][i] for i in range(5)]
    tmp_features_collator = data_collator(tmp_features)
    # print(tmp_features_collator)


    ### === NEW!!! DEFINE DATA LOADER === ###
    # DEBUG: sample only few documents
    MAX_SAMPLES = 10
    train_dataset = tokenized_dataset["validation"].select(range(MAX_SAMPLES))
    eval_dataset = tokenized_dataset["validation"].select(range(MAX_SAMPLES))
    test_dataset = tokenized_dataset["test"].select(range(MAX_SAMPLES))

    BATCH_SIZE = 2
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=data_collator, 
        batch_size=BATCH_SIZE
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        shuffle=True, 
        collate_fn=data_collator, 
        batch_size=BATCH_SIZE
    )

    ### === NEW!!! DEFINE OPTIMIZER & SCHEDULER === ###
    LEARNING_RATE = 2e-5
    NUM_WARMUP_STEPS = 5
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = get_scheduler(
        name='constant',
        optimizer=optimizer,
        num_warmup_steps=NUM_WARMUP_STEPS,
    )

    ### === NEW!!!: DEFINE ACCELERATOR === ###
    # Prepare everything with our accelerator
    GRADIENT_ACC_STEPS = 2
    accelerator = Accelerator(gradient_accumulation_steps=GRADIENT_ACC_STEPS)
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    ### ==== DEFINE METRICS ==== ###
    # for text summarization, ROUGE score is commonly used
    # ROUGE-1, ROUGE-2, ROUGE-L are commonly reported in paper
    metric = evaluate.load("rouge")
    # provide compute_metrics() to evaluate model during training
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        # Replace -100 in the labels as we can't decode them
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Decode generated summaries into text
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # A simple post-processing: ROUGE expects a newline after each sentence
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        # Compute ROUGE scores
        result = metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        return {k: round(v, 4) for k, v in result.items()}
    
    ### ==== DEFINE TRAINING LOOP ==== ###
    NUM_EPOCHS = 2
    NUM_STEPS_PER_EPOCH = math.ceil(len(train_dataloader) / GRADIENT_ACC_STEPS)
    NUM_TRAINING_STEPS = NUM_EPOCHS * NUM_STEPS_PER_EPOCH
    OUTPUT_DIR = 'textsum_with_accelerate'

    # define progress_bar for monitoring
    progress_bar = tqdm(range(NUM_TRAINING_STEPS))

    # define training loop
    for epoch in range(NUM_EPOCHS):
        print(f'This is EPOCH: {epoch}')
        # Training
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model): # NEW!!! for gradient accumulation
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss) # NEW!!!
            
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)

        # TODO: Evaluation
        model.eval()

        # TODO: Compute metrics

        # Save model
        # make sure all processes are joined
        accelerator.wait_for_everyone()    
        # remove all special model wrappers added during the distributed process
        unwrapped_model = accelerator.unwrap_model(model)
        # save
        unwrapped_model.save_pretrained(OUTPUT_DIR, save_function=accelerator.save)