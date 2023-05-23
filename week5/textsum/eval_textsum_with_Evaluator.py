import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable annoying tensorflow warnings

import evaluate
import numpy as np
import nltk

from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from evaluate import evaluator


if __name__ == '__main__':


    ### ==== LOAD DATA ==== ###
    # load Vietnamese Text Summarization in this repo: https://huggingface.co/datasets/ithieund/VietNews-Abs-Sum
    DATA_FILES = {
        'train': 'processed/train_desegmented.jsonl',
        'validation': 'processed/valid_desegmented.jsonl',
        'test': 'processed/test_desegmented.jsonl'
    }
    raw_dataset = load_dataset("ithieund/VietNews-Abs-Sum", data_files=DATA_FILES, split='test').shuffle(seed=42).select(range(10)) # load desegmented parts

    ### ==== LOAD MODEL ==== ###
    # Choose a summarization model
    MODEL_NAME = 'VietAI/vit5-base-vietnews-summarization'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    ### ==== USE METRIC ==== ###
    # load rouge score
    metric = evaluate.load("rouge")
    eval_results_with_metric = metric.compute(predictions=raw_dataset["title"], references=raw_dataset["abstract"])
    # print(eval_results_with_metric)


    ### ==== USE EVALUATOR ==== ###
    # The Evaluator classes allow to evaluate a triplet of model, dataset, and metric.
    # Support several tasks such as TextClassificationEvaluator, TokenClassificationEvaluator, QuestionAnsweringEvaluator,...
    task_evaluator = evaluator("summarization")

    eval_results = task_evaluator.compute(
        model_or_pipeline=MODEL_NAME,
        data=raw_dataset,
        input_column='article',
        label_column='abstract'
    )
    # print(eval_results)

    ### ==== USE EvaluationSuite ==== ###
    # Evaluate models on a variety of different tasks
    # read more https://huggingface.co/docs/evaluate/evaluation_suite