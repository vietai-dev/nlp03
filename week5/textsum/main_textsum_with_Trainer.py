import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable annoying tensorflow warnings

import evaluate
import numpy as np
import nltk
nltk.download("punkt")


from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)



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
    
    ### ==== DEFINE TRAINING ==== ###
    BATCH_SIZE = 2
    NUM_EPOCH = 3
    NUM_STEPS = 10
    GRAD_ACC_STEPS = 2
    LOGGING_STEPS = 10

    # sample only few documents
    MAX_EVAL_SAMPLES = 4
    eval_dataset = tokenized_dataset["validation"].select(range(MAX_EVAL_SAMPLES))
    test_dataset = tokenized_dataset["test"].select(range(MAX_EVAL_SAMPLES))

    # We need to define arguments first
    training_args = Seq2SeqTrainingArguments(
        output_dir='ViTextSum',
        evaluation_strategy="epoch", # can be step as well
        learning_rate=5.6e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        eval_accumulation_steps=GRAD_ACC_STEPS,
        weight_decay=0.01,
        save_total_limit=3,
        # num_train_epochs=NUM_EPOCH,
        max_steps=4,
        predict_with_generate=True,
        logging_steps=10
    )

    # Put everything into Seq2SeqTrainer
    # each problem can have its own Trainer class & TrainingArguments class
    # such as (Trainer+TrainingArguments) and (Seq2SeqTrainer+Seq2SeqTrainingArguments)
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train()
    # Save train output
    trainer.save_model() 
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # See final scores after training
    trainer.evaluate(metric_key_prefix="eval") 
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Predict 
    predict_results = trainer.predict(test_dataset, metric_key_prefix="predict") 
    metrics = predict_results.metrics
    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)
    predictions = predict_results.predictions
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    predictions = tokenizer.batch_decode(
        predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    predictions = [pred.strip() for pred in predictions]
    output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
    with open(output_prediction_file, "w") as writer:
        writer.write("\n".join(predictions))