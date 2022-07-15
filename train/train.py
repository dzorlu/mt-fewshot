#!/usr/bin/env python

import json
import logging
import os
import sys
import random
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from collections import Counter, defaultdict
from typing import List

from tqdm import tqdm
from tqdm import trange
from nlgeval import NLGEval
import nltk


logger = logging.getLogger(__name__)




@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    target_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


    logger.info(f"Training/evaluation parameters {training_args}")


    # TODO: add checkpointing

    # load data
    data_files = {}
    if data_args.train_file is not None:
        # data_files["train"] = data_args.train_file
        data_files = {'train':data_args.train_file}
        extension = data_args.train_file.split(".")[-1]
        train_dataset = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir, split='train')
        column_names = train_dataset.column_names

    if data_args.validation_file is not None:
        # data_files["validation"] = data_args.validation_file
        data_files = {'train':data_args.validation_file}
        extension = data_args.validation_file.split(".")[-1]
        eval_dataset = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir, split='train')

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )


    # tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # no resizing the tokens.
    # no prefix

    # setting data processing params.
    target_column = data_args.target_column
    text_column = data_args.text_column
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else 'longest'
    

    # preprocessing the dataset
    def preprocess_function(examples):

        # remove pairs where at least one record is None
        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] is not None and examples[target_column][i] is not None:
                inputs.append(examples[text_column][i])
                targets.append(examples[target_column][i])# + tokenizer.eos_token)

        selinptus = []
        for i, inp in enumerate(inputs):
            selinptus.append(inp)
        model_inputs = tokenizer(selinptus, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.

        if padding == "max_length" or padding=='longest' and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    max_target_length = data_args.val_max_target_length
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    # data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metrics
    def nlgeval_metrics(refs: List[str] , hyps: List[str]):
        nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=['METEOR'])
        # metrics = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L']
        metrics = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L']
        results = defaultdict(list)
        for ref, hyp in zip(refs, hyps):
            if type(ref) is not list:
                print('PUT REFS in LIST')
                exit(0)
            metrics_dict = nlgeval.compute_individual_metrics(ref, hyp)
            for metric in metrics:
                results[metric].append(metrics_dict[metric])
        return results

    metrics = {
        'rouge': load_metric("rouge"), 
        'accuracy': load_metric("accuracy"), 
        'f1': load_metric("f1"), 
        'recall': load_metric("recall"), 
        'precision': load_metric("precision"), 
        'nlgeval': nlgeval_metrics
    }

    def compute_metrics(eval_preds, input_ids=None, global_step=0):

        print('eval_preds')
        print(eval_preds)
    
        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            # rougeLSum expects newline after each sentence
            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

            return preds, labels

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        print("preds:")
        print(preds)
        print('labels')
        print(labels)

        
        # input_ids = input_ids.cpu()
        input_ids = np.where(input_ids != -100, input_ids, tokenizer.pad_token_id)
        decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        # print(decoded_preds, decoded_labels)
        class_decoded_labels, class_decoded_preds = [], []
        for i in range(len(decoded_labels)):
            # not sure why 'not present' excluded?
            if len(decoded_labels[i])<30 and decoded_labels[i]!='not present':
                class_decoded_labels.append(decoded_labels[i])
                class_decoded_preds.append(decoded_preds[i])

        results = {}
        for metric_name, metric in metrics.items():
            if metric_name == 'nlgeval':
                decoded_labels = [x if len(x)>0 else 'empty' for x in decoded_labels]
                refs = [x.split() for x in decoded_labels]
                decoded_predse = [o for my_o, o in zip(refs, decoded_preds) if 'present' not in my_o]
                refs = [my_o for my_o in refs if 'present' not in my_o]
                result = nlgeval_metrics(refs=refs, hyps=decoded_predse)
                result = {x: np.mean(y) for x, y in result.items()}
            elif 'rouge' in metric_name:
                result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
                # Extract a few results from ROUGE
                result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
            elif metric_name in ['accuracy', 'precision','recall', 'f1']:
                if len(class_decoded_labels)>0:
                    classes_names = set(class_decoded_preds+class_decoded_labels)
                    classes_vocab = {k: v for v, k in enumerate(classes_names)}
                    class_decoded_preds = [classes_vocab[x] for x in class_decoded_preds]
                    class_decoded_labels = [classes_vocab[x] for x in class_decoded_labels]
                    if metric_name=='accuracy':
                        result = metric.compute(predictions=class_decoded_preds, references=class_decoded_labels)
                    else:
                        result = metric.compute(predictions=class_decoded_preds, references=class_decoded_labels, average="micro")

                    # Extract a few results from ROUGE
                    result = {key: value * 100 for key, value in result.items()}
                # print(metric_name, result)
            else:
                result = {metric_name:metric}
            results.update(result)
            # print(results)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        results["gen_len"] = np.mean(prediction_lens)
        results = {k: round(v, 4) for k, v in results.items()}

        # write result
        output_prediction_file = os.path.join(training_args.output_dir, f"intermediate_result_{global_step}.jsonl")
        with open(output_prediction_file, "a") as writer:
            for inputs, preds, labels in zip(decoded_inputs, decoded_preds, decoded_labels):
                output_str = json.dumps({'input_ids': inputs, 'preds': preds, 'labels': labels})
                writer.write(output_str + '\n')

        output_prediction_file = os.path.join(training_args.output_dir, f"intermediate_metrics_{global_step}.json")
        with open(output_prediction_file, "a") as writer:
            json.dump(results, writer)

        return results

    # trainer
    # where do you define the loss function?
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        output_args_file = os.path.join(training_args.output_dir, f"modelargs.json")
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(output_args_file, "w+") as writer:
            all_argsdict = {**model_args.__dict__ , **data_args.__dict__, **training_args.to_dict()}
            json.dump(all_argsdict, writer)

        checkpoint = None # nocheckpoint yet.
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()






