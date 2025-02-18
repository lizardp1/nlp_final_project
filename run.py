import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
import evaluate
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, convert_to_language_rows, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


NUM_PREPROCESSING_WORKERS = 2


def main():

    wandb.init(
        project='xnli_nli_project',
        name='finetune-r1-anli',
        # config=training_args.to_dict(),  # Optionally log training arguments
    )
    argp = HfArgumentParser(TrainingArguments)
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True,
                      help="""This argument specifies which task to train/evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.
        By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--source_filter', type=str, default=None,
                      help='Filter the dataset by source field (e.g., "anli").')
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')
    argp.add_argument('--predict_file', type=str, default=None,
                  help='Path to a JSONL file containing the dataset to use for prediction.')

    training_args, args = argp.parse_args_into_dataclasses()

    # Dataset selection
    # IMPORTANT: this code path allows you to load custom datasets different from the standard SQuAD or SNLI ones.
    # You need to format the dataset appropriately. For SNLI, you can prepare a file with each line containing one
    # example as follows:
    # {"premise": "Two women are embracing.", "hypothesis": "The sisters are hugging.", "label": 1}
    if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        dataset_id = None
        # Load from local json/jsonl file
        dataset_full = datasets.load_dataset('json', data_files=args.dataset)

        train_split = dataset_full['train'].filter(lambda x: x['split'] == 'train')
        eval_split = dataset_full['train'].filter(lambda x: x['split'] == 'eval')

        dataset = {'train': train_split, 'eval': eval_split}

        if training_args.do_predict and args.predict_file is not None:
            custom_test_dataset = datasets.load_dataset('json', data_files={'test': args.predict_file})
            test_dataset = custom_test_dataset['test']
            
        # By default, the "json" dataset loader places all examples in the train split,
        # so if we want to use a jsonl file for evaluation we need to get the "train" split
        # from the loaded dataset
        eval_split = 'eval'
    elif args.dataset == "facebook/xnli":
        dataset_id = None
        xnli_en = datasets.load_dataset("facebook/xnli", "en")
        xnli_ru = datasets.load_dataset("facebook/xnli", "ru")

        train_dataset = datasets.concatenate_datasets([xnli_en['train'], xnli_ru['train']])
        validation_dataset = datasets.concatenate_datasets([xnli_en['validation'], xnli_ru['validation']])

        train_dataset = train_dataset.shuffle(seed=42)
        validation_dataset = validation_dataset.shuffle(seed=42)

        #add 'language' column
        xnli_en = xnli_en.map(lambda examples: {'language': 'en'})
        xnli_ru = xnli_ru.map(lambda examples: {'language': 'ru'})

        if training_args.do_predict and args.predict_file is not None:
            custom_test_dataset = datasets.load_dataset('json', data_files={'test': args.predict_file})
            test_dataset = custom_test_dataset['test']
        else:
            test_dataset = datasets.concatenate_datasets([xnli_en['test'], xnli_ru['test']])
            test_dataset = test_dataset.shuffle(seed=42)

        #limit the number of samples
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.max_eval_samples is not None:
            validation_dataset = validation_dataset.select(range(args.max_eval_samples))

        dataset = {
            "train": train_dataset,
            "validation": validation_dataset,
            "test": test_dataset,
        }

        eval_split = "validation"

    elif args.dataset == "cointegrated/nli-rus-translated-v2021":
        dataset_id = None
        dataset = datasets.load_dataset('cointegrated/nli-rus-translated-v2021')

        if args.source_filter:
            def filter_by_source(example):
                return example['source'] == args.source_filter

            dataset = dataset.filter(filter_by_source)
            print(f"Filtered dataset to source: {args.source_filter}")

        label_to_int = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

        def map_labels(example):
            example['label'] = label_to_int[example['label']]
            return example
        dataset = dataset.map(map_labels)

        def to_en_format(example):
            return {
                'premise': example['premise'],
                'hypothesis': example['hypothesis'],
                'label': example['label'],
                'language': 'en'
            }
    
        def to_ru_format(example):
            return {
                'premise': example['premise_ru'],
                'hypothesis': example['hypothesis_ru'],
                'label': example['label'],
                'language': 'ru'
            }

        dataset_en_train = dataset['train'].map(to_en_format, remove_columns=dataset['train'].column_names)
        dataset_ru_train = dataset['train'].map(to_ru_format, remove_columns=dataset['train'].column_names)
        dataset_en_dev = dataset['dev'].map(to_en_format, remove_columns=dataset['dev'].column_names)
        dataset_ru_dev = dataset['dev'].map(to_ru_format, remove_columns=dataset['dev'].column_names)
        dataset_en_test = dataset['test'].map(to_en_format, remove_columns=dataset['test'].column_names)
        dataset_ru_test = dataset['test'].map(to_ru_format, remove_columns=dataset['test'].column_names)

        train_dataset = datasets.concatenate_datasets([dataset_en_train, dataset_ru_train])
        validation_dataset = datasets.concatenate_datasets([dataset_en_dev, dataset_ru_dev])
        test_dataset = datasets.concatenate_datasets([dataset_en_test, dataset_ru_test])


        #shuffle the datasets
        train_dataset = train_dataset.shuffle(seed=42)
        validation_dataset = validation_dataset.shuffle(seed=42)
        test_dataset = test_dataset.shuffle(seed=42)

        print(f"Converted dataset to two rows")

        if training_args.do_predict and args.predict_file is not None:
            custom_test_dataset = datasets.load_dataset('json', data_files={'test': args.predict_file})
            test_dataset = custom_test_dataset['test']
        else:
            test_dataset = test_dataset

        #limit the number of samples
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.max_eval_samples is not None:
            validation_dataset = validation_dataset.select(range(args.max_eval_samples))

        dataset = {
            "train": train_dataset,
            "validation": validation_dataset,
            "test": test_dataset,
        }

        eval_split = "validation"

    else:
        default_datasets = {'qa': ('squad',), 'nli': ('snli',)}
        dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else \
            default_datasets[args.task]
        # MNLI has two validation splits (one with matched domains and one with mismatched domains). Most datasets just have one "validation" split
        eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
        # Load the raw data
        dataset = datasets.load_dataset(*dataset_id)
    
    # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {'num_labels': 3} if args.task == 'nli' else {}

    # Here we select the right model fine-tuning head
    model_classes = {'qa': AutoModelForQuestionAnswering,
                     'nli': AutoModelForSequenceClassification}
    model_class = model_classes[args.task]
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained(args.model, **task_kwargs)
    # Make tensor contiguous if needed https://github.com/huggingface/transformers/issues/28293
    if hasattr(model, 'electra'):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", use_fast=True)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    if args.task == 'qa':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    elif args.task == 'nli':
        prepare_train_dataset = prepare_eval_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
        # prepare_eval_dataset = prepare_dataset_nli
    else:
        raise ValueError('Unrecognized task name: {}'.format(args.task))

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    if dataset_id == ('snli',):
        # remove SNLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
    train_dataset = None
    eval_dataset = None
    test_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    test_dataset_featurized = None

    if training_args.do_train:
        if 'train' in dataset:
            train_dataset = dataset['train']
            if args.max_train_samples and args.dataset != "facebook/xnli":
                train_dataset = train_dataset.select(range(args.max_train_samples))
    if training_args.do_eval:
        if eval_split in dataset:
            eval_dataset = dataset[eval_split]
            if args.max_eval_samples and args.dataset != "facebook/xnli":
                eval_dataset = eval_dataset.select(range(args.max_eval_samples))
                
    if training_args.do_predict:
        if 'test' in dataset:
            test_dataset = dataset['test']
            if args.max_eval_samples and args.dataset != "facebook/xnli":
                test_dataset = test_dataset.select(range(args.max_eval_samples))
        if training_args.do_predict and test_dataset is not None:
            test_dataset_featurized = test_dataset.map(
                prepare_eval_dataset,
                batched=True,
                num_proc=NUM_PREPROCESSING_WORKERS,
                remove_columns=['source'] if 'source' in test_dataset.column_names else []
            )

    if training_args.do_train and train_dataset is not None:
        print("Training dataset columns before mapping:", train_dataset.column_names)
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=['source'] if 'source' in train_dataset.column_names else []
        )
    if training_args.do_eval and eval_dataset is not None:
        print("Validation dataset columns before mapping:", eval_dataset.column_names)
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=['source'] if 'source' in eval_dataset.column_names else []
        )


    # Select the training configuration
    trainer_class = Trainer
    eval_kwargs = {}
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    #compute_metrics = None
    #if args.task == 'qa':
    #    # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
    #    # to enable the question-answering specific evaluation metrics
    #    trainer_class = QuestionAnsweringTrainer
    #    eval_kwargs['eval_examples'] = eval_dataset
    #    metric = evaluate.load('squad')   # datasets.load_metric() deprecated
    #    compute_metrics = lambda eval_preds: metric.compute(
    #        predictions=eval_preds.predictions, references=eval_preds.label_ids)
    #elif args.task == 'nli':
    #    compute_metrics = compute_accuracy
    

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    training_args.remove_unused_columns = False

    from transformers import DataCollatorWithPadding

    # Then right before initializing the trainer:
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, 
        padding=True,
        pad_to_multiple_of=8
    )

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=None,  # Remove tokenizer since we've already tokenized
        compute_metrics=compute_metrics_and_store_predictions,
    )

    # Train and/or evaluate
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # If you want to customize the way the loss is computed, you should subclass Trainer and override the "compute_loss"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss).
        #
        # You can also add training hooks using Trainer.add_callback:
        #   See https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.add_callback
        #   and https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback

    if training_args.do_predict:
        print("Evaluating on the test set...")

        test_dataset_en = test_dataset_featurized.filter(lambda x: x['language'] == 'en')
        test_dataset_ru = test_dataset_featurized.filter(lambda x: x['language'] == 'ru')

    #evaluate on English test set
        print("\nEvaluation on English test set:")
        results_en = trainer.evaluate(test_dataset_en)
        print(results_en)

    #evaluate on Russian test set
        print("\nEvaluation on Russian test set:")
        results_ru = trainer.evaluate(test_dataset_ru)
        print(results_ru)

    #evaluate on the entire test set
        print("\nEvaluation on the entire test set:")
        results_all = trainer.evaluate(test_dataset_featurized)
        print(results_all)

    #save the results to a JSON file
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(os.path.join(training_args.output_dir, 'test_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump({
                'English': results_en,
                'Russian': results_ru,
                'Combined': results_all
            }, f)

    #save misclassified examples
        def save_misclassified_examples(dataset, results, language_code):
            predictions = trainer.predict(dataset)
            logits = predictions.predictions
            labels = predictions.label_ids
            preds = np.argmax(logits, axis=-1)

            misclassified_indices = np.where(preds != labels)[0]
            misclassified_examples = dataset.select(misclassified_indices)

            output_file = os.path.join(training_args.output_dir, f'misclassified_{language_code}.jsonl')
            with open(output_file, 'w', encoding='utf-8') as f:
                for idx in range(len(misclassified_examples)):
                    example = misclassified_examples[idx]
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_label'] = int(preds[misclassified_indices[idx]])
                    example_with_prediction['true_label'] = int(labels[misclassified_indices[idx]])
                    example_with_prediction['predicted_scores'] = logits[misclassified_indices[idx]].tolist()
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
            print(f"Misclassified examples for {language_code} have been saved to {output_file}")

        save_misclassified_examples(test_dataset_en, results_en, 'English')
        save_misclassified_examples(test_dataset_ru, results_ru, 'Russian')


    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)

        # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.

        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)

        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            if args.task == 'qa':
                predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                for example in eval_dataset:
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
            else:
                for i, example in enumerate(eval_dataset):
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')


if __name__ == "__main__":
    main()
