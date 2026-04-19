import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scripts.parsing.vac_labelling import parse_skills


def concat_skills(raw_skills):
    return ', '.join([s for s in parse_skills(raw_skills)])

def prepare_text(df):
    return "Название: " + df['name'] + ". Требуемые навыки: " + df['raw_skills'].apply(concat_skills) + ". Описание: " + df['description']


def tokenize(batch):
    return tokenizer(batch['text'], padding="max_length", truncation=True, max_length=512)

model_name = "ai-forever/ruBert-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def run_model(df):
    
    df['text'] = prepare_text(df)
    dataset = Dataset.from_pandas(df[['text']])
    tokenized_dataset = dataset.map(tokenize, batched=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model = PeftModel.from_pretrained(base_model, "./trained_bert_model")
    inference_args = TrainingArguments(
        output_dir="./temp_inference",
        per_device_eval_batch_size=8,
        report_to="none",
    )
    
    inference_trainer = Trainer(
        model=model,
        args=inference_args,
    )
    pred_output = inference_trainer.predict(tokenized_dataset)
    ai_scores = np.squeeze(pred_output.predictions)
    df['ai_score'] = ai_scores
    
    return df


if __name__ == '__main__':
    train_df = pd.read_csv('data/train_bigger.csv')
    test_df = pd.read_csv('data/test_manual.csv')

    train_df['text'] = prepare_text(train_df)
    test_df['text'] = prepare_text(test_df)
    train_dataset = Dataset.from_pandas(train_df[['text', 'score']]).train_test_split(test_size=0.1)
    test_dataset = Dataset.from_pandas(test_df[['text', 'true_score']])



    train_set = train_dataset['train'].map(tokenize, batched=True).rename_column("score", "labels")
    val_set = train_dataset['test'].map(tokenize, batched=True).rename_column("score", "labels")
    test_set = test_dataset.map(tokenize, batched=True).rename_column("true_score", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["query", "value"], task_type="SEQ_CLS")
    model = get_peft_model(model, lora_config)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return {
            "mse": mean_squared_error(labels, logits),
            "r2": r2_score(labels, logits),
            "mae": mean_absolute_error(labels, logits)
        }

    args = TrainingArguments(
        output_dir="./bert_results",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=10,
        learning_rate=2e-4,
        per_device_train_batch_size=8
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_set, eval_dataset=val_set,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model("./trained_bert_model")
    tokenizer.save_pretrained("./trained_bert_model")
    print("Model and tokenizer saved to ./trained_bert_model")
    print("\n--- FINAL TEST EVALUATION (BERT) ---")
    results = trainer.evaluate(test_set)
    print(f"MSE: {results['eval_mse']:.4f}, R2: {results['eval_r2']:.4f}, MAE: {results['eval_mae']:.4f}")


