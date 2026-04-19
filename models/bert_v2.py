import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scripts.parsing.vac_labelling import parse_skills

def concat_skills(raw_skills):
    return ', '.join([s for s in parse_skills(raw_skills)])

def prepare_text(df):
    return "Название: " + df['name'] + ". Требуемые навыки: " + df['raw_skills'].apply(concat_skills) + ". Описание: " + df['description']

model_name = "ai-forever/ruBert-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch['text'], padding="max_length", truncation=True, max_length=512)
class TestEvalCallback(TrainerCallback):
    def __init__(self, trainer_ref, test_dataset):
        self.trainer_ref = trainer_ref
        self.test_dataset = test_dataset
        self.epoch_history = []
        self.test_r2_history = []

    def on_epoch_end(self, args, state, control, **kwargs):
        pred_output = self.trainer_ref.predict(self.test_dataset, metric_key_prefix="test")
        
        self.epoch_history.append(float(state.epoch))
        self.test_r2_history.append(pred_output.metrics["test_r2"])
        return control


if __name__ == '__main__':
    train_df = pd.read_csv('data/train_bigger.csv')
    test_df = pd.read_csv('data/test_manual.csv')

    train_df['text'] = prepare_text(train_df)
    test_df['text'] = prepare_text(test_df)
    train_dataset = Dataset.from_pandas(train_df[['text', 'score']]).train_test_split(test_size=0.1, seed=42)
    test_dataset = Dataset.from_pandas(test_df[['text', 'true_score']])
    train_set = train_dataset['train'].map(tokenize, batched=True).rename_column("score", "labels")
    val_set = train_dataset['test'].map(tokenize, batched=True).rename_column("score", "labels")
    test_set = test_dataset.map(tokenize, batched=True).rename_column("true_score", "labels")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["query", "key", "value", "dense"], # Расширенный таргет
        lora_dropout=0.1, 
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_config)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.squeeze(logits)
        return {
            "mse": mean_squared_error(labels, preds),
            "r2": r2_score(labels, preds),
            "mae": mean_absolute_error(labels, preds)
        }
    args = TrainingArguments(
        output_dir="./bert_results",
        eval_strategy="epoch",           
        save_strategy="epoch",           
        load_best_model_at_end=True,
        metric_for_best_model="eval_r2", 
        greater_is_better=True,          
        num_train_epochs=15,           
        learning_rate=3e-4,            
        weight_decay=0.01,
        warmup_ratio=0.1,              
        lr_scheduler_type="cosine",    
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2, 
        per_device_eval_batch_size=8,
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=model, 
        args=args,
        train_dataset=train_set, 
        eval_dataset=val_set,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
    )
    test_callback = TestEvalCallback(trainer, test_set)
    trainer.add_callback(test_callback)
    print("=== START TRAINING RUBERT + LORA ===")
    trainer.train()
    trainer.save_model("./trained_bert_model")
    tokenizer.save_pretrained("./trained_bert_model")
    print("Best model and tokenizer saved to ./trained_bert_model")
    print("\n--- FINAL TEST EVALUATION (BEST CHECKPOINT) ---")
    final_pred_output = trainer.predict(test_set, metric_key_prefix="final_test")
    final_results = final_pred_output.metrics
    print(f"MSE: {final_results['final_test_mse']:.4f}, R2: {final_results['final_test_r2']:.4f}, MAE: {final_results['final_test_mae']:.4f}")

    print("\nГенерация графиков...")
    val_epochs = []
    val_r2 = []
    
    for log in trainer.state.log_history:
        if "eval_r2" in log and "epoch" in log:
            val_epochs.append(float(log["epoch"]))
            val_r2.append(log["eval_r2"])
    plt.figure(figsize=(9, 6))
    plt.plot(val_epochs, val_r2, marker='o', linestyle='-', linewidth=2, label='Validation $R^2$', color='#1f77b4')
    plt.plot(test_callback.epoch_history, test_callback.test_r2_history, marker='s', linestyle='--', linewidth=2, label='Test $R^2$', color='#ff7f0e')
    if val_r2:
        best_epoch_idx = np.argmax(val_r2)
        best_epoch = val_epochs[best_epoch_idx]
        best_val_score = val_r2[best_epoch_idx]
        
        plt.axvline(x=best_epoch, color='red', linestyle=':', alpha=0.6, label=f'Best Checkpoint (Epoch {best_epoch:.1f})')
        plt.scatter(best_epoch, best_val_score, color='red', s=100, zorder=5) # Точка максимума
    
    plt.xlabel('Эпоха (Epoch)', fontsize=12)
    plt.ylabel('Коэффициент детерминации ($R^2$)', fontsize=12)
    plt.title('Динамика $R^2$ в процессе дообучения ruBERT (LoRA)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('rubert_lora_learning_curve.png', dpi=300)
    final_preds = np.squeeze(final_pred_output.predictions)
    true_vals = test_df['true_score'].values
    plt.figure(figsize=(7, 7))
    plt.scatter(true_vals, final_preds, alpha=0.6, color='seagreen', edgecolor='k')
    plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Идеальный прогноз (Y=X)')
    
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Истинные значения (True AI-score)", fontsize=12)
    plt.ylabel("Предсказания ruBERT + LoRA", fontsize=12)
    plt.title("Качество предсказания лучшего чекпоинта на Test Set", fontsize=13)
    textstr = '\n'.join((
        f"$R^2$ = {final_results['final_test_r2']:.3f}",
        f"MAE = {final_results['final_test_mae']:.3f}",
        f"MSE = {final_results['final_test_mse']:.3f}"
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('rubert_lora_scatter.png', dpi=300)
