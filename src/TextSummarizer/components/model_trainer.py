from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset, load_from_disk
from src.TextSummarizer.entity import ModelTrainerConfig
import torch
import os

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        
    ## Creating object for importing and training model    
    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = T5Tokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = T5ForConditionalGeneration.from_pretrained(self.config.model_ckpt).to(device)
        model_pegasus.gradient_checkpointing_enable()
        print("Model vocab size:", model_pegasus.config.vocab_size)


        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
        ## Loading data
        dataset_samsum_pt = load_from_disk(self.config.data_path)
        print(dataset_samsum_pt)
        
        
        trainer_args = TrainingArguments(
            output_dir = 'distilbart-samsum', num_train_epochs = 3, warmup_steps = 500,
            per_device_train_batch_size = 1, per_device_eval_batch_size = 1,
            weight_decay = 0.01, logging_steps = 20,
            evaluation_strategy = 'steps', eval_steps = 500, save_steps = 500,
            gradient_accumulation_steps = 16,save_total_limit=3, 
            fp16=True, save_strategy="steps"
            ) 
        
        trainer = Trainer(model=model_pegasus, args = trainer_args, 
                          tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                          train_dataset=dataset_samsum_pt['train'],
                          eval_dataset=dataset_samsum_pt['validation'])
        
        # trainer.train()    normal train
        trainer.train(resume_from_checkpoint = "distilbart-samsum/checkpoint-2760")
        
        # Save model
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "Distilbert-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))   
        
        