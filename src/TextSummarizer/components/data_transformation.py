import os
from src.TextSummarizer.logging import logger
from transformers import T5Tokenizer
from datasets import load_dataset, load_from_disk
from src.TextSummarizer.entity import DataTransformationConfig

## Data Tranformation component
class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = T5Tokenizer.from_pretrained(config.tokenizer_name)
    
    ## Creating object to convert text to feature
    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(example_batch['dialogue'], max_length = 1024, truncation = True, padding="max_length", return_tensors='pt', return_attention_mask=True)
        
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch['summary'], max_length=128, truncation=True, padding="max_length", return_tensors='pt', return_attention_mask=True)
        
        labels = target_encodings['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        print("Tokenizer vocab size:", self.tokenizer.vocab_size)
        
        print("input_ids max:", input_encodings['input_ids'].max())
        print("input_ids min:", input_encodings['input_ids'].min())

        print("labels max:", labels.max())
        print("labels min:", labels.min())
        
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': labels
            
        }    
        
    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, "samsum_dataset"))    
                