from src.TextSummarizer.config.configuration import ConfigurationManager
from transformers import T5Tokenizer
from transformers import pipeline

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        
    def predict(self, text):
        tokenizer = T5Tokenizer.from_pretrained(self.config.tokenizer_path)
        gen_kwargs = {"length_penalty": 1.0, "num_beams": 8, "max_length": 200}
        
        pipe = pipeline("summarization", model=self.config.model_checkpoint, tokenizer=tokenizer)
        
        print("Dialogue")
        print(text)
        
        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        print("\nModel Summary:")
        print(output)
        
        return output   
        