from transformers import pipeline
import torch
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
print(classifier(
    ["I am struggling to find a good job over Linkedin.",
     "I am a PHd Student and I'm feeling it.",
     "I am a PHd Student and I'm loving it."]))

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = ["I am struggling to find a good job over Linkedin.",
     "I am a PHd Student and I'm feeling it.",
     "I am a PHd Student and I'm loving it."]

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
print(model.config.id2label)