'''To make a base model behave like an instruct model, 
we need to format our prompts in a consistent way that the model can understand'''

from transformers import AutoTokenizer

# These will use different templates automatically
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat")
smol_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]

# Each will format according to its model's template
mistral_chat = mistral_tokenizer.apply_chat_template(messages, tokenize=False)
qwen_chat = qwen_tokenizer.apply_chat_template(messages, tokenize=False)
smol_chat = smol_tokenizer.apply_chat_template(messages, tokenize=False)