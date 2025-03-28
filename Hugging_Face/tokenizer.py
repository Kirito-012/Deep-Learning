from transformers import AutoTokenizer

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
sequence = "I am ran over the transformer and the fox is running after me."

res = tokenizer(sequence)
print(res, "\n")

tokens = tokenizer.tokenize(sequence)
print(tokens, "\n")

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids, "\n")

decoded_string = tokenizer.decode(ids)
print(decoded_string)