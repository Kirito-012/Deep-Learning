from transformers import pipeline

classifier = pipeline('zero-shot-classification')

res = classifier(
    "Wish i could be with you!",
    candidate_labels = ["Education", "Politics", "Love"]
)

print(res)
