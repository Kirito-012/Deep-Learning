from transformers import pipeline

classifier = pipeline('sentiment-analysis')

res = classifier("My eyes long for you, But ahh! This universe won't let us be together")

print(res)