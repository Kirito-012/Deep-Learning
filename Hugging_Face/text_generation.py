from transformers import pipeline

classifier = pipeline("text-generation")

res = classifier(
    "Wish i could be with you, but",
    num_return_sequences = 2,
)

print(res)

# [
# {'generated_text': "Wish i could be with you, but I don't like your name... I think you are a good enough person 
#   to know why you want to kill me. I want to win. I'm not stupid. If you let me, I'm"}, 
# {'generated_text': "Wish i could be with you, but this is soooo important, my father would not have to go to 
#   jail if I didn't have it all in me, so much more than with me being kidnapped by evil forces, and not being 
#   free"}
# ]