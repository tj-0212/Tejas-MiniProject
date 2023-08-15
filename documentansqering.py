# import transformers
# from transformers import pipeline
# from transformers import BertForQuestionAnswering
# from transformers import AutoTokenizer
# model= BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
# tokenizer=AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')
# input_file = "input.txt"
# with open(input_file, "r") as file:
#     text = file.read()
# context=text
# question=[
#     "what is politics"

# ]
# tokenizer.encode(question[0],truncation=True,padding=True)
# nlp=pipeline('question-answering',model=model,tokenizer=tokenizer)
# result=nlp({
#     'question':question[0],
#      'context':context
# })
# print(result['answer'])
import transformers
from transformers import pipeline
from transformers import BertForQuestionAnswering
from transformers import AutoTokenizer

model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')

input_file = "input.txt"
with open(input_file, "r") as file:
    text = file.read()

context = text
question = [
    "what is fest"
]

tokenizer.encode(question[0], truncation=True, padding=True)
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
result = nlp({
    'question': question[0],
    'context': context
})

# Expand the context window to include more sentences before and after the answer
context_start = max(0, result['start'] - 100)
context_end = min(len(context), result['end'] + 150)
expanded_context = context[context_start:context_end]
from transformers import pipeline

# Load the GPT-2 model for text generation (smaller version compared to GPT-3.5)
text_generator_gpt2 = pipeline("text-generation", model="gpt2")

# Input text with errors


# Generate corrected text using GPT-2
corrected_text_gpt2 = text_generator_gpt2(expanded_context, max_length=70, do_sample=True, temperature=100)[0]['generated_text']



# API_TOKEN=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
# import requests

# API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"
# headers = {"Authorization": f"Bearer {API_TOKEN}"}

# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.json()
	
# output = query({
# 	"inputs": "Can you please let us know more details about your ",
# # })
# # Load the GPT-3.5 model for text generation
# text_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

# # Input text with errors
# input_text = expanded_context

# # Generate corrected text using GPT-3.5
# corrected_text = text_generator(input_text, max_length=100, do_sample=True, temperature=0.7)[0]['generated_text']

# # print("Corrected Text:")
# print(corrected_text)

print("Question:", question[0])
print("Answer:", result['answer'])
print("Expanded Context:",corrected_text_gpt2)
