import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import torch.nn as nn
from transformers import BertForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
path="input.txt"
import pandas as pd
from transformers import BertTokenizer,BertModel
def search():
    sum=0

    print("Search key: Fest")
    keywords=["fest"]
    num = len(keywords)
    model_name = 'bert-base-uncased'
    labels=keywords
    num_labels = num  # Number of classification labels
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    paragraphs,sum=parawise(path)

  

    # Function to calculate reliability score for a given label
    def calculate_reliability_score(probs, label):
            label_index = labels.index(label)
            return probs[label_index]

    # Initialize variables to store the most reliable label and its score
    most_reliable_label = None
    max_reliability_score = 0.0
    related_paragraphs = []           
        
    # Process each paragraph and find the most reliable label
    for paragraph in paragraphs:
        # Tokenize input text
        tokens = tokenizer.encode_plus(
            paragraph,
            max_length=128,  # Max sequence length for BERT
            padding='max_length',
            truncation=True,
            return_tensors='pt'  # Return PyTorch tensors
        )
        
        # Forward pass through the model
        outputs = model(**tokens)

        # Get predicted probabilities and labels
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()[0]
        predicted_label = labels[torch.argmax(outputs.logits, dim=1).item()]

        # Calculate the reliability score for the predicted label
        reliability_score = calculate_reliability_score(probs, predicted_label)

        # Update most reliable label if the current one has a higher score
        if reliability_score > max_reliability_score:
            max_reliability_score = reliability_score
            most_reliable_label = predicted_label
            related_paragraphs = [paragraph]
        if reliability_score == max_reliability_score and predicted_label == most_reliable_label:

            related_paragraphs.append(paragraph)

        # Filter paragraphs that contain the given keywords
    keywords = set(keywords)  # Convert to a set for faster lookup
    related_paragraphs_with_keywords = [para for para in related_paragraphs if any(keyword in para.lower() for keyword in keywords)]

        # Print the most reliable label and its related paragraphs with keywords
    print("Most Reliable Label:", most_reliable_label)
    print("Reliability Score:", max_reliability_score)
    print("related data:")
    # print(related_paragraphs)
    print("Related Paragraphs with Keywords:")
    print("\n".join(related_paragraphs_with_keywords))

def parawise(path):
    def split_into_paragraphs(text, delimiter="."):
        return text.split(delimiter)
    sum=0
    # Read text from the input file
    input_file = path
    with open(input_file, "r") as file:
        text = file.read()
    # print(text)
    # Split text into paragraphs
    paragraphs = split_into_paragraphs(text)
    for paragraph in paragraphs:
        #  print(paragraph)
         sum=sum+1
        #  print("***********")
    return paragraphs,sum        
search()