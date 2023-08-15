import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Assuming you have defined `keywords` somewhere above
keywords=["fest"]
num = len(keywords)
model_name = 'bert-base-uncased'
labels=keywords
num_labels = num  # Number of classification labels
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


# Function to split text into paragraphs
def split_into_paragraphs(text, delimiter="\n"):
    return text.split(delimiter)

# Function to calculate reliability score for a given label
def calculate_reliability_score(probs, label):
    label_index = labels.index(label)
    return probs[label_index]

# Read text from the input file
input_file = "input.txt"
with open(input_file, "r") as file:
    text = file.read()
# print(text)
# Split text into paragraphs
paragraphs = split_into_paragraphs(text)
# print(paragraphs)

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
    # print(paragraph)

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
        # print("para here")
        # print(type(paragraph))
        related_paragraphs.append(paragraph)

# Filter paragraphs that contain the given keywords
keywords = set(keywords)  # Convert to a set for faster lookup
related_paragraphs_with_keywords = [para for para in related_paragraphs if any(keyword in para.lower() for keyword in keywords)]

# Print the most reliable label and its related paragraphs with keywords
print("Most Reliable Label:", most_reliable_label)
print("Reliability Score:", max_reliability_score)
print("related data:")
print(related_paragraphs)
print("Related Paragraphs with Keywords:")
print("\n".join(related_paragraphs_with_keywords))


# # Function to split text into paragraphs most relatable
# def split_into_paragraphs(text, delimiter="\n\n"):
#     return text.split(delimiter)

# # Function to calculate reliability score for a given label
# def calculate_reliability_score(probs, label):
#     label_index = labels.index(label)
#     return probs[label_index]

# # Read text from the input file
# input_file = "input.txt"
# with open(input_file, "r", encoding="utf-8") as file:
#     text = file.read()

# # Split text into paragraphs
# paragraphs = split_into_paragraphs(text)

# # Initialize variables to store the most reliable label and its score
# most_reliable_label = None
# max_reliability_score = 0.0
# related_paragraphs = []

# # Process each paragraph and find the most reliable label
# for paragraph in paragraphs:
#     # Tokenize input text
#     tokens = tokenizer.encode_plus(
#         paragraph,
#         max_length=128,  # Max sequence length for BERT
#         padding='max_length',
#         truncation=True,
#         return_tensors='pt'  # Return PyTorch tensors
#     )

#     # Forward pass through the model
#     outputs = model(**tokens)

#     # Get predicted probabilities and labels
#     probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()[0]
#     predicted_label = labels[torch.argmax(outputs.logits, dim=1).item()]

#     # Calculate the reliability score for the predicted label
#     reliability_score = calculate_reliability_score(probs, predicted_label)

#     # Update most reliable label if the current one has a higher score
#     if reliability_score > max_reliability_score:
#         max_reliability_score = reliability_score
#         most_reliable_label = predicted_label
#         related_paragraphs = [paragraph]
#     elif reliability_score == max_reliability_score and predicted_label == most_reliable_label:
#         related_paragraphs.append(paragraph)

# # Print the most reliable label and its related paragraphs
# print("Most Reliable Label:", most_reliable_label)
# print("Reliability Score:", max_reliability_score)
# print("Related Paragraphs:")
# print("\n".join(related_paragraphs))




# # Function to split text into paragraphs
# def split_into_paragraphs(text, delimiter="."):
#     return text.split(delimiter)

# # Function to calculate reliability score for a given label
# def calculate_reliability_score(probs, label):
#     label_index = labels.index(label)
#     return probs[label_index]

# # Read text from the input file
# input_file = "input.txt"
# with open(input_file, "r", encoding="utf-8") as file:
#     text = file.read()

# # Split text into paragraphs
# paragraphs = split_into_paragraphs(text)

# # Initialize variables to store the most reliable label and its score
# most_reliable_label = None
# max_reliability_score = 0.0

# # Process each paragraph and find the most reliable label
# for paragraph in paragraphs:
#     # Tokenize input text
#     tokens = tokenizer.encode_plus(
#         paragraph,
#         max_length=128,  # Max sequence length for BERT
#         padding='max_length',
#         truncation=True,
#         return_tensors='pt'  # Return PyTorch tensors
#     )

#     # Forward pass through the model
#     outputs = model(**tokens)

#     # Get predicted probabilities and labels
#     probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()[0]
#     predicted_label = labels[torch.argmax(outputs.logits, dim=1).item()]

#     # Calculate the reliability score for the predicted label
#     reliability_score = calculate_reliability_score(probs, predicted_label)

#     # Update most reliable label if the current one has a higher score
#     if reliability_score > max_reliability_score:
#         max_reliability_score = reliability_score
#         most_reliable_label = predicted_label

# # Print the most reliable label for the entire text
# print("Most Reliable Label:", most_reliable_label)
# print("Reliability Score:", max_reliability_score)
###respective


# num = len(keywords)
# model_name = 'bert-base-uncased'
# num_labels = num  # Number of classification labels
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# # Function to split text into paragraphs
# def split_into_paragraphs(text, delimiter="."):
#     return text.split(delimiter)

# # Function to calculate reliability score for a given label
# def calculate_reliability_score(probs, label):
#     label_index = labels.index(label)
#     return probs[label_index]

# # Read text from the input file
# input_file = "input.txt"
# with open(input_file, "r", encoding="utf-8") as file:
#     text = file.read()

# # Split text into paragraphs
# paragraphs = split_into_paragraphs(text)

# # Initialize variables to store the most reliable label and its score for each keyword
# keyword_related_paragraphs = {keyword: {'label': None, 'score': 0.0, 'paragraphs': []} for keyword in keywords}

# # Process each paragraph and find the most reliable label for each keyword
# for paragraph in paragraphs:
#     # Tokenize input text
#     tokens = tokenizer.encode_plus(
#         paragraph,
#         max_length=128,  # Max sequence length for BERT
#         padding='max_length',
#         truncation=True,
#         return_tensors='pt'  # Return PyTorch tensors
#     )

#     # Forward pass through the model
#     outputs = model(**tokens)

#     # Get predicted probabilities and labels
#     probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()[0]
#     predicted_label = labels[torch.argmax(outputs.logits, dim=1).item()]

#     # Check if the predicted label matches any of the keywords
#     if predicted_label in keywords:
#         # Calculate the reliability score for the predicted label
#         reliability_score = calculate_reliability_score(probs, predicted_label)

#         # Update most reliable label if the current one has a higher score
#         if reliability_score > keyword_related_paragraphs[predicted_label]['score']:
#             keyword_related_paragraphs[predicted_label]['label'] = predicted_label
#             keyword_related_paragraphs[predicted_label]['score'] = reliability_score

#         # Add the paragraph to the list of related paragraphs for the keyword
#         keyword_related_paragraphs[predicted_label]['paragraphs'].append(paragraph)

# # Print the related paragraphs for each keyword
# for keyword, related_info in keyword_related_paragraphs.items():
#     print("Keyword:", keyword)
#     print("Most Reliable Label:", related_info['label'])
#     print("Reliability Score:", related_info['score'])
#     print("Related Paragraphs:")
#     print("\n".join(related_info['paragraphs']))
#     print("-" * 30)


