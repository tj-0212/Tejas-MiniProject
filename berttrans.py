import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
num_labels = 2  # Number of classification labels
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Define input text
text = "there will be college fest interact held in the month of july 13th,14th and 15th"

# Tokenize input text
tokens = tokenizer.encode_plus(
    text,
    max_length=128,  # Max sequence length for BERT
    padding='max_length',
    truncation=True,
    return_tensors='pt'  # Return PyTorch tensors
)

# Forward pass through the model
outputs = model(**tokens)

# Get predicted probabilities and labels
probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()[0]
predicted_label = torch.argmax(outputs.logits, dim=1).item()

# Define labels
labels = ["fest", "date"]

# Print results
print("Text:", text)
print("Predicted Label:", labels[predicted_label])
print("Label Probabilities:", probs)

