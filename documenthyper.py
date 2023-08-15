from transformers import BertForQuestionAnswering, AutoTokenizer, pipeline
import random
input_file="input.txt"
# Define the hyperparameter search space for the tokenizer
tokenizer_search_space = {
    'max_length': [128, 256],
    'truncation': [True, False],
    'padding': [True, False]
}

# Define the hyperparameter search space for the model
model_search_space = {
    'num_train_epochs': [2, 3, 4],
    'per_device_train_batch_size': [16, 32],
}

# Function to randomly sample from the search space for the tokenizer
def sample_tokenizer_hyperparameters():
    return {param_name: random.choice(values) for param_name, values in tokenizer_search_space.items()}

# Function to randomly sample from the search space for the model
def sample_model_hyperparameters():
    return {param_name: random.choice(values) for param_name, values in model_search_space.items()}

# Function to evaluate the model with given hyperparameters
def evaluate_model_with_hyperparameters(model_hyperparameters, tokenizer_hyperparameters):
    model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
    tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2', **tokenizer_hyperparameters)

    with open(input_file, "r") as file:
        context = file.read()

    question = "what is politics"
    tokenizer.encode(question, truncation=True, padding=True)
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    result = nlp({
        'question': question,
        'context': context
    })

    return result['score']

# Perform random search for 10 iterations
best_score = -1
best_model_hyperparameters = None
best_tokenizer_hyperparameters = None

for i in range(10):
    model_hyperparameters = sample_model_hyperparameters()
    tokenizer_hyperparameters = sample_tokenizer_hyperparameters()
    score = evaluate_model_with_hyperparameters(model_hyperparameters, tokenizer_hyperparameters)
    print(f"Iteration {i + 1}: Model Hyperparameters: {model_hyperparameters}, Tokenizer Hyperparameters: {tokenizer_hyperparameters}, Score: {score}")
    
    if score > best_score:
        best_score = score
        best_model_hyperparameters = model_hyperparameters
        best_tokenizer_hyperparameters = tokenizer_hyperparameters

print("Best Model Hyperparameters:", best_model_hyperparameters)
print("Best Tokenizer Hyperparameters:", best_tokenizer_hyperparameters)
