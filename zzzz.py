import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import threading
import torch.nn as nn
from transformers import BertForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
path="input.txt"
import pandas as pd
from transformers import BertTokenizer,BertModel
keywords=[]
c=[]
e=[]
d=[]
class BERTClass(torch.nn.Module):
        def __init__(self):
            super(BERTClass, self).__init__()
            self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
            self.dropout = torch.nn.Dropout(0.3)
            self.linear = torch.nn.Linear(768, 3)

        def forward(self, input_ids, attn_mask, token_type_ids):
            output = self.bert_model(
                input_ids,
                attention_mask=attn_mask,
                token_type_ids=token_type_ids
            )
            output_dropout = self.dropout(output.pooler_output)
            output = self.linear(output_dropout)
            return output
def CLassify():
    test_path=r"C:\Users\Admin\Desktop\MINI-Project\samplemeetingtest.csv"
    ecount=0
    ccount=0
    dcount=0
    sum=0
    test_df = pd.read_csv(test_path)
    test_df = test_df.drop(test_df.iloc[:, 4:33],axis = 1)
    print(test_df.head())
    target_list=['College Fest', 'Exams and Curriculum',
        'Discipline and Students']
    # hyperparameters
    MAX_LEN = 256
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 16
    EPOCHS = 2
    LEARNING_RATE = 1e-05
    checkpoint_path = r'C:\Users\Admin\Desktop\MINI-Project\best_model.pt'
    model = torch.load(checkpoint_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    paragraphs,sum,text=parawise(path)
    for paragraph in paragraphs:
        encodings = tokenizer.encode_plus(
            paragraph,
            None,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        model.eval()
        with torch.no_grad():
            input_ids = encodings['input_ids'].to(dtype=torch.long)
            attention_mask = encodings['attention_mask'].to(dtype=torch.long)
            token_type_ids = encodings['token_type_ids'].to(dtype=torch.long)
            output = model(input_ids, attention_mask, token_type_ids)
            final_output = torch.sigmoid(output).cpu().detach().numpy().tolist()
            print(target_list[int(np.argmax(final_output, axis=1))])
            data=target_list[int(np.argmax(final_output, axis=1))]
            target_list=['College Fest', 'Exams and Curriculum',
        'Discipline and Students']
            if target_list[0] in data:
                c.append(data)
            if target_list[1] in data:
                e.append(data)    
            if target_list[2] in data:
                d.append(data)
    for i in range(len(d)):
            dcount=dcount+1
    for i in range(len(c)):
            ccount=ccount+1
    for i in range(len(e)):
            ecount=ecount+1 
    print("****Classification******")                           
    print(ccount/sum)
    print(ecount/sum) 
    print(dcount/sum)
    import matplotlib.pyplot as plt

    def visualize_percentage(labels, percentages, save_path=None):
        # Define custom colors for the pie chart
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        # Create a pie chart
        plt.figure(figsize=(8, 8))
        explode = [0.05] * len(labels)  # Explode each slice for a "popping out" effect
        plt.pie(percentages, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140, explode=explode,
                shadow=True, wedgeprops=dict(width=0.4, edgecolor='w'))

        # Add a title and a legend
        plt.title("Percentage of Data")
        plt.legend(labels, loc="best")

        # Display the chart
        plt.axis("equal")

        # Save the chart if save_path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        # Show or save the chart depending on save_path
        if save_path:
            plt.close()  # Close the figure if saving to file
        else:
            plt.show()

    if __name__ == "__main__":
        
        labels =['College Fest', 'Exams and Curriculum',
        'Discipline and Students']
        percentages = [ccount/sum, ecount/sum, dcount/sum]

    save_path = "pie_chart.png"  # Provide the desired path and filename to save the chart
    visualize_percentage(labels, percentages, save_path)     


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
    return paragraphs,sum,text
    # print(paragraphs)
def search(text):
    sum=0

    print("Search key: Fest")
    keywords=[]
    keywords.append(text)
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
def summary():
    #from transformers import pipeline
    # sum=0
    # summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # paragraphs,sum,text=parawise(path)
    # # for paragraph in paragraphs:
    # print("para")
    # # print(paragraphs)
    # # print("text")
    # # print(text)
    # print("Summary:")
    # sumdata=summarizer(text, max_length=200, min_length=100, do_sample=True)
    # print(sumdata[0]['summary_text'])
    from transformers import pipeline

    def post_process_summary(summary):
        summary = summary.replace('\n', ' ')  # Remove newlines
        summary = ' '.join(summary.split())  # Remove excessive whitespace
        return summary

    def generate_summary(text):
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(text, max_length=250, min_length=100, do_sample=False)
        summary_text = post_process_summary(summary[0]['summary_text'])
        return summary_text

    # Assuming you have defined 
    paragraphs, sum, text = parawise(path)
    sumdata = generate_summary(text)
    print("Summary:")
    print(sumdata)

   
    
def questioning(text):
        model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
        tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')

        input_file = "input.txt"
        with open(input_file, "r") as file:
            text = file.read()

        context = text

        question = []
        question.append(text)

        tokenizer.encode(question[0], truncation=True, padding=True)
        nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
        result = nlp({
            'question': question[0],
            'context': context
        })
        context_start = max(0, result['start'] - 100)
        context_end = min(len(context), result['end'] + 150)
        expanded_context = context[context_start:context_end]   


        print("Question:", question[0])
        print("Answer:", result['answer'])
        print("Expanded Context:", expanded_context)
        # print("Expanded Context:", expanded_context1)
# parawise(path)
# def thread_function(target_function):
#     thread = threading.Thread(target=target_function)
#     thread.start()
#     thread.join()
# thread_classify = threading.Thread(target=CLassify)
# thread_summary = threading.Thread(target=summary)

# # Start the threads
# # thread_function(CLassify)
# # thread_function(search)
# thread_function(summary)
# CLassify()
summary()
