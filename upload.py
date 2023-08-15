from flask import Flask, render_template, request
import os

app = Flask(__name__)

app.config["UPLOAD_DIR"] = "uploads"
@app.route("/", methods = ["GET", "POST"])
def upload():
    if request.method == 'POST':
        for file in request.files.getlist('file'):
             file.save(os.path.join(app.config['UPLOAD_DIR'], file.filename))
        return render_template("upload.html", msg = "Files uplaoded successfully.")

    return render_template("upload.html", msg = "")
    ##get file 
                    # import torch
                    # from transformers import BertTokenizer, BertForSequenceClassification
                    # import numpy as np
                    # import torch.nn as nn
                    # from transformers import BertForQuestionAnswering
                    # from transformers import AutoTokenizer, AutoModelForTokenClassification
                    # from transformers import pipeline

                    # import pandas as pd
                    # from transformers import BertTokenizer,BertModel
                    # # from untitled13 import BertModel  # Import your custom BERTClass from the original file where you defined the model architecture
                    # # model = BERTClass()
                    # keywords=[]
                    # c=[]
                    # e=[]
                    # d=[]
                    # def CLassify():
                    #     test_path=r"C:\Users\Admin\Desktop\MINI-Project\samplemeetingtest.csv"
                    #     ecount=0
                    #     ccount=0
                    #     dcount=0
                    #     sum=0
                    #     test_df = pd.read_csv(test_path)



                    #     test_df = test_df.drop(test_df.iloc[:, 4:33],axis = 1)
                    #     print(test_df.head())


                    #     target_list=['College Fest', 'Exams and Curriculum',
                    #         'Discipline and Students']

                    #     # hyperparameters
                    #     MAX_LEN = 256
                    #     TRAIN_BATCH_SIZE = 16
                    #     VALID_BATCH_SIZE = 16
                    #     EPOCHS = 2
                    #     LEARNING_RATE = 1e-05

                    #     class BERTClass(torch.nn.Module):
                    #         def __init__(self):
                    #             super(BERTClass, self).__init__()
                    #             self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
                    #             self.dropout = torch.nn.Dropout(0.3)
                    #             self.linear = torch.nn.Linear(768, 3)

                    #         def forward(self, input_ids, attn_mask, token_type_ids):
                    #             output = self.bert_model(
                    #                 input_ids,
                    #                 attention_mask=attn_mask,
                    #                 token_type_ids=token_type_ids
                    #             )
                    #             output_dropout = self.dropout(output.pooler_output)
                    #             output = self.linear(output_dropout)
                    #             return output
                    #     checkpoint_path = r'C:\Users\Admin\Desktop\MINI-Project\best_model.pt'
                    #     # checkpoint = torch.load(checkpoint_path)
                    #     # model.load_state_dict(checkpoint['state_dict'])
                    #     model = torch.load(checkpoint_path)

                    #     # from untitled13 import test_df,MAX_LEN,target_list
                    #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                    #     paragraphs,sum=parawise()
                    #     for paragraph in paragraphs:
                    #         encodings = tokenizer.encode_plus(
                    #             paragraph,
                    #             None,
                    #             add_special_tokens=True,
                    #             max_length=MAX_LEN,
                    #             padding='max_length',
                    #             return_token_type_ids=True,
                    #             truncation=True,
                    #             return_attention_mask=True,
                    #             return_tensors='pt'
                    #         )
                    #         model.eval()
                    #         with torch.no_grad():
                    #             input_ids = encodings['input_ids'].to(dtype=torch.long)
                    #             attention_mask = encodings['attention_mask'].to(dtype=torch.long)
                    #             token_type_ids = encodings['token_type_ids'].to(dtype=torch.long)
                    #             output = model(input_ids, attention_mask, token_type_ids)
                    #             final_output = torch.sigmoid(output).cpu().detach().numpy().tolist()
                    #             print(target_list[int(np.argmax(final_output, axis=1))])
                    #             data=target_list[int(np.argmax(final_output, axis=1))]
                    #             target_list=['College Fest', 'Exams and Curriculum',
                    #         'Discipline and Students']
                    #             if target_list[0] in data:
                    #                 c.append(data)
                    #             if target_list[1] in data:
                    #                 e.append(data)    
                    #             if target_list[2] in data:
                    #                 d.append(data)
                    #             for i in range(len(d)):
                    #                     dcount=dcount+len(d[i])
                    #             for i in range(len(c)):
                    #                     ccount=ccount+len(c[i])
                    #             for i in range(len(e)):
                    #                     ecount=ecount+len(e[i])                    
                    #             print(ccount/sum)
                    #             print(ecount/sum) 
                    #             print(dcount/sum)
                    #import matplotlib.pyplot as plt

                        # def visualize_percentage(labels, percentages, save_path=None):
                        #     # Define custom colors for the pie chart
                        #     colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

                        #     # Create a pie chart
                        #     plt.figure(figsize=(8, 8))
                        #     explode = [0.05] * len(labels)  # Explode each slice for a "popping out" effect
                        #     plt.pie(percentages, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140, explode=explode,
                        #             shadow=True, wedgeprops=dict(width=0.4, edgecolor='w'))

                        #     # Add a title and a legend
                        #     plt.title("Percentage of Data")
                        #     plt.legend(labels, loc="best")

                        #     # Display the chart
                        #     plt.axis("equal")

                        #     # Save the chart if save_path is provided
                        #     if save_path:
                        #         plt.savefig(save_path, bbox_inches="tight")

                        #     # Show or save the chart depending on save_path
                        #     if save_path:
                        #         plt.close()  # Close the figure if saving to file
                        #     else:
                        #         plt.show()

                        # if __name__ == "__main__":
                        #     # Sample data - replace with your actual data
                        #     labels = ["Label A", "Label B", "Label C", "Label D"]
                        #     percentages = [25, 30, 20, 25]

                        #     save_path = "pie_chart.png"  # Provide the desired path and filename to save the chart
                        #     visualize_percentage(labels, percentages, save_path)   


                    # def parawise(path):
                    #     def split_into_paragraphs(text, delimiter="\n" or "."):
                    #         return text.split(delimiter)
                    #     sum=0
                    #     # Read text from the input file
                    #     input_file = path
                    #     with open(input_file, "r") as file:
                    #         text = file.read()
                    #     # print(text)
                    #     # Split text into paragraphs
                    #     paragraphs = split_into_paragraphs(text)
                    #     total=len(paragraphs)
                    #     for i in range(total):
                    #         sum=sum+len(paragraphs[i])
                    #     print(total)
                    #     return paragraphs,sum
                    #     # print(paragraphs)
                    # def search(keywords):
                    #     sum=0
                    #     keywords=["fest"]
                    #     num = len(keywords)
                    #     model_name = 'bert-base-uncased'
                    #     labels=keywords
                    #     num_labels = num  # Number of classification labels
                    #     tokenizer = BertTokenizer.from_pretrained(model_name)
                    #     model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
                    #     paragraphs,sum=parawise()

                    #     # Function to calculate reliability score for a given label
                    #     def calculate_reliability_score(probs, label):
                    #             label_index = labels.index(label)
                    #             return probs[label_index]



                    #     # Initialize variables to store the most reliable label and its score
                    #     most_reliable_label = None
                    #     max_reliability_score = 0.0
                    #     related_paragraphs = []           
                            
                    #     # Process each paragraph and find the most reliable label
                    #     for paragraph in paragraphs:
                    #         # Tokenize input text
                    #         tokens = tokenizer.encode_plus(
                    #             paragraph,
                    #             max_length=128,  # Max sequence length for BERT
                    #             padding='max_length',
                    #             truncation=True,
                    #             return_tensors='pt'  # Return PyTorch tensors
                    #         )
                            
                    #         # Forward pass through the model
                    #         outputs = model(**tokens)

                    #         # Get predicted probabilities and labels
                    #         probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()[0]
                    #         predicted_label = labels[torch.argmax(outputs.logits, dim=1).item()]

                    #         # Calculate the reliability score for the predicted label
                    #         reliability_score = calculate_reliability_score(probs, predicted_label)

                    #         if reliability_score > max_reliability_score:
                    #             max_reliability_score = reliability_score
                    #             most_reliable_label = predicted_label
                    #             related_paragraphs = [paragraph]
                    #         if reliability_score == max_reliability_score and predicted_label == most_reliable_label:
                    #             related_paragraphs.append(paragraph)

                    #         # Filter paragraphs that contain the given keywords
                    #         keywords = set(keywords)  # Convert to a set for faster lookup
                    #         related_paragraphs_with_keywords = [para for para in related_paragraphs if any(keyword in para.lower() for keyword in keywords)]

                    #         # Print the most reliable label and its related paragraphs with keywords
                    #         print("Most Reliable Label:", most_reliable_label)
                    #         print("Reliability Score:", max_reliability_score)
                    #         print("related data:")
                    #         print(related_paragraphs)
                    #         print("Related Paragraphs with Keywords:")
                    #         print("\n".join(related_paragraphs_with_keywords))
                    # def summary():
                    #     from transformers import pipeline
                    #     sum=0
                    #     summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                    #     paragraphs,sum=parawise()
                    #     for paragraph in paragraphs:
                    #         print("Summary:")
                    #         print(summarizer(paragraph, max_length=130, min_length=30, do_sample=True))
                    
                            
                    # def recognize():     
                    #         tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
                    #         model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
                    #         token_data_list = []
                    #         nlp = pipeline("ner", model=model, tokenizer=tokenizer)
                    #         #example = "An employee lives in Berlin and works at UNESCO and his name is Wolfgang"
                    #         paragraphs=parawise()
                    #         for paragraph in paragraphs:
                    #             ner_results = nlp(paragraph)
                    #             # print(ner_results)
                    #             print("***************recognise\n")
                                
                    #             for token_data in ner_results:
                    #                 word = token_data["word"]   # The word (token) itself
                    #                 score = token_data["score"] # The confidence score for the predicted label
                    #                 entity = token_data["entity"] # The predicted named entity label
                    #                 index = token_data["index"]   # The index of the token in the original sentence

                    #             token_dict = {
                    #                 "Token": word,
                    #                 "Score": score,
                    #                 "Entity": entity,
                    #                 "Index": index
                    #             }
                    #             token_data_list.append(token_dict)
                                
                    #         print(token_data_list)
                    # # day=parawise()
                    # # print(day)
                    # def questioning():
                    #         model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
                    #         tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')

                    #         input_file = "input.txt"
                    #         with open(input_file, "r") as file:
                    #             text = file.read()

                    #         context = text

                    #         question = [
                    #             "what is politics"
                    #         ]

                    #         tokenizer.encode(question[0], truncation=True, padding=True)
                    #         nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
                    #         result = nlp({
                    #             'question': question[0],
                    #             'context': context
                    #         })

                    #         # Expand the context window to include more sentences before and after the answer
                    #         context_start1 = max(0, result['start'] - 50)
                    #         context_end1 = min(len(context), result['end'] + 50)
                    #         context_start = max(0, result['start'] - 100)
                    #         context_end = min(len(context), result['end'] + 150)
                    #         inputed=input("enter size 1 max 2 min")
                    #         if "1" in inputed:
                    #             expanded_context = context[context_start:context_end]
                    #         if "2" in inputed:
                    #             expanded_context = context[context_start1:context_end1]    


                    #         print("Question:", question[0])
                    #         print("Answer:", result['answer'])
                    #         print("Expanded Context:", expanded_context)

                    # import threading

                    # def thread_function(target_function):
                    #     thread = threading.Thread(target=target_function)
                    #     thread.start()
                    #     thread.join()

                    # # Create threads for each function
                    # thread_classify = threading.Thread(target=CLassify)
                    # thread_recognize = threading.Thread(target=recognize)
                    # thread_questioning = threading.Thread(target=questioning)
                    # thread_summary = threading.Thread(target=summary)

                    # # Start the threads
                    # thread_function(CLassify)
                    # thread_function(recognize)
                    # thread_function(questioning)
                    # thread_function(summary)

          

if __name__ == "__main__":
    app.run()