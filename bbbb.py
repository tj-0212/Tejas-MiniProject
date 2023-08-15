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
def recognize():
        sum=0     
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        token_data_list = []
        nlp = pipeline("ner", model=model, tokenizer=tokenizer)
        # example = "An employee lives in Berlin and works at UNESCO and his name is Wolfgang"
        # ner_results=nlp(example)
        # print(ner_results)
        paragraphs,sum=parawise(path)
        # print(paragraphs)
        for paragraph in paragraphs:
            # print(paragraph)
            ner_results = nlp(paragraph)
            if len(ner_results) == 0:
                #  print("not identified")
                pass
            else:
                 print(ner_results)     
            # if ner_resultNone:
            #     print(token_data_list)
            # print(ner_results)
            # print("***************recognise\n")
            
            # for token_data in ner_results:
            #      print(token_data)
                # word = token_data[0]   # The word (token) itself
                # score = token_data[1] # The confidence score for the predicted label
                # entity = token_data[2] # The predicted named entity label
                # index = token_data[3]   # The index of the token in the original sentence


            # token_dict = {
            #     "Token": word,
            #     "Score": score,
            #     "Entity": entity,
            #     "Index": index
            # }
            # token_data_list.append(token_dict)
            
        
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
recognize()         