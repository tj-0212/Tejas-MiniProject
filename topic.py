# from transformers import pipeline
# classifier = pipeline("zero-shot-classification",
#                       model="facebook/bart-large-mnli")
# sequence_to_classify = "one day I will see the world"
# candidate_labels = ['travel', 'cooking', 'dancing', 'exploration']
# classifier(sequence_to_classify, candidate_labels, multi_class=True)
import transformers
from transformers import pipeline
from transformers import BertForQuestionAnswering
from transformers import AutoTokenizer
def questioning():
        model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
        tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')

        input_file = "input.txt"
        with open(input_file, "r") as file:
            text = file.read()

        context = text

        question = [
            "what is politics"
        ]

        # Tokenize the question
        tokenized_question = tokenizer.encode(question[0], truncation=True, padding=True, return_tensors='pt')

        nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
        result = nlp({
            'question': tokenizer.decode(tokenized_question[0]),
            'context': context
        })


        # Expand the context window to include more sentences before and after the answer
        # context_start1 = max(0, result['start'] - 50)
        # context_end1 = min(len(context), result['end'] + 50)
        context_start = max(0, result['start'] - 70)
        context_end = min(len(context), result['end'] + 170)
        # inputed=input("enter size 1 max 2 min")
        # if "1" in inputed:
        expanded_context = context[context_start:context_end]
        # # if "2" in inputed:
        # expanded_context1 = context[context_start1:context_end1]    


        print("Question:", question[0])
        print("Answer:", result['answer'])
        print("Expanded Context:", expanded_context)
        # print("Expanded Context:", expanded_context1)
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
questioning()                