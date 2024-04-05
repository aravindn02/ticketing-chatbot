
import pandas as pd

import sqlite3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib
import torch
from transformers import pipeline
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import date
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
new_model = AutoModelForSequenceClassification.from_pretrained('/home/aravindnarayanan/UI/Bert82_82')
type_clf=AutoModelForSequenceClassification.from_pretrained('/home/aravindnarayanan/TypeClf')
new_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
API_URL = "https://api-inference.huggingface.co/models/google/pegasus-large"
headers = {"Authorization": "Your_Token_Here"}
# Define label mapping
label_map={0:'Bug', 1:'Documentation', 2:'Enhancement', 3: 'Task'} 



# Function to write dictionary to JSON file
def write_to_json(data,file_path):
    with open(file_path, 'a') as json_file:
        json.dump(data, json_file)
        json_file.write('\n') 


def query(payload):
    print("Entered API call")
    response = requests.post(API_URL, headers=headers, json=payload)
    print("Returning json")
    return response.json()



def get_sent(label):
    if label==-1:
        return {}
    print("Label received in get sentences function:",label)
    print("Entered get sentence function")
    df=pd.read_csv("/home/aravindnarayanan/spring_gh_dataset_2.csv")
    print("Data file imported")
    df=df[['Title','URL','IssueType']]
    selected_entries = df.loc[df['IssueType'] == label, ['Title', 'URL']]
    df_dict = selected_entries.to_dict(orient='records')
    dit={}
    for i in df_dict[0:10]:
        dit[i['Title']]=i['URL']
    
    print("Data going to be returned")
    return dit



def compute_similarity(sentence1, sentence2):

    print("Sim computation entered")
    vectorizer = CountVectorizer().fit([sentence1, sentence2])
    vector1, vector2 = vectorizer.transform([sentence1, sentence2])
    cosine_sim = cosine_similarity(vector1, vector2)
    return cosine_sim

def override_classify_text(text):
    print("Enter classify text function")
    encoding = new_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    encoding = {k: v for k, v in encoding.items()}
    score_list=[]
    with torch.no_grad():
        outputs = new_model(**encoding)
        
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)  # Applying softmax to convert logits to probabilities
    score_list.append(probabilities.cpu().numpy())
    print("Probs predicted:",score_list)
    probs = F.softmax(logits, dim=1)
    label = torch.argmax(probs, dim=1).item()
    print("Predicted Label is:",label)
    predicted_class = label_map[label]
    print("Label returning")
    return predicted_class




# Function to preprocess text and get prediction
def classify_text(text):
    print("Enter classify text function")
    encoding = new_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    encoding = {k: v for k, v in encoding.items()}
    score_list=[]
    with torch.no_grad():
        outputs = new_model(**encoding)
        
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)  # Applying softmax to convert logits to probabilities
    score_list.append(probabilities.cpu().numpy())
    print("Probs predicted:",score_list)
    probs = F.softmax(logits, dim=1)
    for li in score_list:
        for val in list(li):
            val=list(val)
            val.sort(reverse=True)
            print(val)
            if val[0]>0.5:
                label = torch.argmax(probs, dim=1).item()
                print("Predicted Label is:",label)
                predicted_class = label_map[label]
                print("Label returning")
                return predicted_class
            else:
                return -1
                
                    
           


def get_output(text_prompt,override=0):

    if override==0:
        prediction = classify_text(text_prompt)
    else:
        prediction=override_classify_text(text_prompt)
    if prediction==-1:
        print("It is negative")
        return prediction,[]
    result = query({"inputs": text_prompt})
    for res in result:
        summary = res['summary_text']
    dictn = get_sent(prediction)
    sents = list(dictn.keys())
    if len(sents)==0:
        return -1,[]
    print("Sentences received:",sents)
    similarity_scores = []
    
    for text in sents:
        similarity= compute_similarity(summary, text)
        similarity_scores.append((text, similarity))  # Update shared value every 1 second by incrementing by 1
    top_3_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:3]
    return prediction,top_3_scores
def biggerFunc(prompt,override=0):
        
    global user_prompt
        
    user_prompt=prompt
    if(override==1):
        prompt=prompt.replace("aravindalochanan","")
    global user_label
    global user_solns
    user_solns={}
    print("Entered the larger than one prompt section")
    print("Prompt:",prompt)
    print("-----------------")
    result=query({"inputs":prompt})
    print("Result:",result)
    print(type(result))
    summary=result[0]['summary_text']
    if(override==0):
        class_label,scores=get_output(prompt)
    else:
        class_label,scores=get_output(prompt,1)
    if (class_label==-1 and len(scores)==0):
        print("Negative received at big function and displaying")
        responses=[]
        #responses.append(f"As on {date.today()},the data that I've been trained on tells me that the input is not valid in this context. If this is not the case,please click the reject button to override the model decision.")
        return responses
    user_label=class_label
    ditn=get_sent(class_label)
    responses=[]
    responses.append(f"Summary : {summary}\n")
    responses.append(f"Issue-Type : {class_label}\n")
    for idx, (text, score) in enumerate(scores, start=1):
        responses.append(f"{idx} : {ditn[text]}\n")
        user_solns[idx]=ditn[text]
    

    return responses


def type_classify(text):
    print("Entered the module to check if input is valid in the context of this chatbot")
    encoding = new_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    encoding = {k: v for k, v in encoding.items()}

    with torch.no_grad():
        outputs = type_clf(**encoding)
        logits = outputs.logits

    probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
    label = np.argmax(probs, axis=-1)
    print("Predicted Label is:",label)
    return label
def clean(text):
    print("Entered text filtering section")
    pattern = r'[^a-zA-Z0-9\s]'
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text



app = FastAPI()

# Configure CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with the origin of your React.js app
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Define a route using the `app.get` decorator
@app.get("/main")

def main(prompt:str):
      
                
        print("Query received:",prompt)
        temp=prompt.split()
        print(len(temp))
        if("aravindalochanan" in prompt):
            responses=biggerFunc(prompt,1)
            return responses
        check_val=type_classify(str(prompt))
        if len(clean(prompt))>1 and check_val!=0:
            responses=biggerFunc(prompt)
            
            print("Valid input,response received from computation functions")
            print("Displayed responses and calling feedback")
            return responses
        
        else:
            responses=[]
            return responses

@app.get("/databack")


def databack(solution):
    # Connect to the SQLite database
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    up=user_prompt
    # Create a table to store the data if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS accepted_data
                      (issue TEXT, solution TEXT,override INTEGER)''')

    user_data = {}
    if("aravindalochanan" in up):
        override=1
        up=up.replace("aravindalochanan","")
    else:
        override=0
    user_data["issue"] = up
    solns = []
    solns.append(user_label)
    rcvd = solution.split(",")
    for index in rcvd:
        solns.append(user_solns[int(index)])
    user_data['solution'] = ','.join(solns)

    # Insert the data into the table
    cursor.execute('INSERT INTO accepted_data (issue, solution,override) VALUES (?, ?,?)', (user_data["issue"], user_data["solution"],override))
    
    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    return "Data Received"

@app.get("/ticket")
def ticket(details):
    # Connect to the SQLite database
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()

    # Create a table to store the data if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS ticket_data
                      (tid INTEGER,issue TEXT)''')

    user_data = {}
    
    user_data["issue"] = user_prompt
    tid=int(details)
  

    # Insert the data into the table
    cursor.execute('INSERT INTO ticket_data (tid,issue) VALUES (?, ?)', (tid,user_data["issue"]))
    
    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    return "Data Received"

@app.get("/check")
def check(tno):
    tno=int(tno)
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute('SELECT tid, issue FROM ticket_data where tid=?',(tno,))
    row = cursor.fetchone()
    if row:
        tid, issue = row
        print("tid:", tid)
        print("issue:", issue)
        return (f"Ticket ID: #{tid} \n Issue:{issue}")
    else:
        return(f"There is no ticket with this id #{tno}")
    conn.close()
