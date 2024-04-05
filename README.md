**Contributed by Aravindalochanan.N**

The flow of this Application:
1. User chooses to Raise a new issue or Check/Update an already raised ticket.
2. If new issue:
    1. User enters the issue.
    2.  First it is given to the Type_Clf model which predicts if the input is relevant to the ticketing context (springboot) or not.
    3.  If it's irrelevant then user is prompted to provide more details.
    4.  If relevant (this could be a misclassification at times),the input goes through the Actual prediction model that tell what kind of an issue it is.
    5.  Here a threshold is set for the model score so that even if the Type_Clf model misclassified an irrelavant input as relevant,it can be handled here.
    6.  If Irrelevant,the user can again provide more details and the cycle will continue.
    7.   If relevant,the model predicts the type of issue and provides solution by looking for a similar issue in the closed issues.
    8.   If there is one,then the solutions of that issue is provided to the user.
    9.   If it works,the user can accept it. If it doesn't a ticket will be raised.


                                                                                                
3.Check/Update:
 1. Pretty much straight forward-User is asked to enter the ticket ID to check status or update it.
 

 
 
  
 
  


To run the backend:

1. Open the BackEnd.py file
2. Change the paths to your actual path without changing the file name and change the api token
   ```
   new_model = AutoModelForSequenceClassification.from_pretrained('your_path/Bert82_82')
   type_clf=AutoModelForSequenceClassification.from_pretrained('your_path/TypeClf')
   new_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
   API_URL = "https://api-inference.huggingface.co/models/google/pegasus-large"
   headers = {"Authorization": "your_api_token"}
   ```
4. Change the path-    ```df=pd.read_csv("your_path/spring_gh_dataset_2.csv")```
5. run ```pip install -r final.txt to install the required packages```
6. run ```uvicorn BackEnd:app --reload```

To run the frontend:


1. install node js
2. run ```npm install -g serve```
3. run ``` serve -s build -l [PORT_NUMBER]```
4. For sample inputs,refer to inputs.txt

