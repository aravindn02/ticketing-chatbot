

To run the backend:

1. open the BackEnd.py file
2. Change the paths to your actual path without changing the file name and change the api token
   new_model = AutoModelForSequenceClassification.from_pretrained('your_path/Bert82_82')
   type_clf=AutoModelForSequenceClassification.from_pretrained('your_path/TypeClf')
   new_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
   API_URL = "https://api-inference.huggingface.co/models/google/pegasus-large"
   headers = {"Authorization": "your_api_token"}
3. In line 54 of the code change the path- df=pd.read_csv("your_path/spring_gh_dataset_2.csv")
4. run -- pip install -r final.txt to install the required packages
5. run -- uvicorn BackEnd:app --reload

To run the frontend:


1. install node js
2. run -- npm install -g serve
3. run -- serve -s build -l [PORT_NUMBER]
4. For sample inputs,refer to inputs.txt

Workflow of Frontend
 --Whether new issue input or checking existing ticket?
 --for new issue,user enters the input
 --model tells whether it's valid or not. User can accept this or reject this and override the model decision
 --model gives a solution-If user accepts,then for the purpose of other users,the solutions that worked are received from user
 --If rejected,model tries again. User can further reject and this will go for 3 times after which a ticket is created 
 NOTE: In case of user overriding a invalid input and getting a solution,it is stored in the database but with a flag that this was an overriden issue. 
 
 --In case of checking ticket,user enters the ticket id
 --If it exists,then the details are shown,else it is shown that there is no such ticket.
