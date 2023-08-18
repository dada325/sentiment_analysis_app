# Sentiment_Analysis (Using Distill-BERT pre-trained model to fine tune a personal Financial News Sentiment analysis web app)


# Try with the app under this address:  http://34.106.112.201:8502/
## and don't abuse it... it is still paying from my wallet. 


<img width="1629" alt="Screenshot 2023-08-18 at 02 27 16" src="https://github.com/dada325/sentiment_analysis_app/assets/7775973/3395f206-5872-4723-9044-4c5c4681d68b">


### **1. Research Phase**

#### **Problem Definition**:
The primary goal was to develop a system capable of analyzing textual data to determine its sentiment. The sentiments could be categorized as positive (0), neutral(1), or negative(2).

#### **Data Collection**:
- I am using the dataset from Kaggle Financial news sentiment analysis [Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)

#### **Exploratory Data Analysis (EDA)**:
- It's essential to explore the dataset to understand its structure, any missing values, the distribution of sentiments, etc.
- This dataset is very clean, and we have balanced sample of three categories. 

#### **Data Preprocessing**:
- Tokenization: Break down text into words, phrases, symbols, or other meaningful elements.
- Stopword Removal: Get rid of commonly used words that don't carry significant meaning. I hand picked the stopword, hard coded in the training. but you can also use NLTK library and download the stop words to use for removal. 
- Lemmatization/Stemming: Reduce words to their base/root form.
- Encoding: Convert textual data into numerical form so that it can be fed into a model. Techniques like Bag of Words, TF-IDF, and Word Embeddings (like Word2Vec or GloVe) can be used.

### **2. Model Development and Training**

#### **Model Selection**:
- For this app, the DistilBert model was chosen. DistilBert is a smaller version of BERT, which retains BERT's performance while being lighter and faster. I am using the Huggingface version, and use this pre-trained model to fine tune on my own dataset, the financial news sentiment analysis. 

#### **Model Training**:
- Training just costed about 30 mins on a CPU, I didn't use parallel workers to work on multiple gup. I don't think i need to for the pre-trained model is small, and the dataset is not that enormous.

  <img width="1041" alt="Screenshot 2023-08-18 at 02 02 35" src="https://github.com/dada325/sentiment_analysis_app/assets/7775973/3d37aee1-a652-4d3b-b2a3-870853ed4656">

This hitted a accuracy of 0.92, pretty good for a 30 mins training for a LLM. 
But we are not getting better since the first epoch of the accuracy on the test set of 0.80
  
#### **Model Evaluation**:
- After training, the model was evaluated on a separate validation set to determine its accuracy and overall performance.

### **3. App Development**

#### **Streamlit Framework**:
- Streamlit is a Python library that simplifies the process of turning data scripts into shareable web applications.
- This Library is extreamly quick for a fast prototype like this. You just don't need to know anything about the frontend development.

  My app looks like this 
<img width="1641" alt="Screenshot 2023-08-18 at 02 24 24" src="https://github.com/dada325/sentiment_analysis_app/assets/7775973/0b9d7c3b-e903-495a-b833-a7a9053e3079">

#### **Integrating the Model**:
- The trained DistilBert model was integrated into the Streamlit app. When a user inputs text, the app tokenizes and preprocesses the text, feeds it to the model, and displays the predicted sentiment.

### **4. Deployment Phase**

#### **Containerization**:
- Docker was used to containerize the app, making it easy to replicate and deploy anywhere.
- A Dockerfile was created, specifying the base Python image, the necessary dependencies (like Streamlit and Transformers), and the command to run the app.

#### **Pushing to Google Cloud Registry**:
- The Docker image was pushed to Google Cloud Registry (GCR), providing a centralized location to manage the Docker images and ensuring that the same image could be deployed consistently.

#### **Deploying on Google Cloud VM**:
- A Google Cloud VM instance was created.
- The Docker image from GCR was pulled onto this VM.
- The app was then run on the VM, making it accessible via the VM's external IP address.

### **5. Troubleshooting and Optimization**

- Several challenges were faced, such as Docker image size optimization, ensuring that the necessary files (like the saved model) are available in the correct directory, and setting up correct firewall rules to allow external access to the app.
- App optimization and troubleshooting were iterative processes, with changes made based on the errors encountered and the desired optimizations.

---

This entire process, from research to deployment, represents a typical machine learning pipeline. The specific steps and technologies can vary based on the problem, the available data, and the deployment environment. However, the overall structure—research, development, and deployment—remains consistent across many machine learning projects.


### possible improvement 


Parameter Efficient Fine-Tuning Techniques

1 Knowledge Distillation
Pruning
Quantization
Low-Rank Factorization
Knowledge Injection
Adapter Modules
