# Derma Clinic AI Agent

The idea is to create an assistant for a dermatology clinic that can work with customer data, answer questions about clinic performance, and make use of machine learning models to predict customer behavior.

The project combines LangGraph, Ollama, and Scikit-learn, with a Gradio interface.

# What it can do

- Look up a customer using their ID and retrieve their profile and visit history.
- Answer questions about clinic visits, revenue, services, and customer acquisition channels.
- Predict the likelihood of a customer returning within 90 days.
- Estimate customer responses to different discount offers.

# How it works

The assistant runs on Llama 3.2 locally through Ollama. I used LangGraph to manage the conversation and decide when to call one of the tools.

The project currently has three tools:

- lookup_customer – retrieves customer information and visit history.
- clinic_analytics – provides statistics about clinic performance.
- predict_customer_return_by_id – prepares customer features and runs the prediction models.

# Machine Learning

I trained two classification models using Scikit-learn:

- Logistic Regression for predicting customer returns within 90 days.
- Random Forest for estimating customer responses to discount offers.


# Technologies

- Python
- LangGraph and LangChain
- Ollama / Llama 3.2
- Scikit-learn
- Pandas
- Gradio

# Dataset

The project uses synthetic clinic data created for demonstration purposes. No real patient or customer information is included.