#!/usr/bin/env python
# coding: utf-8

# # DS Automation Assignment

# Using our prepared churn data from week 2:
# - use pycaret to find an ML algorithm that performs best on the data
#     - Choose a metric you think is best to use for finding the best model; by default, it is accuracy but it could be AUC, precision, recall, etc. The week 3 FTE has some information on these different metrics.
# - save the model to disk
# - create a Python script/file/module with a function that takes a pandas dataframe as an input and returns the probability of churn for each row in the dataframe
#     - your Python file/function should print out the predictions for new data (new_churn_data.csv)
#     - the true values for the new data are [1, 0, 0, 1, 0] if you're interested
# - test your Python module and function with the new data, new_churn_data.csv
# - write a short summary of the process and results at the end of this notebook
# - upload this Jupyter Notebook and Python file to a Github repository, and turn in a link to the repository in the week 5 assignment dropbox
# 
# *Optional* challenges:
# - return the probability of churn for each new prediction, and the percentile where that prediction is in the distribution of probability predictions from the training dataset (e.g. a high probability of churn like 0.78 might be at the 90th percentile)
# - use other autoML packages, such as TPOT, H2O, MLBox, etc, and compare performance and features with pycaret
# - create a class in your Python module to hold the functions that you created
# - accept user input to specify a file using a tool such as Python's `input()` function, the `click` package for command-line arguments, or a GUI
# - Use the unmodified churn data (new_unmodified_churn_data.csv) in your Python script. This will require adding the same preprocessing steps from week 2 since this data is like the original unmodified dataset from week 1.

# In[1]:


get_ipython().system('conda create -n msds python=3.10.14 -y')
get_ipython().system('conda activate msds')
get_ipython().system('pip install --upgrade pycaret')


# In[50]:


import pandas as pd
df=pd.read_csv('C:/Users/DELL/Downloads/week2prepared_churn.csv',index_col='customerID')
df
df.drop('TotalCharges_to_MonthlyCharges',axis=1,inplace=True)


# In[51]:


df


# In[52]:


from pycaret.classification import setup, compare_models, predict_model, save_model, load_model


# PyCaret's classification module simplifies the machine learning workflow with user-friendly functions. The setup function initializes the environment and prepares the data. The compare_models function evaluates different algorithms to find the best one, while predict_model is used for making predictions. Additionally, save_model and load_model help manage trained models efficiently.
# 

#  We used the setup() method with the dataset (df) and specified the target variable as 'Churn' in order to prepare the data for automatic machine learning (AutoML). PyCaret's AutoML module was utilized for this purpose.
# 

# In[53]:


automl = setup(df, target='Churn')


# Here the used the compare models method() to compare the different models with each model performances 

# 

# In[55]:


best_model = compare_models(sort='AUC')


# In[56]:


best_model


# In[57]:


df.iloc[-2:-1].shape


# The predict_model(best_model, df.iloc[-2:-1]) is used to predict the outcome for the second-to-last row in the DataFrame. 
# It processes this single data instance and returns the original features along with the predicted class and its probabilities.

# In[58]:


predict_model(best_model, df.iloc[-2:-1])


# Here we can see that two new coloumns are generated prediction_label which will be for coustmour churning and predection of score 

# The best_model has saved 

# In[59]:


save_model(best_model, 'h_churn')


# The data set was load from csv file and predections was made using churn data using a pre-trained machine learining from pyCaret.Here first the data was read into a data frame and then loads the model is called best_model.after making predections it checks a specific coloumn for predection label, if it founds then it rename that coloumn to "Churn_prediction" and replaces the numeric labels (1 and 0) with  ("Churn" and "No Churn"). 

# In[60]:


from pycaret.classification import load_model, predict_model

def load_data(filepath):
   
    df = pd.read_csv(filepath, index_col='customerID')
    return df

def make_predictions(df):
   
    model = load_model('best_model') 
    predictions = predict_model(model, data=df)

    print("Predictions DataFrame Columns:")
    print(predictions.columns)

    if 'prediction_label' in predictions.columns:
        predictions.rename(columns={'prediction_label': 'Churn_prediction'}, inplace=True)
        
        predictions['Churn_prediction'].replace({1: 'Churn', 0: 'No Churn'}, inplace=True)
        
        return predictions[['Churn_prediction', 'prediction_score']]
    else:
        raise KeyError("The 'prediction_label' column was not found in the predictions DataFrame")

if __name__ == "__main__":
    new_data = load_data('C:/Users/DELL/Downloads/new_churn_data.csv')
    predictions = make_predictions(new_data)
    true_values = [1, 0, 0, 1, 0]
    
    print('Predictions:')
    print(predictions)
    print('True Values:')
    print(true_values)



    


# # Summary

# Write a short summary of the process and results here.
# 
# 
# First, the required packages were installed, and the necessary PyCaret modules were imported to set up the AutoML environment. Various classifiers were evaluated to predict churn, and since AUC is a key metric, the Gradient Boosting Classifier (GBC) was chosen with an AUC of 0.8289, slightly higher than Logistic Regression's 0.8273. The best model was saved as best_model.pkl, along with its preprocessing pipeline. When applied to the new dataset, new_churn_data.csv, the model predicted "No Churn" for all instances, while the actual values were [1, 0, 0, 1, 0]. This indicates that the model failed to identify any churning cases, highlighting the need for further refinement to improve its predictive capabilities. Overall, this process illustrates the steps of training a model, saving it, and using it for churn predictions.
# 
