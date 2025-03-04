import os
import sys 
import plotly.express as px
import plotly.io as pio
import numpy as np 
import pandas as pd
import dill
from src.exception import Custom_exception
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from flask import Flask


def save_objects(file_path,obj):
    try:
        dir_name= os.path.dirname(file_path)
        
        os.makedirs(dir_name,exist_ok=True)
        
        with open(file_path,'wb') as file:
            dill.dump(obj,file)

    except Exception as e:
        raise Custom_exception(e,sys)

def evaluate_model(x_train,y_train,x_test,y_test,models,params):
    report ={}
    for i,value in models.items():
        model = value
        params = params[i]
        grid = GridSearchCV(model,param_grid=params,n_jobs=-1,cv=5,)
        grid.fit(x_train,y_train)
        model.set_params(**grid.best_params_)
        model.fit(x_train,y_train)
        y_train_pred = model.predict(x_train)

        model.fit(x_test,y_test)
        y_test_pred = model.predict(x_test)

        train_model_score = r2_score(y_train,y_train_pred)
        test_model_score = r2_score(y_test,y_test_pred)
        report[i] = test_model_score 
        return report
    
def load_object(file_path):
    try:       
        with open(file_path,'rb')as file:
            obj = dill.load(file)
            return obj
    except Exception as e:
        raise Custom_exception(e,sys)

def dashboard():
    # Load the dataset
    df = pd.read_csv("notebook/data/Clean_Dataset.csv")
    
    # Graph 1: Price Distribution Histogram
    # Downsample the data if necessary
    df_hist = df.sample(n=1000, random_state=42) if len(df) > 1000 else df
    fig1 = px.histogram(df_hist, x='price', nbins=50, title='Price Distribution')
    graph1 = pio.to_html(fig1, full_html=False)
    
    # Graph 2: Average Price by Days Left (Line Chart)
    avg_price_by_days = df.groupby('days_left', as_index=False)['price'].mean()
    fig2 = px.line(avg_price_by_days, x='days_left', y='price', title='Average Price by Days Left')
    graph2 = pio.to_html(fig2, full_html=False)
    
    # Graph 3: Price Distribution by Flight Class (Box Plot)
    # Sample up to 1000 rows per flight class to avoid rendering too many points
    df_box = df.groupby('class', group_keys=False).apply(lambda x: x.sample(n=min(1000, len(x)), random_state=42))
    fig3 = px.box(df_box, x='class', y='price', title='Price Distribution by Flight Class')
    graph3 = pio.to_html(fig3, full_html=False)
    
    # Graph 4: Frequency of Flight Stops (Bar Chart)
    stops_count = df['stops'].value_counts().reset_index()
    stops_count.columns = ['stops', 'count']
    fig4 = px.bar(stops_count, x='stops', y='count', title='Frequency of Flight Stops')
    graph4 = pio.to_html(fig4, full_html=False)
    
    # Data Summary: Statistical overview of numeric features
    summary = df.describe().to_html(classes='table table-striped')
    
    return (graph1, graph2, graph3, graph4, summary)
                            
                            
                           
                           