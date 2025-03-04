import pickle
from  flask import Flask,render_template,request
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData,Predict_pipeline
application = Flask(__name__)

app =application

## route
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/predict",methods = ["GET","POST"])
def predict_data():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            duration=float(request.form.get("duration"))  ,
            days_left=float(request.form.get("days_left")),
            source_city=request.form.get("source_city"),
            departure_time=request.form.get("departure_time"),
            stops=request.form.get("stops"),
            arrival_time=request.form.get("arrival_time"),
            destination_city=request.form.get("destination_city"),
            Class=request.form.get("Class")
            
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline = Predict_pipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template("home.html",results=f'{(results[0]):,.2f}')

if __name__ =="__main__":
    app.run(host="0.0.0.0",debug=True)