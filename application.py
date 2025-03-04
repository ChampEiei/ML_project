import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData, Predict_pipeline
import plotly.express as px
import plotly.io as pio

application = Flask(__name__)
app = application

# Base routes for portfolio navigation
@app.route("/")
def index():
    # This can serve as your landing or home page.
    return render_template("index.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/projects")
def projects():
    return render_template("projects.html")

@app.route("/skills")
def skills():
    return render_template("skills.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

# Prediction route (can be linked to a form or separate functionality)
@app.route("/predict", methods=["GET", "POST"])
def predict_data():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            duration=float(request.form.get("duration")),
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

        # Example of generating a Plotly graph (replace with your actual graph logic)
        fig = px.bar(
            x=['Feature1', 'Feature2', 'Feature3'],
            y=[10, 20, 30],
            title="Prediction Results"
        )

        # Convert Plotly figure to HTML string
        graph_html = pio.to_html(fig, full_html=False)

        # Pass the graph HTML and results to the template
        return render_template("home.html", results=f'{results[0]:,.2f}', graph_html=graph_html)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5001)
