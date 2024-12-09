import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import torch

# Define constants
DATA_FILE = "Selected_Station_Observations_Daily_Xtab_202410121100.csv"
MODEL_FILE = "nbeats_1day.pth"
STATION_NAME = "RIO GRANDE AT THIRTYMILE BRIDGE, NR CREEDE, CO."
STATION_LAT = 37.72472199
STATION_LON = -107.2556094

# Data loading and preprocessing
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop("Unnamed: 7", axis=1)
    df["Date Time"] = pd.to_datetime(df["Date Time"].str[2:18])
    df = df[["Date Time", "DISCHRG Value"]].dropna()
    df["station"] = STATION_NAME
    df["lat"] = STATION_LAT
    df["lon"] = STATION_LON
    diffs = df["DISCHRG Value"].diff().fillna(0)
    df["diffs"] = diffs
    df["color"] = np.where(diffs >= 0, "green", "red")
    return df

# Model loading
def load_model(file_path):
    model = torch.load(file_path)
    model.eval()
    return model

# Create visualization
def create_map(df):
    fig = go.Figure(
        data=[
            go.Scattergeo(
                lon=[df.iloc[-1]["lon"]],
                lat=[df.iloc[-1]["lat"]],
                text=df.iloc[-1]["station"],
                mode="markers",
                marker=dict(
                    size=15,
                    color=df.iloc[-1]["color"],
                    line=dict(width=2, color="black"),
                ),
            )
        ]
    )
    fig.update_geos(
        showsubunits=True,
        subunitcolor="black",
        fitbounds="locations",
    )
    fig.update_layout(title="Rio Grande Streamflow", geo_scope="usa")
    return fig

def create_time_series(df):
    fig = px.line(df, x="Date Time", y="DISCHRG Value", color="color", title="Streamflow Over Time")
    fig.update_layout(xaxis=dict(rangeslider_visible=True), yaxis_title="Streamflow (DISCHRG Value)")
    return fig

# Add this new function for making predictions
def make_prediction(model, df):
    # Get the last 7 days of data
    last_week = df["DISCHRG Value"].tail(7).values
    # Normalize the data (adjust based on your model's training)
    input_data = torch.FloatTensor(last_week).unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_data)
    return prediction.numpy()[0]

# Load data and model
df = load_data(DATA_FILE)
model = load_model(MODEL_FILE)

# Dash application
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("Streamflow Prediction Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50'}),
        html.Div([
            html.Div([
                dcc.Graph(id="map", figure=create_map(df))
            ], className='six columns'),
            html.Div([
                html.H3("Current Statistics", style={'textAlign': 'center'}),
                html.Div(id="current-stats", style={'padding': '20px'})
            ], className='six columns'),
        ], className='row'),
        html.Div([
            dcc.Graph(id="time-series", figure=create_time_series(df))
        ], className='row'),
        html.Div([
            html.H3("Prediction for Next Day", style={'textAlign': 'center'}),
            html.Div(id="prediction-output", style={'textAlign': 'center'})
        ], className='row'),
    ], className='container')
])

# Add these callbacks
@app.callback(
    Output("current-stats", "children"),
    Input("time-series", "figure")
)
def update_stats(figure):
    current_value = df["DISCHRG Value"].iloc[-1]
    daily_change = df["diffs"].iloc[-1]
    return [
        html.P(f"Current Streamflow: {current_value:.2f}"),
        html.P(f"Daily Change: {daily_change:.2f}",
               style={'color': 'green' if daily_change >= 0 else 'red'})
    ]

@app.callback(
    Output("prediction-output", "children"),
    Input("time-series", "figure")
)
def update_prediction(figure):
    prediction = make_prediction(model, df)
    return f"Predicted Streamflow: {prediction[0]:.2f}"

# Add this before app.run_server
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == "__main__":
    app.run_server(debug=True)
