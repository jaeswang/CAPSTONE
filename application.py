# application.py
import pandas as pd
import numpy as np
from dash import Dash, html
import plotly.express as px

# Load dataset
dataset_dir = "data/Selected_Station_Observations_Daily_Xtab_202410121100.csv"
df = pd.read_csv(dataset_dir)

# Data Cleaning
df = df.drop("Unnamed: 7", axis=1)
df["Date Time"] = pd.to_datetime(df["Date Time"].str[2:18])
df = df[["Date Time", "DISCHRG Value"]].dropna()

# Add calculated fields
df["diffs"] = df["DISCHRG Value"].diff().fillna(0)
df["color"] = ['green' if x >= 0 else 'red' for x in df["diffs"]]
df['lat'] = [37.72472199] * len(df)
df['lon'] = [-107.2556094] * len(df)

# Create Plotly graph
fig = px.line(df, x="Date Time", y="diffs", title='Streamflow')

# Dash App
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Streamflow Dashboard", style={"textAlign": "center"}),
    html.Div(children=[
        html.Div("Streamflow Data", style={"padding": "10px"}),
        html.Div("Dashboard in Progress", style={"padding": "10px"})
    ]),
])

if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0")