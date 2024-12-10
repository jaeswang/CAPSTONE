# application.py
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import geopandas as gpd
import shapely.geometry
import wget
import os
import plotly.io as pio
import webbrowser

# Station configurations
STATIONS = {
    'Rio Milco': {
        'file': 'Predictions/preds_30day_riomilco.csv',
        'color': 'blue',
        'coordinates': [29.1825, -102.9572]
    },
    'El Paso': {
        'file': 'Predictions/preds_30day_elpaso.csv',
        'color': 'red',
        'coordinates': [31.8032, -106.541]
    },
    'Big Bend': {
        'file': 'Predictions/preds_30day_bigbend.csv',
        'color': 'green',
        'coordinates': [29.3300, -103.2080]
    }
}

# Load predictions for all stations
predictions = {}
for station_name, config in STATIONS.items():
    df = pd.read_csv(config['file'])
    df["Date"] = pd.to_datetime(df["Date"])
    predictions[station_name] = df

# Calculate statistics for all stations
stats = {}
for station_name, df in predictions.items():
    stats[station_name] = {
        "Current Flow": {
            "value": f"{df['DISCHRG Value'].iloc[0]:.2f}",
            "description": "Most recent streamflow measurement in cubic feet per second"
        },
        "30-Day Max": {
            "value": f"{df['DISCHRG Value'].max():.2f}",
            "description": "Highest predicted streamflow over the next 30 days"
        },
        "30-Day Min": {
            "value": f"{df['DISCHRG Value'].min():.2f}",
            "description": "Lowest predicted streamflow over the next 30 days"
        },
        "30-Day Average": {
            "value": f"{df['DISCHRG Value'].mean():.2f}",
            "description": "Average predicted streamflow over the next 30 days"
        },
        "Daily Change Rate": {
            "value": f"{df['DISCHRG Value'].diff().mean():.2f}",
            "description": "Average day-to-day change in streamflow"
        }
    }

# Create time series figure with all stations
fig1 = go.Figure()

for station_name, config in STATIONS.items():
    df = predictions[station_name]
    fig1.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["DISCHRG Value"],
            name=f"{station_name}",
            line=dict(color=config['color'])
        )
    )

fig1.update_layout(
    title="30-Day Streamflow Predictions by Station",
    xaxis=dict(
        rangeslider=dict(visible=True, thickness=0.05),
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=14, label="2w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(step="all", label="All")
            ])
        )
    ),
    yaxis_title="Streamflow (ft³/s)",
    hovermode="x unified"
)

# Export time series figure
pio.write_html(fig1, file="timeseries.html", full_html=True, auto_open=True)

# Download the shapefile if it doesn't exist
shapefile_path = "ne_50m_rivers_lake_centerlines.zip"
if not os.path.exists(shapefile_path):
    print("Downloading river shapefile...")
    wget.download("https://plotly.github.io/datasets/ne_50m_rivers_lake_centerlines.zip")

# Try to read the shapefile
try:
    geo_df = gpd.read_file("zip://ne_50m_rivers_lake_centerlines.zip")
except Exception as e:
    print(f"Error reading from zip: {e}")
    # Fallback to reading from extracted directory if it exists
    try:
        geo_df = gpd.read_file("River_centerlines/ne_50m_rivers_lake_centerlines.shp")
    except Exception as e:
        print(f"Error reading from directory: {e}")
        raise

# Initialize arrays for coordinates
lats = np.array([])
lons = np.array([])
names = np.array([])

# Extract river coordinates
for feature, name in zip(geo_df.geometry, geo_df.name):
    if isinstance(feature, shapely.geometry.linestring.LineString):
        linestrings = [feature]
    elif isinstance(feature, shapely.geometry.multilinestring.MultiLineString):
        linestrings = feature.geoms
    else:
        continue
    
    if name == "Rio Grande":
        for linestring in linestrings:
            x, y = linestring.xy
            lats = np.append(lats, y)
            lons = np.append(lons, x)
            names = np.append(names, [name]*len(y))
            lats = np.append(lats, None)
            lons = np.append(lons, None)
            names = np.append(names, None)

# Create river DataFrame
river_df = pd.DataFrame({"lats": lats, "lons": lons})

# Create map
map_fig = go.Figure()

# Add the river path using the extracted coordinates
map_fig.add_trace(
    go.Scattergeo(
        lon=river_df.lons,
        lat=river_df.lats,
        mode='lines',
        line=dict(
            width=2,
            color='blue'
        ),
        name='Rio Grande River'
    )
)

# Add stations with enhanced markers
for station_name, config in STATIONS.items():
    map_fig.add_trace(
        go.Scattergeo(
            lon=[config['coordinates'][1]],
            lat=[config['coordinates'][0]],
            text=[f"{station_name}<br>Current Flow: {predictions[station_name]['DISCHRG Value'].iloc[0]:.2f} ft³/s"],
            mode="markers+text",
            textposition="top center",
            marker=dict(
                size=10,
                color='red',
                line=dict(width=1, color="white"),
                symbol='circle'
            ),
            textfont=dict(color='black', size=10),
            name=station_name
        )
    )

# Update map settings for better zoom and size
map_fig.update_geos(
    showland=True,
    showcountries=True,
    showsubunits=True,
    subunitcolor="lightgray",
    landcolor='white',
    countrycolor='lightgray',
    scope="north america",
    lonaxis_range=[-110, -95],
    lataxis_range=[25, 35],
    projection_scale=3,
    center=dict(lat=30, lon=-103),
    showcoastlines=True,
    coastlinecolor="lightgray",
    showocean=True,
    oceancolor='lightblue'
)

# Update map layout
map_fig.update_layout(
    height=600,
    width=1100,
    margin=dict(l=0, r=0, t=0, b=0),
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)

# Export map figure
pio.write_html(map_fig, file="map.html", full_html=True, auto_open=True)

# If you want to combine both figures into a single HTML file:
combined_fig = go.Figure()
combined_fig.add_traces(fig1.data + map_fig.data)
combined_fig.update_layout(
    height=1200,  # Increased height to accommodate both figures
    grid={"rows": 2, "columns": 1, "pattern": "independent"},
    title="Rio Grande Streamflow Dashboard"
)

# Export combined figure
pio.write_html(combined_fig, file="index.html", full_html=True, auto_open=True)

# Or alternatively, use webbrowser module to open manually:
webbrowser.open('index.html')

# Dash App
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Rio Grande Streamflow Dashboard", 
            style={"textAlign": "center", "color": "#2c3e50", "marginBottom": "20px"}),
    
    # Map container with full width
    html.Div([
        dcc.Graph(
            figure=map_fig,
            style={
                'height': '600px',
                'width': '100%'
            }
        )
    ], style={
        'marginBottom': '20px',
        'width': '100%',
        'display': 'flex',
        'justifyContent': 'center'
    }),
    
    # Station selector
    html.Div([
        html.H3("Select Station", style={"textAlign": "center"}),
        dcc.Dropdown(
            id='station-selector',
            options=[{'label': station, 'value': station} for station in STATIONS.keys()],
            value=list(STATIONS.keys())[0],
            style={'width': '100%', 'marginBottom': '20px'}
        )
    ]),
    
    # Statistics panel with explanations
    html.Div([
        # Left side - metrics
        html.Div(id='station-stats', style={
            'width': '50%',
            'display': 'inline-block',
            'verticalAlign': 'top'
        }),
        
        # Right side - explanations
        html.Div([
            html.Div([
                html.H4("Metric Explanations", style={"color": "#2c3e50"}),
                html.Div([
                    html.P("Current Flow:", style={"fontWeight": "bold"}),
                    html.P("The most recent measured streamflow value in cubic feet per second (cfs)"),
                    
                    html.P("Predicted Flow:", style={"fontWeight": "bold"}),
                    html.P("The forecasted streamflow for the next 30 days based on our prediction model"),
                    
                    html.P("Change from Previous:", style={"fontWeight": "bold"}),
                    html.P("The percentage change in flow compared to the previous measurement"),
                    
                    html.P("30-Day Trend:", style={"fontWeight": "bold"}),
                    html.P("The overall direction of flow changes over the next 30 days"),
                    
                    html.P("Confidence Level:", style={"fontWeight": "bold"}),
                    html.P("Statistical confidence in the prediction accuracy (High: >90%, Medium: 70-90%, Low: <70%)")
                ], style={
                    'backgroundColor': '#f8f9fa',
                    'padding': '15px',
                    'borderRadius': '5px',
                    'border': '1px solid #dee2e6'
                })
            ])
        ], style={
            'width': '45%',
            'display': 'inline-block',
            'marginLeft': '5%',
            'verticalAlign': 'top'
        })
    ], style={'marginBottom': '20px'}),
    
    # Time series graph
    html.Div([
        html.H3("Streamflow Predictions", style={"textAlign": "center"}),
        dcc.Graph(figure=fig1, style={'height': '400px'})
    ]),
    
    # Footer
    html.Div([
        html.P("Data Sources:", style={"fontWeight": "bold"}),
        html.P("Predictions: 30-day forecast models for Rio Grande monitoring stations")
    ], style={'marginTop': '20px', 'textAlign': 'center', 'color': '#666'})
], style={
    'width': '100%',
    'maxWidth': '1200px',
    'margin': 'auto',
    'padding': '20px',
    'backgroundColor': '#f8f9fa'
})

@app.callback(
    Output('station-stats', 'children'),
    [Input('station-selector', 'value')]
)
def update_stats(selected_station):
    # Get the current data for the selected station
    station_data = predictions[selected_station]
    
    # Print debug information
    print(f"First few rows of data for {selected_station}:")
    print(station_data[['Date', 'DISCHRG Value']].head())
    
    # Calculate metrics
    current_flow = float(stats[selected_station]["Current Flow"]["value"])
    max_flow = float(stats[selected_station]["30-Day Max"]["value"])
    min_flow = float(stats[selected_station]["30-Day Min"]["value"])
    avg_flow = float(stats[selected_station]["30-Day Average"]["value"])
    
    # Calculate percent change (comparing to previous day)
    previous_flow = station_data['DISCHRG Value'].iloc[1]
    current_flow_raw = station_data['DISCHRG Value'].iloc[0]
    change_percent = ((current_flow_raw - previous_flow) / previous_flow) * 100 if previous_flow != 0 else 0
    
    # Determine trend using first and last week averages
    first_week = station_data['DISCHRG Value'].iloc[:7].mean()
    last_week = station_data['DISCHRG Value'].iloc[-7:].mean()
    trend = "Increasing" if last_week > first_week else "Decreasing" if last_week < first_week else "Stable"
    
    # Calculate confidence based on prediction variance
    prediction_std = station_data['DISCHRG Value'].std()
    confidence = "High" if prediction_std < 10 else "Medium" if prediction_std < 20 else "Low"
    
    return html.Div([
        html.H4("Station Metrics", style={"color": "#2c3e50"}),
        html.Div([
            html.P(f"Current Flow: {current_flow:.1f} cfs"),
            html.P(f"30-Day Maximum: {max_flow:.1f} cfs"),
            html.P(f"30-Day Minimum: {min_flow:.1f} cfs"),
            html.P(f"30-Day Average: {avg_flow:.1f} cfs"),
            html.P(f"Change from Previous: {change_percent:.1f}%"),
            html.P(f"30-Day Trend: {trend}"),
            html.P(f"Prediction Confidence: {confidence}")
        ], style={
            'backgroundColor': '#f8f9fa',
            'padding': '15px',
            'borderRadius': '5px',
            'border': '1px solid #dee2e6'
        })
    ])

if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0")