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
import socket

# Station configurations
STATIONS = {
    'Rio Milco': {
        'file': 'Predictions/preds_30day_riomilco.csv',
        'color': 'rgb(0, 0, 255)',  # Changed from 'blue' to RGB
        'coordinates': [37.4833, -106.8167]
    },
    'El Paso': {
        'file': 'Predictions/preds_30day_elpaso.csv',
        'color': 'rgb(255, 0, 0)',
        'coordinates': [31.7619, -106.4850]
    },
    'Big Bend': {
        'file': 'Predictions/preds_30day_bigbend.csv',
        'color': 'rgb(0, 255, 0)',
        'coordinates': [29.2498, -103.2502]
    }
}

# Add debug prints to verify color consistency
print("\nVerifying station colors:")
for station_name, config in STATIONS.items():
    print(f"{station_name}: {config['color']}")

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
    if station_name == 'Rio Milco':
        marker_color = 'blue'
    elif station_name == 'El Paso':
        marker_color = 'red'
    else:  # Big Bend
        marker_color = 'green'
        
    fig1.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["DISCHRG Value"],
            name=f"{station_name}",
            line=dict(color=marker_color)  # Using same explicit color
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

# Add the river path
map_fig.add_trace(
    go.Scattergeo(
        lon=river_df.lons,
        lat=river_df.lats,
        mode='lines',
        line=dict(width=2, color='lightblue'),
        name='Rio Grande River'
    )
)

# First, create a dataframe with the station data
stations_df = pd.DataFrame([
    {
        'station': 'Rio Milco',
        'lat': 37.4833,
        'lon': -106.8167,
        'color': 'blue'
    },
    {
        'station': 'El Paso',
        'lat': 31.7619,
        'lon': -106.4850,
        'color': 'red'
    },
    {
        'station': 'Big Bend',
        'lat': 29.2498,
        'lon': -103.2502,
        'color': 'green'
    }
])

# Add stations using the dataframe
for _, row in stations_df.iterrows():
    map_fig.add_trace(
        go.Scattergeo(
            lat=[row['lat']],
            lon=[row['lon']],
            text=[f"{row['station']}<br>Current Flow: {predictions[row['station']]['DISCHRG Value'].iloc[0]:.2f} ft³/s"],
            mode="markers+text",
            textposition="top center",
            marker=dict(
                size=12,
                color=row['color'],
                line=dict(width=2, color='white'),
                symbol='circle',
                opacity=1.0
            ),
            textfont=dict(color='black', size=12),
            name=row['station'],
            showlegend=True
        )
    )

# Update map settings
map_fig.update_geos(
    showland=True,
    showcountries=True,
    showsubunits=True,
    subunitcolor="lightgray",
    landcolor='white',
    countrycolor='lightgray',
    scope="north america",
    lonaxis_range=[-110, -100],
    lataxis_range=[26, 40],
    projection_scale=3.5,
    center=dict(lat=33, lon=-105),
    showcoastlines=True,
    coastlinecolor="lightgray",
    showocean=True,
    oceancolor='lightblue'
)

# Update layout
map_fig.update_layout(
    height=700,
    width=1200,
    margin=dict(l=0, r=0, t=0, b=0),
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)

# Dash App
app = Dash(__name__)

app.layout = html.Div([
    html.Div(id='dummy-timestamp', style={'display': 'none'}, children=str(datetime.now())),
    
    html.H1("Rio Grande Streamflow Dashboard", 
            style={"textAlign": "center", "color": "#2c3e50", "marginBottom": "20px"}),
    
    # Add forecast period selector
    html.Div([
        html.H3("Select Forecast Period", style={"textAlign": "center"}),
        dcc.RadioItems(
            id='forecast-period',
            options=[
                {'label': '1-Day Forecast', 'value': '1day'},
                {'label': '4-Day Forecast', 'value': '4day'},
                {'label': '7-Day Forecast', 'value': '7day'},
                {'label': '30-Day Forecast', 'value': '30day'}
            ],
            value='7day',  # default value
            style={'display': 'flex', 'justifyContent': 'center', 'gap': '20px', 'marginBottom': '20px'}
        )
    ]),
    
    # Map container
    html.Div([
        dcc.Graph(id='map-graph', style={'width': '100%', 'height': '600px'})
    ], style={
        'width': '90vw',
        'maxWidth': '1200px',
        'margin': 'auto',
        'marginTop': '20px',
        'marginBottom': '20px'
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

# Add callback to update map based on forecast period
@app.callback(
    Output('map-graph', 'figure'),
    [Input('forecast-period', 'value')]
)
def update_map(forecast_period):
    # Load the appropriate prediction files
    predictions = {}
    for station in STATIONS:
        file_path = f'Predictions/preds_{forecast_period}_{station.lower().replace(" ", "")}.csv'
        predictions[station] = pd.read_csv(file_path)
    
    # Create new map figure
    map_fig = go.Figure()
    
    # Add river path
    map_fig.add_trace(
        go.Scattergeo(
            lon=river_df.lons,
            lat=river_df.lats,
            mode='lines',
            line=dict(width=2, color='lightblue'),
            name='Rio Grande River'
        )
    )
    
    # Add station markers
    for station in STATIONS:
        map_fig.add_trace(
            go.Scattergeo(
                lat=[STATIONS[station]['coordinates'][0]],
                lon=[STATIONS[station]['coordinates'][1]],
                text=[f"{station}<br>Current Flow: {predictions[station]['DISCHRG Value'].iloc[0]:.2f} ft³/s"],
                mode="markers+text",
                textposition="top center",
                marker=dict(
                    size=12,
                    color=STATIONS[station]['color'],
                    line=dict(width=2, color='white'),
                    symbol='circle'
                ),
                textfont=dict(color='black', size=12),
                name=station
            )
        )
    
    # Update map settings
    map_fig.update_geos(
        showland=True,
        showcountries=True,
        showsubunits=True,
        subunitcolor="lightgray",
        landcolor='white',
        countrycolor='lightgray',
        scope="north america",
        lonaxis_range=[-110, -100],
        lataxis_range=[26, 40],
        projection_scale=3.5,
        center=dict(lat=33, lon=-105),
        showcoastlines=True,
        coastlinecolor="lightgray",
        showocean=True,
        oceancolor='lightblue'
    )
    
    map_fig.update_layout(
        height=700,
        width=1200,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return map_fig

def calculate_confidence(station_data, forecast_period):
    """
    Calculate prediction confidence with adjusted thresholds for river flow data
    """
    values = station_data['DISCHRG Value']
    
    # Calculate coefficient of variation (CV)
    mean_flow = values.mean()
    std_flow = values.std()
    cv = (std_flow / mean_flow) if mean_flow != 0 else float('inf')
    
    # Calculate outlier score with more lenient thresholds
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    outlier_count = ((values < (q1 - 2.0 * iqr)) | (values > (q3 + 2.0 * iqr))).sum()  # More lenient outlier definition
    outlier_score = outlier_count / len(values) if len(values) > 0 else 0
    
    # Adjusted base confidence scores
    if forecast_period == '1day':
        base_score = 0.95  # Higher base confidence for 1-day
    elif forecast_period == '4day':
        base_score = 0.85
    elif forecast_period == '7day':
        base_score = 0.80
    else:  # 30day
        base_score = 0.75  # Higher base confidence for long-term
    
    # More lenient penalties
    cv_penalty = min(0.2, cv * 0.3)  # Reduced CV penalty
    outlier_penalty = outlier_score * 0.1  # Reduced outlier penalty
    
    final_score = base_score - cv_penalty - outlier_penalty
    
    # Adjusted thresholds for confidence levels
    if final_score >= 0.70:
        return "High"
    elif final_score >= 0.45:
        return "Medium"
    else:
        return "Low"

# Update the stats callback to use new confidence calculation
@app.callback(
    Output('station-stats', 'children'),
    [Input('station-selector', 'value'),
     Input('forecast-period', 'value')]
)
def update_stats(selected_station, forecast_period):
    # Load the appropriate prediction file for the selected period
    file_path = f'Predictions/preds_{forecast_period}_{selected_station.lower().replace(" ", "")}.csv'
    station_data = pd.read_csv(file_path)
    
    # Calculate metrics based on the loaded data
    current_flow = station_data['DISCHRG Value'].iloc[0]
    
    # For 1-day forecast, use different calculations
    if forecast_period == '1day':
        max_flow = current_flow
        min_flow = current_flow
        avg_flow = current_flow
        trend = "N/A for 1-day forecast"
        change_percent = 0
    else:
        max_flow = station_data['DISCHRG Value'].max()
        min_flow = station_data['DISCHRG Value'].min()
        avg_flow = station_data['DISCHRG Value'].mean()
        
        # Calculate percent change (comparing to next day)
        if len(station_data) > 1:
            next_day_flow = station_data['DISCHRG Value'].iloc[1]
            change_percent = ((next_day_flow - current_flow) / current_flow) * 100 if current_flow != 0 else 0
        else:
            change_percent = 0
        
        # Determine trend (using more data points for 30-day forecast)
        if forecast_period == '30day' and len(station_data) > 15:
            # For 30-day, look at broader trend
            first_half_avg = station_data['DISCHRG Value'].iloc[:15].mean()
            second_half_avg = station_data['DISCHRG Value'].iloc[15:].mean()
            trend = "Increasing" if second_half_avg > first_half_avg else "Decreasing" if second_half_avg < first_half_avg else "Stable"
        elif len(station_data) > 1:
            last_value = station_data['DISCHRG Value'].iloc[-1]
            trend = "Increasing" if last_value > current_flow else "Decreasing" if last_value < current_flow else "Stable"
        else:
            trend = "N/A"
    
    # Replace the old confidence calculation with:
    confidence = calculate_confidence(station_data, forecast_period)
    
    return html.Div([
        html.H4(f"Station Metrics ({forecast_period} forecast)", style={"color": "#2c3e50"}),
        html.Div([
            html.P(f"Current Flow: {current_flow:.1f} cfs"),
            html.P(f"Maximum Flow: {max_flow:.1f} cfs"),
            html.P(f"Minimum Flow: {min_flow:.1f} cfs"),
            html.P(f"Average Flow: {avg_flow:.1f} cfs"),
            html.P(f"Predicted Change: {change_percent:.1f}%"),
            html.P(f"Trend: {trend}"),
            html.P(f"Prediction Confidence: {confidence}")
        ], style={
            'backgroundColor': '#f8f9fa',
            'padding': '15px',
            'borderRadius': '5px',
            'border': '1px solid #dee2e6'
        })
    ])

# Important: For AWS Elastic Beanstalk
application = app.server

if __name__ == '__main__':
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    port = find_free_port()
    print(f"Starting server on port {port}")
    application.run(debug=True, port=port)