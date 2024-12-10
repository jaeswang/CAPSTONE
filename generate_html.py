import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import geopandas as gpd
import shapely.geometry
import wget
import os
import webbrowser
from plotly.subplots import make_subplots

# Station configurations
STATIONS = {
    'Rio Milco': {
        'file': 'Predictions/preds_30day_riomilco.csv',
        'color': 'blue',
        'coordinates': [37.72472199, -107.2556094]
    },
    'El Paso': {
        'file': 'Predictions/preds_30day_elpaso.csv',
        'color': 'red',
        'coordinates': [31.8032, -106.541]
    },
    'Big Bend': {
        'file': 'Predictions/preds_30day_bigbend.csv',
        'color': 'green',
        'coordinates': [29.18333333, -102.97527778]
    }
}

# Load predictions for all stations
predictions = {}
for station_name, config in STATIONS.items():
    df = pd.read_csv(config['file'])
    df["Date"] = pd.to_datetime(df["Date"])
    predictions[station_name] = df

# Create time series figure
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
                size=12,
                color='red',
                line=dict(width=2, color="white"),
                symbol='circle'
            ),
            textfont=dict(
                color='black',
                size=12,
                family="Arial Black"
            ),
            name=station_name
        )
    )

# Update map settings
map_fig.update_geos(
    showland=True,
    showcountries=True,
    showsubunits=True,
    subunitcolor="dimgray",
    subunitwidth=1,
    landcolor='white',
    countrycolor='black',
    countrywidth=2,
    scope="north america",
    lonaxis_range=[-125, -80],
    lataxis_range=[20, 45],
    projection_scale=1.5,
    center=dict(lat=32, lon=-102.5),
    showcoastlines=True,
    coastlinecolor="dimgray",
    coastlinewidth=1,
    showocean=True,
    oceancolor='lightblue',
    framecolor="gray",
    framewidth=1,
    resolution=50
)

# Update map layout
map_fig.update_layout(
    height=800,
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

# Create combined figure with subplots
combined_fig = make_subplots(
    rows=2, 
    cols=1,
    specs=[[{"type": "xy"}],  # First row for time series
           [{"type": "scattergeo"}]],  # Second row for map
    vertical_spacing=0.12,    # Increased spacing between plots
    subplot_titles=("30-Day Streamflow Predictions", "Station Locations")
)

# Add time series traces to first subplot
for trace in fig1.data:
    combined_fig.add_trace(trace, row=1, col=1)

# Add map traces to second subplot
for trace in map_fig.data:
    combined_fig.add_trace(trace, row=2, col=1)

# Update layout with original title format
combined_fig.update_layout(
    height=1400,
    title="Rio Grande Streamflow Dashboard",
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    ),
    # Add annotations settings to adjust subplot titles
    annotations=[
        dict(
            text="30-Day Streamflow Predictions",
            x=0.5,
            y=0.95,
            xref="paper",
            yref="paper",
            showarrow=False
        ),
        dict(
            text="Station Locations",
            x=0.5,
            y=0.45,  # Adjusted position for second subplot title
            xref="paper",
            yref="paper",
            showarrow=False
        )
    ]
)

# Update geo settings
combined_fig.update_geos(
    showland=True,
    showcountries=True,
    showsubunits=True,
    subunitcolor="dimgray",
    subunitwidth=1,
    landcolor='white',
    countrycolor='black',
    countrywidth=2,
    scope="north america",
    lonaxis_range=[-125, -80],
    lataxis_range=[20, 45],
    projection_scale=1.5,
    center=dict(lat=32, lon=-102.5),
    showcoastlines=True,
    coastlinecolor="dimgray",
    coastlinewidth=1,
    showocean=True,
    oceancolor='lightblue',
    framecolor="gray",
    framewidth=1,
    resolution=50
)

# Update xaxis without button positioning
combined_fig.update_xaxes(
    rangeslider=dict(visible=True, thickness=0.05),
    rangeselector=dict(
        buttons=list([
            dict(count=7, label="1w", step="day", stepmode="backward"),
            dict(count=14, label="2w", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(step="all", label="All")
        ])
    ),
    row=1, col=1
)
combined_fig.update_yaxes(title="Streamflow (ft³/s)", row=1, col=1)

# Export figures
print("Generating HTML files...")
pio.write_html(fig1, file="timeseries.html", full_html=True)
pio.write_html(map_fig, file="map.html", full_html=True)

# Export combined figure
pio.write_html(combined_fig, file="index.html", full_html=True)

print("Opening index.html in browser...")
webbrowser.open('file://' + os.path.realpath('index.html')) 