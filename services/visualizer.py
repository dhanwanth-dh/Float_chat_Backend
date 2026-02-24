import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def temperature_depth_plot(df: pd.DataFrame):
    """Generate variable vs depth plot"""
    if df.empty or "pressure" not in df.columns:
        return None
    
    variable = "temperature" if "temperature" in df.columns else "salinity"
    if variable not in df.columns:
        return None
    
    # Sample data if too large
    plot_df = df.sample(min(1000, len(df))) if len(df) > 1000 else df
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df[variable],
        y=plot_df["pressure"],
        mode="markers",
        marker=dict(
            size=5,
            color=plot_df[variable],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=variable.capitalize())
        ),
        name=variable.capitalize(),
        text=[f"{variable}: {v:.2f}<br>Depth: {d:.0f} dbar" for v, d in zip(plot_df[variable], plot_df["pressure"])],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"{variable.capitalize()} vs Depth Profile",
        xaxis_title=f"{variable.capitalize()} ({'°C' if variable == 'temperature' else 'PSU'})",
        yaxis_title="Depth (dbar)",
        yaxis=dict(autorange="reversed"),
        height=450,
        template="plotly_white",
        hovermode='closest'
    )
    
    return fig.to_json()

def generate_heatmap(df: pd.DataFrame, variable: str):
    """Generate geographic heatmap"""
    if df.empty or variable not in df.columns:
        return None
    
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return None
    
    # Sample for performance
    plot_df = df.sample(min(2000, len(df))) if len(df) > 2000 else df
    
    # Create grid for heatmap
    lat_bins = np.linspace(plot_df["latitude"].min(), plot_df["latitude"].max(), 30)
    lon_bins = np.linspace(plot_df["longitude"].min(), plot_df["longitude"].max(), 30)
    
    plot_df['lat_bin'] = pd.cut(plot_df['latitude'], bins=lat_bins)
    plot_df['lon_bin'] = pd.cut(plot_df['longitude'], bins=lon_bins)
    
    heatmap_data = plot_df.groupby(['lat_bin', 'lon_bin'])[variable].mean().reset_index()
    heatmap_data['lat'] = heatmap_data['lat_bin'].apply(lambda x: x.mid)
    heatmap_data['lon'] = heatmap_data['lon_bin'].apply(lambda x: x.mid)
    
    fig = go.Figure(go.Scattermapbox(
        lat=heatmap_data['lat'],
        lon=heatmap_data['lon'],
        mode='markers',
        marker=dict(
            size=15,
            color=heatmap_data[variable],
            colorscale='RdYlBu_r' if variable == 'temperature' else 'Viridis',
            showscale=True,
            colorbar=dict(title=variable.capitalize()),
            opacity=0.7
        ),
        text=[f"{variable}: {v:.2f}" for v in heatmap_data[variable]],
        hovertemplate='Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<br>%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=plot_df["latitude"].mean(), lon=plot_df["longitude"].mean()),
            zoom=2
        ),
        height=450,
        margin={"r":0,"t":30,"l":0,"b":0},
        title=f"{variable.capitalize()} Geographic Distribution"
    )
    
    return fig.to_json()

def generate_probability_distribution(df: pd.DataFrame, variable: str):
    """Generate probability distribution histogram"""
    if df.empty or variable not in df.columns:
        return None
    
    # Calculate statistics
    mean_val = df[variable].mean()
    median_val = df[variable].median()
    std_val = df[variable].std()
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=df[variable],
        nbinsx=50,
        name="Frequency",
        marker_color="#3b82f6",
        opacity=0.7,
        hovertemplate='Value: %{x:.2f}<br>Count: %{y}<extra></extra>'
    ))
    
    # Add mean line
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {mean_val:.2f}", annotation_position="top")
    
    # Add median line
    fig.add_vline(x=median_val, line_dash="dash", line_color="green",
                  annotation_text=f"Median: {median_val:.2f}", annotation_position="bottom")
    
    fig.update_layout(
        title=f"{variable.capitalize()} Distribution (σ={std_val:.2f})",
        xaxis_title=f"{variable.capitalize()} ({'°C' if variable == 'temperature' else 'PSU'})",
        yaxis_title="Frequency",
        height=350,
        template="plotly_white",
        showlegend=False,
        bargap=0.1
    )
    
    return fig.to_json()
