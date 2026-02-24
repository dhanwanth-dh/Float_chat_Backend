import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

tsunami_model = None
scaler = None
model_path = Path("models/tsunami_model.pkl")
scaler_path = Path("models/tsunami_scaler.pkl")

def calculate_tsunami_risk_score(df_region):
    """Calculate tsunami risk based on oceanographic indicators"""
    if df_region.empty:
        return 0
    
    risk_score = 0
    
    # Pressure anomalies (sudden changes indicate seismic activity)
    if "pressure" in df_region.columns:
        pressure_std = df_region["pressure"].std()
        pressure_range = df_region["pressure"].max() - df_region["pressure"].min()
        if pressure_std > 500:
            risk_score += 30
        elif pressure_std > 300:
            risk_score += 15
    
    # Temperature anomalies
    if "temperature" in df_region.columns:
        temp_mean = df_region["temperature"].mean()
        temp_std = df_region["temperature"].std()
        if temp_std > 5:
            risk_score += 20
        if temp_mean < 5:  # Cold deep water displacement
            risk_score += 15
    
    # Salinity variations (water mass displacement)
    if "salinity" in df_region.columns:
        sal_std = df_region["salinity"].std()
        if sal_std > 0.5:
            risk_score += 25
    
    # Depth profile irregularities
    if "pressure" in df_region.columns:
        depth_gradient = np.gradient(df_region["pressure"].values)
        if np.std(depth_gradient) > 100:
            risk_score += 10
    
    return min(risk_score, 100)

def analyze_tsunami_risk_by_region(df):
    """Analyze tsunami risk for different geographic regions"""
    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        return []
    
    regions = [
        {"name": "Alaska Coast", "lat_range": (51, 71), "lon_range": (-180, -130), "base_risk": 25},
        {"name": "Pacific Northwest (Canada/USA)", "lat_range": (42, 60), "lon_range": (-135, -122), "base_risk": 20},
        {"name": "Japan Coast", "lat_range": (30, 46), "lon_range": (129, 146), "base_risk": 30},
        {"name": "Indonesia Region", "lat_range": (-11, 6), "lon_range": (95, 141), "base_risk": 35},
        {"name": "Chile Coast", "lat_range": (-56, -17), "lon_range": (-76, -66), "base_risk": 28},
        {"name": "New Zealand", "lat_range": (-47, -34), "lon_range": (166, 179), "base_risk": 22},
        {"name": "Philippines", "lat_range": (5, 21), "lon_range": (117, 127), "base_risk": 30},
        {"name": "Peru Coast", "lat_range": (-18, -1), "lon_range": (-82, -70), "base_risk": 25},
        {"name": "Aleutian Islands", "lat_range": (51, 55), "lon_range": (-180, -160), "base_risk": 27},
        {"name": "Caribbean", "lat_range": (10, 27), "lon_range": (-85, -60), "base_risk": 18}
    ]
    
    risk_results = []
    
    for region in regions:
        region_df = df[
            (df["latitude"] >= region["lat_range"][0]) & 
            (df["latitude"] <= region["lat_range"][1]) &
            (df["longitude"] >= region["lon_range"][0]) & 
            (df["longitude"] <= region["lon_range"][1])
        ]
        
        if len(region_df) > 10:
            data_risk = calculate_tsunami_risk_score(region_df)
            total_risk = (region["base_risk"] + data_risk) / 2
            
            # Calculate confidence based on data availability
            confidence = min(len(region_df) / 100, 1.0) * 100
            
            risk_results.append({
                "region": region["name"],
                "risk_score": round(total_risk, 1),
                "confidence": round(confidence, 1),
                "data_points": len(region_df),
                "indicators": {
                    "pressure_anomaly": round(region_df["pressure"].std(), 2) if "pressure" in region_df.columns else 0,
                    "temp_variation": round(region_df["temperature"].std(), 2) if "temperature" in region_df.columns else 0,
                    "salinity_variation": round(region_df["salinity"].std(), 2) if "salinity" in region_df.columns else 0
                }
            })
    
    # Sort by risk score
    risk_results.sort(key=lambda x: x["risk_score"], reverse=True)
    return risk_results

def predict_tsunami_timeframe(risk_score):
    """Estimate timeframe based on risk score"""
    if risk_score >= 70:
        return "High risk - Possible within 1-6 months"
    elif risk_score >= 50:
        return "Moderate risk - Possible within 6-12 months"
    elif risk_score >= 30:
        return "Low-moderate risk - Possible within 1-2 years"
    else:
        return "Low risk - No immediate threat detected"

def generate_tsunami_analysis(df, user_prompt):
    """Generate comprehensive tsunami risk analysis"""
    risk_by_region = analyze_tsunami_risk_by_region(df)
    
    if not risk_by_region:
        return {
            "summary": "Insufficient data to perform tsunami risk analysis. Need geographic coverage of high-risk coastal regions.",
            "top_risks": [],
            "recommendations": []
        }
    
    top_3 = risk_by_region[:3]
    
    summary = f"Based on analysis of {len(df)} oceanographic measurements, tsunami risk assessment:\n\n"
    
    for i, region in enumerate(top_3, 1):
        summary += f"{i}. **{region['region']}** - Risk Score: {region['risk_score']}/100\n"
        summary += f"   Timeframe: {predict_tsunami_timeframe(region['risk_score'])}\n"
        summary += f"   Confidence: {region['confidence']}% (based on {region['data_points']} measurements)\n\n"
    
    summary += "\n**Key Indicators:**\n"
    for region in top_3:
        summary += f"- {region['region']}: "
        indicators = []
        if region['indicators']['pressure_anomaly'] > 300:
            indicators.append("High pressure variations")
        if region['indicators']['temp_variation'] > 3:
            indicators.append("Temperature anomalies")
        if region['indicators']['salinity_variation'] > 0.3:
            indicators.append("Salinity fluctuations")
        summary += ", ".join(indicators) if indicators else "Normal conditions"
        summary += "\n"
    
    recommendations = [
        "Monitor seismic activity in high-risk regions",
        "Maintain tsunami early warning systems",
        "Conduct regular coastal evacuation drills",
        "Update emergency response protocols"
    ]
    
    return {
        "summary": summary,
        "top_risks": risk_by_region[:5],
        "all_regions": risk_by_region,
        "recommendations": recommendations,
        "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
