import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

model = None
model_path = Path("models/argo_model.pkl")

def train_model(df):
    global model
    
    if df.empty or len(df) < 100:
        return None
    
    features = ["latitude", "longitude", "pressure"]
    target = "temperature"
    
    required_cols = features + [target]
    if not all(col in df.columns for col in required_cols):
        return None
    
    X = df[features].dropna()
    y = df.loc[X.index, target]
    
    if len(X) < 100:
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    model_path.parent.mkdir(exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    score = model.score(X_test, y_test)
    return {"r2_score": round(score, 3), "samples": len(X_train)}

def load_model():
    global model
    
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return True
    return False

def predict_temperature(latitude, longitude, pressure):
    global model
    
    if model is None:
        load_model()
    
    if model is None:
        return None
    
    X = np.array([[latitude, longitude, pressure]])
    return round(model.predict(X)[0], 2)

def analyze_anomalies(df, variable="temperature"):
    if df.empty or variable not in df.columns:
        return []
    
    mean = df[variable].mean()
    std = df[variable].std()
    
    anomalies = []
    
    extreme_high = df[df[variable] > mean + 2*std]
    extreme_low = df[df[variable] < mean - 2*std]
    
    if len(extreme_high) > 0:
        anomalies.append({
            "type": "high_anomaly",
            "severity": "warning",
            "message": f"Detected {len(extreme_high)} unusually high {variable} readings",
            "value": round(extreme_high[variable].mean(), 2)
        })
    
    if len(extreme_low) > 0:
        anomalies.append({
            "type": "low_anomaly",
            "severity": "warning",
            "message": f"Detected {len(extreme_low)} unusually low {variable} readings",
            "value": round(extreme_low[variable].mean(), 2)
        })
    
    if len(df) < 50:
        anomalies.append({
            "type": "data_sparse",
            "severity": "info",
            "message": "Limited data available for this region",
            "value": len(df)
        })
    
    return anomalies

def get_location_insights(df, query):
    if df.empty:
        return "No data available for analysis."
    
    insights = []
    
    if "latitude" in df.columns:
        avg_lat = df["latitude"].mean()
        if avg_lat > 60:
            insights.append("Arctic region: Expect cold temperatures and seasonal ice coverage.")
        elif avg_lat < -60:
            insights.append("Antarctic region: Extremely cold waters with high salinity.")
        elif abs(avg_lat) < 23.5:
            insights.append("Tropical region: Warm surface waters with strong stratification.")
        else:
            insights.append("Mid-latitude region: Moderate temperatures with seasonal variations.")
    
    if "pressure" in df.columns:
        avg_depth = df["pressure"].mean()
        if avg_depth < 100:
            insights.append("Surface layer: High biological activity and temperature variability.")
        elif avg_depth < 1000:
            insights.append("Intermediate depth: Transition zone with decreasing temperature.")
        else:
            insights.append("Deep ocean: Cold, stable conditions with minimal variability.")
    
    if "temperature" in df.columns:
        temp = df["temperature"].mean()
        insights.append(f"Current average temperature: {round(temp, 2)}°C")
    
    if "salinity" in df.columns:
        sal = df["salinity"].mean()
        insights.append(f"Current average salinity: {round(sal, 2)} PSU")
    
    return " ".join(insights)

def calculate_probabilities(df, variable="temperature"):
    if df.empty or variable not in df.columns:
        return {}
    
    values = df[variable].values
    percentiles = np.percentile(values, [10, 25, 50, 75, 90])
    
    return {
        "p10": round(percentiles[0], 2),
        "p25": round(percentiles[1], 2),
        "median": round(percentiles[2], 2),
        "p75": round(percentiles[3], 2),
        "p90": round(percentiles[4], 2),
        "mean": round(values.mean(), 2),
        "std": round(values.std(), 2)
    }

def summarize(prompt: str, stats: dict):
    variable = stats.get("variable", "temperature")
    mean_val = stats.get("mean_value", 0)
    min_val = stats.get("min_value", 0)
    max_val = stats.get("max_value", 0)
    depth_range = stats.get("depth_range", (0, 0))
    data_points = stats.get("data_points", 0)
    
    summary = f"Based on {data_points} ARGO measurements, the {variable} ranges from {min_val} to {max_val}, "
    summary += f"with an average of {mean_val}. "
    summary += f"Data spans depths from {depth_range[0]} to {depth_range[1]} dbar. "
    
    if variable == "temperature":
        if mean_val < 5:
            summary += "These cold temperatures suggest polar or deep ocean conditions."
        elif mean_val > 20:
            summary += "These warm temperatures indicate tropical surface waters."
        else:
            summary += "These moderate temperatures are typical of mid-latitude oceans."
    elif variable == "salinity":
        if mean_val < 34:
            summary += "Lower salinity may indicate freshwater influence or high precipitation areas."
        elif mean_val > 36:
            summary += "Higher salinity suggests evaporation-dominated regions."
    
    return summary
