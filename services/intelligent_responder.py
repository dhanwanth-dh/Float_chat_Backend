import pandas as pd
import numpy as np

def classify_query_intent(prompt):
    """Classify user query into specific intent categories"""
    prompt_lower = prompt.lower()
    
    # Tsunami/Disaster
    if any(word in prompt_lower for word in ["tsunami", "disaster", "flood", "earthquake", "hazard"]):
        return "tsunami"
    
    # Glacier/Ice melting
    if any(word in prompt_lower for word in ["glacier", "ice", "melting", "arctic", "antarctic", "polar"]):
        return "glacier_ice"
    
    # Marine life
    if any(word in prompt_lower for word in ["fish", "whale", "marine life", "coral", "ecosystem", "species", "biodiversity"]):
        return "marine_life"
    
    # Water pressure specific
    if "pressure" in prompt_lower and not any(word in prompt_lower for word in ["temperature", "salinity"]):
        return "pressure"
    
    # Salinity specific
    if "salinity" in prompt_lower and not any(word in prompt_lower for word in ["temperature", "pressure"]):
        return "salinity"
    
    # Temperature specific
    if any(word in prompt_lower for word in ["temperature", "warm", "cold", "heat"]):
        return "temperature"
    
    # Ocean currents
    if any(word in prompt_lower for word in ["current", "circulation", "flow", "gulf stream"]):
        return "currents"
    
    # Climate change
    if any(word in prompt_lower for word in ["climate", "warming", "change", "carbon"]):
        return "climate"
    
    # Regional query
    if any(word in prompt_lower for word in ["indian ocean", "pacific", "atlantic", "southern ocean", "arctic ocean"]):
        return "regional"
    
    return "general"

def extract_region_from_prompt(prompt):
    """Extract specific ocean region from prompt"""
    prompt_lower = prompt.lower()
    
    if "indian ocean" in prompt_lower or "india" in prompt_lower:
        return {"name": "Indian Ocean", "lat_range": (-40, 30), "lon_range": (40, 120)}
    elif "pacific" in prompt_lower:
        return {"name": "Pacific Ocean", "lat_range": (-60, 60), "lon_range": (120, -70)}
    elif "atlantic" in prompt_lower:
        return {"name": "Atlantic Ocean", "lat_range": (-60, 60), "lon_range": (-70, 20)}
    elif "southern ocean" in prompt_lower or "antarctica" in prompt_lower or "antarctic" in prompt_lower:
        return {"name": "Southern Ocean/Antarctica", "lat_range": (-90, -40), "lon_range": (-180, 180)}
    elif "arctic" in prompt_lower:
        return {"name": "Arctic Ocean", "lat_range": (60, 90), "lon_range": (-180, 180)}
    
    return None

def generate_pressure_response(df, region_info):
    """Generate response for water pressure queries"""
    if df.empty or "pressure" not in df.columns:
        return "Insufficient pressure data available."
    
    avg_pressure = df["pressure"].mean()
    max_pressure = df["pressure"].max()
    min_pressure = df["pressure"].min()
    
    region_name = region_info["name"] if region_info else "the analyzed region"
    
    response = f"**Water Pressure Analysis for {region_name}:**\n\n"
    response += f"Average depth: {round(avg_pressure, 1)} dbar (approximately {round(avg_pressure, 0)} meters)\n"
    response += f"Depth range: {round(min_pressure, 1)} to {round(max_pressure, 1)} dbar\n"
    response += f"Data points: {len(df)} measurements\n\n"
    
    if avg_pressure < 100:
        response += "This represents primarily surface and shallow water measurements. Surface pressure is crucial for understanding wave dynamics and near-surface ocean processes."
    elif avg_pressure < 1000:
        response += "This covers the upper ocean and thermocline region. Pressure at these depths affects nutrient distribution and marine life habitats."
    else:
        response += "This includes deep ocean measurements. High pressure at these depths creates unique environments for deep-sea organisms and affects ocean circulation patterns."
    
    return response

def generate_glacier_ice_response(df, prompt):
    """Generate response for glacier/ice melting queries"""
    prompt_lower = prompt.lower()
    
    if "antarctica" in prompt_lower or "antarctic" in prompt_lower:
        region = "Antarctic"
        antarctic_df = df[(df["latitude"] < -60)] if "latitude" in df.columns else df
        
        if not antarctic_df.empty and "temperature" in antarctic_df.columns:
            avg_temp = antarctic_df["temperature"].mean()
            response = f"**Antarctic Glacier and Ice Analysis:**\n\n"
            response += f"Current average temperature: {round(avg_temp, 2)}°C\n"
            response += f"Measurements: {len(antarctic_df)} data points\n\n"
            
            if avg_temp > -1:
                response += "⚠️ **Critical Finding:** Temperatures are above typical Antarctic levels. "
                response += "Warmer ocean water accelerates ice shelf melting from below. "
                response += "This contributes to sea level rise and disrupts ocean circulation patterns.\n\n"
            else:
                response += "Temperatures are within expected range for Antarctic waters. "
                response += "However, even small increases can significantly impact ice stability.\n\n"
            
            response += "**Key Impacts:**\n"
            response += "- Ice shelf melting increases freshwater input\n"
            response += "- Affects global ocean salinity and circulation\n"
            response += "- Contributes to sea level rise\n"
            response += "- Disrupts marine ecosystems adapted to cold conditions"
            
            return response
    
    response = "**Polar Ice and Glacier Melting:**\n\n"
    response += "Ocean temperature and salinity data indicate:\n"
    response += "- Warmer ocean currents accelerate ice melting from below\n"
    response += "- Freshwater from melting ice reduces ocean salinity\n"
    response += "- This affects global ocean circulation (thermohaline circulation)\n"
    response += "- Sea level rise impacts coastal communities worldwide\n\n"
    response += "Current data shows ongoing changes in polar regions that require continuous monitoring."
    
    return response

def generate_marine_life_response(df, region_info):
    """Generate response for marine life queries"""
    if df.empty:
        return "Insufficient data for marine life analysis."
    
    region_name = region_info["name"] if region_info else "this region"
    
    response = f"**Marine Life Conditions in {region_name}:**\n\n"
    
    if "temperature" in df.columns:
        avg_temp = df["temperature"].mean()
        response += f"Water temperature: {round(avg_temp, 2)}°C\n"
        
        if avg_temp < 5:
            response += "- Cold water supports: Krill, seals, penguins, cold-water fish\n"
            response += "- High oxygen levels support dense populations\n"
        elif avg_temp < 15:
            response += "- Temperate conditions support: Diverse fish species, marine mammals, kelp forests\n"
            response += "- Rich biodiversity zone\n"
        else:
            response += "- Warm water supports: Coral reefs, tropical fish, sea turtles\n"
            response += "- High biodiversity but sensitive to temperature changes\n"
    
    if "salinity" in df.columns:
        avg_sal = df["salinity"].mean()
        response += f"\nSalinity: {round(avg_sal, 2)} PSU\n"
        if avg_sal < 34:
            response += "- Lower salinity may indicate freshwater influence\n"
            response += "- Affects species distribution and osmoregulation\n"
    
    if "pressure" in df.columns:
        avg_depth = df["pressure"].mean()
        if avg_depth < 200:
            response += "\n**Habitat Zone:** Sunlight-rich surface waters (photic zone)\n"
            response += "- Supports photosynthesis and primary production\n"
            response += "- Most diverse marine life\n"
        elif avg_depth < 1000:
            response += "\n**Habitat Zone:** Twilight zone (mesopelagic)\n"
            response += "- Limited light, specialized species\n"
            response += "- Important for carbon cycling\n"
        else:
            response += "\n**Habitat Zone:** Deep ocean (bathypelagic/abyssal)\n"
            response += "- Extreme pressure, no light\n"
            response += "- Unique adapted species\n"
    
    return response

def generate_climate_response(df):
    """Generate response for climate change queries"""
    response = "**Climate Change and Ocean Impact:**\n\n"
    
    if "temperature" in df.columns:
        avg_temp = df["temperature"].mean()
        temp_std = df["temperature"].std()
        
        response += f"Current ocean temperature: {round(avg_temp, 2)}°C\n"
        response += f"Temperature variability: {round(temp_std, 2)}°C\n\n"
        
        response += "**Key Climate Indicators:**\n"
        response += "- Ocean absorbs 90% of excess heat from global warming\n"
        response += "- Rising temperatures affect marine ecosystems\n"
        response += "- Changes in ocean circulation patterns\n"
        response += "- Increased stratification reduces nutrient mixing\n\n"
    
    if "salinity" in df.columns:
        response += "**Salinity Changes:**\n"
        response += "- Freshwater input from melting ice\n"
        response += "- Altered precipitation patterns\n"
        response += "- Impacts ocean density and circulation\n\n"
    
    response += "**Global Impacts:**\n"
    response += "- Sea level rise\n"
    response += "- Ocean acidification\n"
    response += "- Coral bleaching\n"
    response += "- Shifts in marine species distribution"
    
    return response

def generate_intelligent_response(prompt, df):
    """Generate context-aware response based on query intent"""
    intent = classify_query_intent(prompt)
    region_info = extract_region_from_prompt(prompt)
    
    # Filter by region if specified
    filtered_df = df
    if region_info and "latitude" in df.columns and "longitude" in df.columns:
        filtered_df = df[
            (df["latitude"] >= region_info["lat_range"][0]) &
            (df["latitude"] <= region_info["lat_range"][1]) &
            (df["longitude"] >= region_info["lon_range"][0]) &
            (df["longitude"] <= region_info["lon_range"][1])
        ]
    
    if intent == "pressure":
        return generate_pressure_response(filtered_df, region_info)
    elif intent == "glacier_ice":
        return generate_glacier_ice_response(filtered_df, prompt)
    elif intent == "marine_life":
        return generate_marine_life_response(filtered_df, region_info)
    elif intent == "climate":
        return generate_climate_response(filtered_df)
    elif intent == "salinity":
        if "salinity" in filtered_df.columns:
            avg_sal = filtered_df["salinity"].mean()
            region_name = region_info["name"] if region_info else "the region"
            return f"**Salinity Analysis for {region_name}:**\n\nAverage salinity: {round(avg_sal, 2)} PSU\nRange: {round(filtered_df['salinity'].min(), 2)} to {round(filtered_df['salinity'].max(), 2)} PSU\n\nSalinity affects ocean density, circulation, and marine life. Values between 34-36 PSU are typical for open ocean."
        return "Salinity data not available."
    elif intent == "currents":
        return "**Ocean Currents:**\n\nOcean currents are driven by wind, temperature, and salinity differences. ARGO data helps track water mass movement through temperature and salinity profiles. Major currents like the Gulf Stream transport heat globally, affecting climate patterns."
    
    return None
