import os
import requests
from pathlib import Path

# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

def is_oceanographic_query(prompt):
    """Check if query is related to oceanography"""
    ocean_keywords = [
        "ocean", "sea", "water", "marine", "temperature", "salinity", "pressure",
        "depth", "tsunami", "wave", "current", "tide", "fish", "whale", "coral",
        "ice", "glacier", "arctic", "antarctic", "climate", "argo", "pacific",
        "atlantic", "indian ocean", "southern ocean", "coastal", "beach", "shore"
    ]
    
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in ocean_keywords)

def query_gemini(prompt):
    """Query Google Gemini API for general questions"""
    if not GEMINI_API_KEY:
        return "External AI service not configured. Please set GEMINI_API_KEY environment variable."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    
    headers = {"Content-Type": "application/json"}
    
    data = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        if "candidates" in result and len(result["candidates"]) > 0:
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            return text
        
        return "Unable to get response from AI service."
    
    except requests.exceptions.RequestException as e:
        return f"Error connecting to AI service: {str(e)}"
    except Exception as e:
        return f"Error processing AI response: {str(e)}"

def get_fallback_response(prompt):
    """Get response for non-oceanographic queries"""
    
    # Try Gemini first
    if GEMINI_API_KEY:
        return query_gemini(prompt)
    
    # Fallback message if no API key
    return (
        "This question is outside my oceanography expertise. "
        "I specialize in ocean data analysis, marine conditions, tsunami prediction, "
        "and climate impacts on oceans. Please ask questions related to:\n\n"
        "- Ocean temperature, salinity, and pressure\n"
        "- Marine life and ecosystems\n"
        "- Tsunami and flood risk analysis\n"
        "- Glacier and ice melting\n"
        "- Ocean currents and circulation\n"
        "- Climate change impacts on oceans\n\n"
        "For general questions, please configure GEMINI_API_KEY environment variable."
    )
