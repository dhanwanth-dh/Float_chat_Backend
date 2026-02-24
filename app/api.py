from fastapi import APIRouter
from app.models import ChatRequest
from services.data_loader import load_data
from services.query_engine import parse_prompt, filter_data
from services.visualizer import temperature_depth_plot, generate_heatmap, generate_probability_distribution
from services.ai_engine import summarize, train_model, load_model, analyze_anomalies, get_location_insights, calculate_probabilities
from services.conversation import conversation_manager
from services.tsunami_predictor import generate_tsunami_analysis
from services.intelligent_responder import generate_intelligent_response, classify_query_intent
from services.external_ai import is_oceanographic_query, get_fallback_response

router = APIRouter()

df = load_data()


if not load_model() and not df.empty:
    train_result = train_model(df)
    if train_result:
        print(f"✅ AI Model trained: R² = {train_result['r2_score']}, Samples = {train_result['samples']}")

@router.post("/chat")
def chat(request: ChatRequest):
    session_id = request.session_id
    conversation_manager.add_message(session_id, "user", request.prompt)
    
    # Check if user wants visualizations
    show_visualizations = any(word in request.prompt.lower() for word in ["graph", "chart", "plot", "heatmap", "map", "visualize", "show"])
    
    if not is_oceanographic_query(request.prompt):
        fallback_response = get_fallback_response(request.prompt)
        conversation_manager.add_message(session_id, "assistant", fallback_response)
        return {
            "summary": fallback_response,
            "query_type": "external",
            "conversation_history": conversation_manager.get_history(session_id)
        }
    
    intent = classify_query_intent(request.prompt)
    
    if intent == "tsunami":
        tsunami_analysis = generate_tsunami_analysis(df, request.prompt)
        conversation_manager.add_message(session_id, "assistant", tsunami_analysis["summary"])
        
        return {
            "summary": tsunami_analysis["summary"],
            "tsunami_risks": tsunami_analysis["top_risks"],
            "all_regions": tsunami_analysis["all_regions"],
            "recommendations": tsunami_analysis["recommendations"],
            "query_type": "tsunami",
            "conversation_history": conversation_manager.get_history(session_id)
        }
    
    intelligent_response = generate_intelligent_response(request.prompt, df)
    if intelligent_response:
        conversation_manager.add_message(session_id, "assistant", intelligent_response)
        return {
            "summary": intelligent_response,
            "query_type": "intelligent",
            "conversation_history": conversation_manager.get_history(session_id)
        }
    
    query = parse_prompt(request.prompt)
    filtered_df = filter_data(df, query)

    if filtered_df.empty:
        response_text = "No ARGO data available for this query. Try asking about temperature, salinity, pressure, marine life, glaciers, or climate change."
        conversation_manager.add_message(session_id, "assistant", response_text)
        return {
            "summary": response_text,
            "chart": None,
            "heatmap": None,
            "probabilities": {},
            "issues": [],
            "location_insights": "No data available.",
            "query_type": "general",
            "show_visualizations": False,
            "conversation_history": conversation_manager.get_history(session_id)
        }

    variable = query["variable"]

    stats = {
        "variable": variable,
        "mean_value": round(filtered_df[variable].mean(), 2),
        "min_value": round(filtered_df[variable].min(), 2),
        "max_value": round(filtered_df[variable].max(), 2),
        "depth_range": (
            int(filtered_df["pressure"].min()),
            int(filtered_df["pressure"].max())
        ),
        "data_points": int(len(filtered_df)),
        "total_records": len(df)
    }

    ai_summary = summarize(request.prompt, stats)
    
    # Generate visualizations only if requested
    chart_json = temperature_depth_plot(filtered_df) if show_visualizations else None
    heatmap_json = generate_heatmap(filtered_df, variable) if show_visualizations else None
    prob_dist_json = generate_probability_distribution(filtered_df, variable) if show_visualizations else None
    
    anomalies = analyze_anomalies(filtered_df, variable)
    location_insights = get_location_insights(filtered_df, query)
    probabilities = calculate_probabilities(filtered_df, variable)

    conversation_manager.add_message(session_id, "assistant", ai_summary, metadata=stats)

    return {
        "summary": ai_summary,
        "stats": stats,
        "chart": chart_json,
        "heatmap": heatmap_json,
        "probability_distribution": prob_dist_json,
        "probabilities": probabilities,
        "issues": anomalies,
        "location_insights": location_insights,
        "query_type": "general",
        "show_visualizations": show_visualizations,
        "conversation_history": conversation_manager.get_history(session_id)
    }

@router.post("/train")
def train():
    """Train AI model on ARGO dataset"""
    if df.empty:
        return {"status": "error", "message": "No data available for training"}
    
    result = train_model(df)
    if result:
        return {"status": "success", "metrics": result}
    return {"status": "error", "message": "Training failed"}