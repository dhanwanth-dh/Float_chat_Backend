def parse_prompt(prompt: str):
    prompt = prompt.lower()

    query = {
        "variable": "temperature",
        "min_depth": None,
        "max_depth": None,
        "region": None,
        "query_type": "general"
    }

    # Tsunami detection
    if any(word in prompt for word in ["tsunami", "flood", "disaster", "risk", "threat", "hazard"]):
        query["query_type"] = "tsunami"
        return query

    # Variable detection
    if "salinity" in prompt:
        query["variable"] = "salinity"

    # Depth detection
    if "deep" in prompt:
        query["min_depth"] = 1000
    elif "surface" in prompt:
        query["max_depth"] = 50

    # Region detection
    if "antarctica" in prompt or "southern ocean" in prompt:
        query["region"] = "southern"

    return query


def filter_data(df, query):
    if query["min_depth"] is not None:
        df = df[df["pressure"] >= query["min_depth"]]

    if query["max_depth"] is not None:
        df = df[df["pressure"] <= query["max_depth"]]

    if query["region"] == "southern":
        df = df[df["latitude"] < -40]

    return df