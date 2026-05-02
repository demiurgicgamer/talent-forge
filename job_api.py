import requests

def fetch_jobs(query="XR Developer", location="Canada"):
    """
    Simple job fetcher (replace API later if needed)
    """

    # Example API structure (placeholder-friendly)
    url = f"https://api.adzuna.com/v1/api/jobs/ca/search/1"

    # NOTE: You can replace this with real API keys later
    params = {
        "app_id": "demo",
        "app_key": "demo",
        "results_per_page": 10,
        "what": query,
        "where": location,
        "content-type": "application/json"
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        jobs = []

        for item in data.get("results", []):
            jobs.append({
                "title": item.get("title", ""),
                "description": item.get("description", "")[:500]
            })

        return jobs

    except Exception as e:
        print("API failed, using fallback mock jobs:", e)

        return [
            {"title": "XR Developer", "description": "Unity, AR/VR, C#, AI integration"},
            {"title": "AI Engineer", "description": "LLMs, Python, APIs"},
        ]