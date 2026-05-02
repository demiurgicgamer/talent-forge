import requests
import os
from dotenv import load_dotenv

load_dotenv()

APP_ID = os.getenv("ADZUNA_APP_ID")
APP_KEY = os.getenv("ADZUNA_APP_KEY")


def fetch_jobs(query="XR Developer", location="Canada"):
    url = "https://api.adzuna.com/v1/api/jobs/ca/search/1"

    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": 10,
        "what": query,
        "where": location,
        "content-type": "application/json"
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        # -------------------
        # CHECK STATUS
        # -------------------
        if response.status_code != 200:
            print(f"API error: {response.status_code}")
            raise Exception("API failed")

        data = response.json()

        jobs = []

        for item in data.get("results", []):
            jobs.append({
                "title": item.get("title", "No Title"),
                "description": item.get("description", "")[:500],
                "apply_url": item.get("redirect_url", "#")
            })

        return jobs

    except Exception as e:
        print("API failed, switching to fallback mode:", e)

        # fallback ONLY for safety
        return [
            {
                "title": "XR Developer",
                "description": "Unity, AR/VR, C#, AI integration",
                "apply_url": "#"
            },
            {
                "title": "AI Engineer",
                "description": "LLMs, Python, APIs",
                "apply_url": "#"
            }
        ]