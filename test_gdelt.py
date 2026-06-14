import requests

query = '"HDFC Bank Limited" OR "HDFCBANK.NS" OR "HDFCBANK"'
params = {
    "query": query,
    "mode": "ArtList",
    "maxrecords": 250,
    "startdatetime": "20230101000000",
    "enddatetime": "20231231235959",
    "format": "json",
    "sourcelang": "english",
    "sourcecountry": "IN",
}

resp = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, headers={"User-Agent": "Mozilla/5.0"})
try:
    data = resp.json()
    articles = data.get('articles', [])
    print(f"Yearly query returned {len(articles)} articles.")
except Exception as e:
    print(f"Error: {e}, Response: {resp.text}")
