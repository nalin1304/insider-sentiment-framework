import requests
import json

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Referer": "https://www.bseindia.com/",
    "Accept": "application/json, text/plain, */*"
}

params = {
    "scripcode": "500325",
    "fromdate": "01/01/2020",
    "todate": "01/01/2025"
}

resp = requests.get("https://api.bseindia.com/BseIndiaAPI/api/InsiderTrading/w", headers=headers, params=params)
try:
    print(len(resp.json()))
except:
    print(resp.text[:500])

