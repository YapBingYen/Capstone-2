import requests

try:
    response = requests.get('http://localhost:5000/api/found-cats-map')
    if response.status_code == 200:
        data = response.json()
        print(f"✅ API returned {len(data)} cats")
        print(data)
    else:
        print(f"❌ API returned {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"❌ Error connecting to API: {e}")
