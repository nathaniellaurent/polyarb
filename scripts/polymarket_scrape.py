import requests
import pandas as pd
import time

def fetch_all_active_markets():
    base_url = "https://gamma-api.polymarket.com/markets"
    all_markets = []
    limit = 500  # Number of markets per request
    offset = 0
    max_markets = 100000  # Adjust this threshold as needed
    
    print("Fetching active markets from Polymarket...")

    while True:
        # Parameters: active=true ensures we only get open markets
        # closed=false and archived=false filter out finished ones
        params = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset,
            "order": "volume24hr", # Optional: sort by activity
            "ascending": "false"
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            all_markets.extend(data)
            print(f"Retrieved {len(all_markets)} markets...")
            
            # If we received fewer items than the limit, we've reached the end
            if len(data) < limit or len(all_markets) >= max_markets:  # Adjust threshold as needed
                break
                
            offset += limit
            time.sleep(0.05)  # Less Polite rate limiting
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break

    return all_markets

# Execute the scraper
markets_list = fetch_all_active_markets()

# Create data directory if it doesn't exist
import os
os.makedirs("data", exist_ok=True)

# Convert to DataFrame for easy analysis/viewing
df = pd.DataFrame(markets_list)

# Use all available columns from the API response
print(f"Available columns: {list(df.columns)}")

# Remove newlines and carriage returns from string columns
df_clean = df.copy()
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        df_clean[col] = df_clean[col].apply(lambda x: x.replace('\n', ' ').replace('\r', ' ') if isinstance(x, str) else x)

# Save to CSV
df_clean.to_csv("data/polymarket_open_markets.csv", index=False, lineterminator='\n')
print(f"\nSuccess! Saved {len(df_clean)} active markets to data/polymarket_open_markets.csv")

# Also create a filtered CSV with specific columns of interest
pm_cols = [
    'slug', 'id', 'question', 'description', 'outcomes', 'endDate',
    'endDateIso', 'bestBid', 'bestAsk', 'enableOrderBook',
    'sportsMarketType', 'resolutionSource', 'resolvedBy'
]

available_pm_cols = [c for c in pm_cols if c in df_clean.columns]
missing_pm_cols = [c for c in pm_cols if c not in df_clean.columns]
if missing_pm_cols:
    print(f"Warning: Missing columns for data/polymarket_filtered_markets.csv: {missing_pm_cols}")

df_filtered = df_clean[available_pm_cols].copy()
df_filtered.to_csv("data/polymarket_filtered_markets.csv", index=False, lineterminator='\n')
print(f"Saved {len(df_filtered)} markets to data/polymarket_filtered_markets.csv with {len(available_pm_cols)} columns")

# Quick preview
if 'question' in df_clean.columns:
    print(df_clean[['question']].head())
else:
    print(df_clean.head())