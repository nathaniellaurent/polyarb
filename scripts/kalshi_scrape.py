import requests
import pandas as pd
import time

def fetch_kalshi_markets():
    # Note: the "elections" subdomain hosts ALL markets (Economics, Weather, etc.)
    base_url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    all_markets = []
    cursor = ""
    limit = 1000  # Number of markets per request
    max_markets = 300000  # Adjust this threshold as needed
    
    print("Fetching active markets from Kalshi...")

    while True:
        # Status 'open' filters for active, tradable markets
        params = {
            "limit": limit,
            "status": "open",
            "cursor": cursor
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            markets = data.get("markets", [])
            if not markets:
                break
                
            all_markets.extend(markets)
            print(f"Retrieved {len(all_markets)} markets...")
            
            # Kalshi uses a 'cursor' for the next page
            cursor = data.get("cursor")
            if not cursor or len(all_markets) >= max_markets:
                break
                
            time.sleep(0.01)  # Minor delay to be respectful
            
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            break

    return all_markets

# Run the fetcher
markets_data = fetch_kalshi_markets()

print(f"\nTotal markets fetched: {len(markets_data)}")

# Filter out parlays (MVE - Multi-Variable Events)
# Parlays are implemented internally as MVE with mve_collection_ticker set
markets_data = [m for m in markets_data if m.get("mve_collection_ticker") is None]
print(f"Filtered to {len(markets_data)} markets (removed parlays)")

# Process into a DataFrame
df = pd.DataFrame(markets_data)

# Use all available columns from the API response
print(f"Available columns: {list(df.columns)}")
df_final = df.copy()

df_final.to_csv("data/kalshi_active_markets.csv", index=False)
print(f"\nSaved {len(df_final)} markets to data/kalshi_active_markets.csv")

# Create filtered CSV with specific columns
cols_to_filter = ['expected_expiration_time', 'ticker', 'title', 'subtitle', 'yes_sub_title', 
                   'no_sub_title', 'category', 'yes_bid_dollars', 'yes_ask_dollars', 
                   'no_bid_dollars', 'no_ask_dollars', 'market_type', 'rules_primary', 
                   'rules_secondary', 'primary_participant_key']

available_filter_cols = [col for col in cols_to_filter if col in df.columns]
missing_filter_cols = [col for col in cols_to_filter if col not in df.columns]

if missing_filter_cols:
    print(f"\nWarning: Missing columns for filtered CSV: {missing_filter_cols}")

df_filtered = df[available_filter_cols].copy()
df_filtered.to_csv("data/kalshi_filtered_markets.csv", index=False)
print(f"Saved {len(df_filtered)} markets to data/kalshi_filtered_markets.csv with {len(available_filter_cols)} columns")