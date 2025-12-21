import pandas as pd
import numpy as np
import pickle
from openai import OpenAI
import os
from tqdm import tqdm
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def create_embedding(text, model="text-embedding-3-large"):
    """Create embedding for a single text string."""
    if pd.isna(text) or text == "":
        return None
    
    try:
        response = client.embeddings.create(
            input=str(text),
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return None

def create_embeddings_batch(texts, model="text-embedding-3-large", batch_size=100):
    """Create embeddings for a list of texts in batches."""
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
        batch = texts[i:i + batch_size]
        # Filter out empty/null texts
        valid_batch = [str(t) for t in batch if pd.notna(t) and str(t).strip() != ""]
        
        if not valid_batch:
            embeddings.extend([None] * len(batch))
            continue
        
        try:
            response = client.embeddings.create(
                input=valid_batch,
                model=model
            )
            batch_embeddings = [e.embedding for e in response.data]
            
            # Match embeddings back to original batch (accounting for filtered items)
            result_embeddings = []
            valid_idx = 0
            for t in batch:
                if pd.notna(t) and str(t).strip() != "":
                    result_embeddings.append(batch_embeddings[valid_idx])
                    valid_idx += 1
                else:
                    result_embeddings.append(None)
            
            embeddings.extend(result_embeddings)
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            embeddings.extend([None] * len(batch))
    
    return embeddings

def main():
    print("Loading filtered market data...")
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Load Kalshi markets
    try:
        kalshi_df = pd.read_csv("data/kalshi_filtered_markets.csv")
        print(f"Loaded {len(kalshi_df)} Kalshi markets")
    except FileNotFoundError:
        print("Error: data/kalshi_filtered_markets.csv not found")
        kalshi_df = None
    
    # Load Polymarket markets
    try:
        polymarket_df = pd.read_csv("data/polymarket_filtered_markets.csv")
        print(f"Loaded {len(polymarket_df)} Polymarket markets")
    except FileNotFoundError:
        print("Error: data/polymarket_filtered_markets.csv not found")
        polymarket_df = None
    
    embeddings_data = {}
    
    # Create Kalshi embeddings from "title" field
    if kalshi_df is not None and 'title' in kalshi_df.columns:
        print("\nCreating embeddings for Kalshi titles...")
        kalshi_texts = kalshi_df['title'].tolist()
        kalshi_embeddings = create_embeddings_batch(kalshi_texts)
        
        # Store with ticker as ID (or index if ticker not available)
        if 'ticker' in kalshi_df.columns:
            kalshi_ids = kalshi_df['ticker'].tolist()
        else:
            kalshi_ids = list(range(len(kalshi_df)))
        
        embeddings_data['kalshi'] = {
            'ids': kalshi_ids,
            'texts': kalshi_texts,
            'embeddings': kalshi_embeddings,
            'field': 'title'
        }
        print(f"Created {sum(1 for e in kalshi_embeddings if e is not None)} Kalshi embeddings")
    
    # Create Polymarket embeddings from "description" field
    if polymarket_df is not None and 'description' in polymarket_df.columns:
        print("\nCreating embeddings for Polymarket descriptions...")
        polymarket_texts = polymarket_df['description'].tolist()
        polymarket_embeddings = create_embeddings_batch(polymarket_texts)
        
        # Store with id as ID (or slug if id not available)
        if 'id' in polymarket_df.columns:
            polymarket_ids = polymarket_df['id'].tolist()
        elif 'slug' in polymarket_df.columns:
            polymarket_ids = polymarket_df['slug'].tolist()
        else:
            polymarket_ids = list(range(len(polymarket_df)))
        
        embeddings_data['polymarket'] = {
            'ids': polymarket_ids,
            'texts': polymarket_texts,
            'embeddings': polymarket_embeddings,
            'field': 'description'
        }
        print(f"Created {sum(1 for e in polymarket_embeddings if e is not None)} Polymarket embeddings")
    
    # Save embeddings to file
    output_file = "output/market_embeddings.pkl"
    print(f"\nSaving embeddings to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings_data, f)
    
    print(f"\nSuccess! Embeddings saved to {output_file}")
    print("\nData structure:")
    print("- embeddings_data['kalshi']['ids']: List of Kalshi tickers")
    print("- embeddings_data['kalshi']['embeddings']: List of embedding vectors")
    print("- embeddings_data['polymarket']['ids']: List of Polymarket IDs")
    print("- embeddings_data['polymarket']['embeddings']: List of embedding vectors")
    
    # Save a numpy version as well for easier access
    np_file = "output/market_embeddings.npz"
    print(f"\nAlso saving numpy format to {np_file}...")
    
    save_dict = {}
    if 'kalshi' in embeddings_data:
        kalshi_emb_array = np.array([e for e in embeddings_data['kalshi']['embeddings'] if e is not None])
        if len(kalshi_emb_array) > 0:
            save_dict['kalshi_embeddings'] = kalshi_emb_array
            save_dict['kalshi_ids'] = np.array([id for i, id in enumerate(embeddings_data['kalshi']['ids']) 
                                                 if embeddings_data['kalshi']['embeddings'][i] is not None])
    
    if 'polymarket' in embeddings_data:
        poly_emb_array = np.array([e for e in embeddings_data['polymarket']['embeddings'] if e is not None])
        if len(poly_emb_array) > 0:
            save_dict['polymarket_embeddings'] = poly_emb_array
            save_dict['polymarket_ids'] = np.array([id for i, id in enumerate(embeddings_data['polymarket']['ids']) 
                                                     if embeddings_data['polymarket']['embeddings'][i] is not None])
    
    np.savez(np_file, **save_dict)
    print("Done!")

if __name__ == "__main__":
    main()
