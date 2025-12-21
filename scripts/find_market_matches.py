import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os

def load_embeddings(embeddings_file="output/market_embeddings.pkl"):
    """Load embeddings from pickle file."""
    print(f"Loading embeddings from {embeddings_file}...")
    with open(embeddings_file, 'rb') as f:
        embeddings_data = pickle.load(f)
    return embeddings_data

def compute_similarity_matrix(kalshi_embeddings, polymarket_embeddings):
    """Compute cosine similarity between all Kalshi and Polymarket markets."""
    print("\nComputing cosine similarity matrix...")
    print(f"Kalshi markets: {len(kalshi_embeddings)}")
    print(f"Polymarket markets: {len(polymarket_embeddings)}")
    
    # Convert to numpy arrays if not already
    kalshi_array = np.array(kalshi_embeddings)
    polymarket_array = np.array(polymarket_embeddings)
    
    # Compute cosine similarity matrix (rows=Kalshi, cols=Polymarket)
    similarity_matrix = cosine_similarity(kalshi_array, polymarket_array)
    
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    return similarity_matrix

def find_best_matches(embeddings_data, similarity_threshold=0.75):
    """Find the single best Polymarket match for each Kalshi market above the threshold."""
    
    # Extract embeddings and IDs
    kalshi_embeddings = [e for e in embeddings_data['kalshi']['embeddings'] if e is not None]
    kalshi_ids = [id for i, id in enumerate(embeddings_data['kalshi']['ids']) 
                  if embeddings_data['kalshi']['embeddings'][i] is not None]
    kalshi_texts = [text for i, text in enumerate(embeddings_data['kalshi']['texts']) 
                    if embeddings_data['kalshi']['embeddings'][i] is not None]
    
    polymarket_embeddings = [e for e in embeddings_data['polymarket']['embeddings'] if e is not None]
    polymarket_ids = [id for i, id in enumerate(embeddings_data['polymarket']['ids']) 
                      if embeddings_data['polymarket']['embeddings'][i] is not None]
    polymarket_texts = [text for i, text in enumerate(embeddings_data['polymarket']['texts']) 
                        if embeddings_data['polymarket']['embeddings'][i] is not None]
    
    print(f"\nValid embeddings: {len(kalshi_embeddings)} Kalshi, {len(polymarket_embeddings)} Polymarket")
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(kalshi_embeddings, polymarket_embeddings)
    
    # Find the single best match per Kalshi market, if above threshold
    matches = []
    print(f"\nFinding best match above similarity threshold: {similarity_threshold}")

    for i in tqdm(range(len(kalshi_ids)), desc="Processing Kalshi markets"):
        row = similarity_matrix[i]
        best_j = int(np.argmax(row))
        best_similarity = row[best_j]

        if best_similarity >= similarity_threshold:
            matches.append({
                'kalshi_ticker': kalshi_ids[i],
                'kalshi_title': kalshi_texts[i],
                'polymarket_id': polymarket_ids[best_j],
                'polymarket_question': polymarket_texts[best_j],
                'cosine_similarity': best_similarity
            })
    
    return matches, similarity_matrix

def save_matches(matches, output_dir="output"):
    """Save best matches to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save high-similarity matches
    matches_df = pd.DataFrame(matches)
    matches_file = f"{output_dir}/market_matches.csv"
    matches_df.to_csv(matches_file, index=False)
    print(f"\nSaved {len(matches)} high-similarity matches to {matches_file}")
    
    return matches_df

def main():
    # Load embeddings
    embeddings_data = load_embeddings()
    
    # Find matches
    # Adjust similarity_threshold (0.75-0.90 typical range)
    # Adjust top_k to see more/fewer top matches per market
    matches, similarity_matrix = find_best_matches(
        embeddings_data,
        similarity_threshold=0.9  # Only pairs above 90% similarity
    )
    
    print(f"\nFound {len(matches)} market pairs with similarity >= 0.9")
    
    # Save results
    matches_df = save_matches(matches)
    
    # Display some statistics
    if len(matches) > 0:
        print("\nSimilarity statistics for high-confidence matches:")
        print(f"  Mean: {matches_df['cosine_similarity'].mean():.4f}")
        print(f"  Median: {matches_df['cosine_similarity'].median():.4f}")
        print(f"  Max: {matches_df['cosine_similarity'].max():.4f}")
        print(f"  Min: {matches_df['cosine_similarity'].min():.4f}")
        
        print("\nTop 5 matches:")
        print(matches_df.nlargest(5, 'cosine_similarity')[
            ['kalshi_ticker', 'kalshi_title', 'polymarket_question', 'cosine_similarity']
        ].to_string(index=False))
    
    # Save similarity matrix for later analysis
    similarity_matrix_file = "output/similarity_matrix.npz"
    np.savez_compressed(
        similarity_matrix_file,
        similarity_matrix=similarity_matrix,
        kalshi_ids=[id for i, id in enumerate(embeddings_data['kalshi']['ids']) 
                    if embeddings_data['kalshi']['embeddings'][i] is not None],
        polymarket_ids=[id for i, id in enumerate(embeddings_data['polymarket']['ids']) 
                        if embeddings_data['polymarket']['embeddings'][i] is not None]
    )
    print(f"\nSaved similarity matrix to {similarity_matrix_file}")
    
    print("\nDone! Check output/ directory for results.")

if __name__ == "__main__":
    main()
