"""
Embedding Analysis and Visualization Tool
==========================================
This script provides utilities to analyze and visualize the learned embeddings.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from pathlib import Path
from tensorflow import keras
from cat_identity_model_trainer import preprocess_image
from typing import List, Tuple


def load_embeddings(embeddings_path='cat_embeddings.npz', 
                    metadata_path='cat_metadata.csv') -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load saved embeddings and metadata.
    
    Returns:
        embeddings: NumPy array of embeddings
        metadata: DataFrame with image paths and labels
    """
    if not os.path.exists(embeddings_path):
        print(f"❌ Error: Embeddings not found at {embeddings_path}")
        print("Please train the model first or extract embeddings.")
        return None, None
    
    print(f"Loading embeddings from {embeddings_path}...")
    data = np.load(embeddings_path)
    embeddings = data['embeddings']
    image_paths = data['image_paths']
    
    print(f"✅ Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    
    # Load metadata if available
    metadata = None
    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path)
        print(f"✅ Loaded metadata for {len(metadata)} images")
    
    return embeddings, metadata


def visualize_embeddings_tsne(embeddings: np.ndarray, metadata: pd.DataFrame, 
                               save_path='embeddings_tsne.png'):
    """
    Visualize embeddings using t-SNE dimensionality reduction.
    
    Args:
        embeddings: Array of embeddings (N, embedding_dim)
        metadata: DataFrame with breed information
        save_path: Path to save the visualization
    """
    print("\nComputing t-SNE projection (this may take a few minutes)...")
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Get unique breeds and assign colors
    breeds = metadata['breed_name'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(breeds)))
    
    # Plot each breed with different color
    for i, breed in enumerate(breeds):
        mask = metadata['breed_name'] == breed
        plt.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1],
            c=[colors[i]], 
            label=breed, 
            alpha=0.6,
            s=50
        )
    
    plt.title('Cat Embeddings Visualization (t-SNE)', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ t-SNE visualization saved to {save_path}")
    plt.show()


def visualize_embeddings_pca(embeddings: np.ndarray, metadata: pd.DataFrame,
                             save_path='embeddings_pca.png'):
    """
    Visualize embeddings using PCA dimensionality reduction.
    
    Args:
        embeddings: Array of embeddings (N, embedding_dim)
        metadata: DataFrame with breed information
        save_path: Path to save the visualization
    """
    print("\nComputing PCA projection...")
    
    # Compute PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Get unique breeds and assign colors
    breeds = metadata['breed_name'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(breeds)))
    
    # Plot each breed with different color
    for i, breed in enumerate(breeds):
        mask = metadata['breed_name'] == breed
        plt.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1],
            c=[colors[i]], 
            label=breed, 
            alpha=0.6,
            s=50
        )
    
    plt.title('Cat Embeddings Visualization (PCA)', fontsize=16, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ PCA visualization saved to {save_path}")
    plt.show()


def compute_similarity_statistics(embeddings: np.ndarray, metadata: pd.DataFrame):
    """
    Compute and display similarity statistics.
    
    Args:
        embeddings: Array of embeddings
        metadata: DataFrame with breed information
    """
    print("\n" + "="*80)
    print("SIMILARITY STATISTICS")
    print("="*80)
    
    # Compute pairwise similarities within same breed
    same_breed_similarities = []
    different_breed_similarities = []
    
    breeds = metadata['breed_name'].unique()
    
    print("\nComputing pairwise similarities...")
    
    for breed in breeds:
        breed_mask = metadata['breed_name'] == breed
        breed_embeddings = embeddings[breed_mask]
        
        # Same breed similarities
        if len(breed_embeddings) > 1:
            for i in range(len(breed_embeddings)):
                for j in range(i+1, len(breed_embeddings)):
                    sim = np.dot(breed_embeddings[i], breed_embeddings[j])
                    same_breed_similarities.append(sim)
    
    # Different breed similarities (sample to avoid too many comparisons)
    n_samples = 10000
    for _ in range(n_samples):
        breed1, breed2 = np.random.choice(breeds, size=2, replace=False)
        
        emb1_idx = np.random.choice(np.where(metadata['breed_name'] == breed1)[0])
        emb2_idx = np.random.choice(np.where(metadata['breed_name'] == breed2)[0])
        
        sim = np.dot(embeddings[emb1_idx], embeddings[emb2_idx])
        different_breed_similarities.append(sim)
    
    same_breed_similarities = np.array(same_breed_similarities)
    different_breed_similarities = np.array(different_breed_similarities)
    
    # Print statistics
    print("\n📊 Same Breed (Positive Pairs):")
    print(f"   Mean: {np.mean(same_breed_similarities):.4f}")
    print(f"   Std:  {np.std(same_breed_similarities):.4f}")
    print(f"   Min:  {np.min(same_breed_similarities):.4f}")
    print(f"   Max:  {np.max(same_breed_similarities):.4f}")
    
    print("\n📊 Different Breeds (Negative Pairs):")
    print(f"   Mean: {np.mean(different_breed_similarities):.4f}")
    print(f"   Std:  {np.std(different_breed_similarities):.4f}")
    print(f"   Min:  {np.min(different_breed_similarities):.4f}")
    print(f"   Max:  {np.max(different_breed_similarities):.4f}")
    
    print("\n📊 Separation:")
    separation = np.mean(same_breed_similarities) - np.mean(different_breed_similarities)
    print(f"   Mean Difference: {separation:.4f}")
    
    # Plot distributions
    plt.figure(figsize=(12, 6))
    
    plt.hist(same_breed_similarities, bins=50, alpha=0.6, label='Same Breed', color='green')
    plt.hist(different_breed_similarities, bins=50, alpha=0.6, label='Different Breeds', color='red')
    
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Embedding Similarities', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('similarity_distribution.png', dpi=150, bbox_inches='tight')
    print("\n✅ Similarity distribution plot saved to similarity_distribution.png")
    plt.show()
    
    # Suggest optimal threshold
    # Find threshold that maximizes separation
    thresholds = np.linspace(0.5, 1.0, 100)
    accuracies = []
    
    for thresh in thresholds:
        tp = np.sum(same_breed_similarities >= thresh)
        tn = np.sum(different_breed_similarities < thresh)
        accuracy = (tp + tn) / (len(same_breed_similarities) + len(different_breed_similarities))
        accuracies.append(accuracy)
    
    optimal_threshold = thresholds[np.argmax(accuracies)]
    optimal_accuracy = np.max(accuracies)
    
    print(f"\n💡 Recommended Threshold: {optimal_threshold:.3f}")
    print(f"   Expected Accuracy: {optimal_accuracy:.2%}")


def find_nearest_neighbors(query_embedding: np.ndarray, embeddings: np.ndarray,
                          metadata: pd.DataFrame, top_k: int = 5) -> List[Tuple]:
    """
    Find the k nearest neighbors to a query embedding.
    
    Args:
        query_embedding: Query embedding vector
        embeddings: All embeddings
        metadata: Metadata DataFrame
        top_k: Number of neighbors to return
    
    Returns:
        List of (image_path, breed, similarity) tuples
    """
    # Compute similarities
    similarities = np.dot(embeddings, query_embedding)
    
    # Get top k indices
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Get results
    results = []
    for idx in top_k_indices:
        results.append((
            metadata.iloc[idx]['image_path'],
            metadata.iloc[idx]['breed_name'],
            similarities[idx]
        ))
    
    return results


def analyze_model_performance(embeddings: np.ndarray, metadata: pd.DataFrame):
    """
    Analyze model performance on breed classification.
    
    Args:
        embeddings: Array of embeddings
        metadata: DataFrame with breed information
    """
    print("\n" + "="*80)
    print("BREED CLASSIFICATION ANALYSIS (k-NN)")
    print("="*80)
    
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, 
        metadata['breed_label'].values,
        test_size=0.2,
        random_state=42,
        stratify=metadata['breed_label'].values
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train k-NN classifier
    print("\nTraining k-NN classifier (k=5)...")
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(X_train, y_train)
    
    # Predict
    y_pred = knn.predict(X_test)
    
    # Print results
    accuracy = np.mean(y_pred == y_test)
    print(f"\n✅ Test Accuracy: {accuracy:.2%}")
    
    # Per-breed accuracy
    print("\n📊 Per-Breed Performance:")
    breeds = metadata['breed_name'].unique()
    breed_to_label = {breed: metadata[metadata['breed_name'] == breed]['breed_label'].iloc[0] 
                      for breed in breeds}
    label_to_breed = {v: k for k, v in breed_to_label.items()}
    
    for breed in breeds:
        breed_label = breed_to_label[breed]
        breed_mask = y_test == breed_label
        if np.sum(breed_mask) > 0:
            breed_accuracy = np.mean(y_pred[breed_mask] == y_test[breed_mask])
            print(f"   {breed:20s}: {breed_accuracy:6.2%} ({np.sum(breed_mask):3d} samples)")


def interactive_search(model_path='cat_identity_model.h5',
                       embeddings_path='cat_embeddings.npz',
                       metadata_path='cat_metadata.csv'):
    """
    Interactive search: find similar cats to a query image.
    
    Args:
        model_path: Path to trained model
        embeddings_path: Path to saved embeddings
        metadata_path: Path to metadata CSV
    """
    print("\n" + "="*80)
    print("INTERACTIVE SIMILARITY SEARCH")
    print("="*80)
    
    # Load model and embeddings
    print("\nLoading model...")
    model = keras.models.load_model(model_path, compile=False)
    
    embeddings, metadata = load_embeddings(embeddings_path, metadata_path)
    
    if embeddings is None or metadata is None:
        return
    
    while True:
        print("\n" + "-"*80)
        query_path = input("Enter path to query image (or 'quit' to exit): ").strip()
        
        if query_path.lower() == 'quit':
            print("Goodbye! 👋")
            break
        
        if not os.path.exists(query_path):
            print(f"❌ Error: Image not found at {query_path}")
            continue
        
        # Get query embedding
        query_img = preprocess_image(query_path)
        query_embedding = model.predict(np.expand_dims(query_img, 0), verbose=0)[0]
        
        # Find nearest neighbors
        top_k = int(input("How many similar images to find? (default 10): ") or "10")
        neighbors = find_nearest_neighbors(query_embedding, embeddings, metadata, top_k)
        
        # Display results
        print("\n" + "="*80)
        print(f"TOP {top_k} MOST SIMILAR CATS")
        print("="*80)
        
        for i, (img_path, breed, similarity) in enumerate(neighbors, 1):
            print(f"{i:2d}. Similarity: {similarity:.4f} | Breed: {breed:20s} | Path: {img_path}")
        
        print("="*80)


def main():
    """Main function"""
    print("="*80)
    print("CAT EMBEDDINGS ANALYSIS TOOL")
    print("="*80)
    
    # Load embeddings
    embeddings, metadata = load_embeddings()
    
    if embeddings is None or metadata is None:
        print("\n❌ Cannot proceed without embeddings. Please train the model first.")
        return
    
    # Menu
    while True:
        print("\n" + "="*80)
        print("ANALYSIS OPTIONS")
        print("="*80)
        print("1. Visualize embeddings (t-SNE)")
        print("2. Visualize embeddings (PCA)")
        print("3. Compute similarity statistics")
        print("4. Analyze breed classification performance")
        print("5. Interactive similarity search")
        print("6. Run all analyses")
        print("7. Exit")
        print("="*80)
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            visualize_embeddings_tsne(embeddings, metadata)
        
        elif choice == '2':
            visualize_embeddings_pca(embeddings, metadata)
        
        elif choice == '3':
            compute_similarity_statistics(embeddings, metadata)
        
        elif choice == '4':
            analyze_model_performance(embeddings, metadata)
        
        elif choice == '5':
            interactive_search()
        
        elif choice == '6':
            print("\n🚀 Running all analyses...")
            visualize_embeddings_pca(embeddings, metadata)
            visualize_embeddings_tsne(embeddings, metadata)
            compute_similarity_statistics(embeddings, metadata)
            analyze_model_performance(embeddings, metadata)
        
        elif choice == '7':
            print("Goodbye! 👋")
            break
        
        else:
            print("❌ Invalid choice, please enter 1-7")


if __name__ == "__main__":
    main()

