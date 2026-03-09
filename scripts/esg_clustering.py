#!/usr/bin/env python3
"""
ESG Report Clustering Pipeline
Using Hugging Face transformers for summarization and embeddings
"""

import os
import glob
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')
import torch

# Step 1: Load all text files from the data folder
def load_text_files(data_folder="data"):
    """Load all .txt files from the data folder"""
    texts = []
    filenames = []
    
    # Get all txt files
    txt_files = glob.glob(os.path.join(data_folder, "*.txt"))
    
    print(f"Found {len(txt_files)} text files")
    
    if len(txt_files) == 0:
        print(f"No .txt files found in {data_folder} folder!")
        print(f"Current directory: {os.getcwd()}")
        return filenames, texts
    
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                texts.append(content)
                filenames.append(os.path.basename(file_path))
                print(f"✅ Loaded: {os.path.basename(file_path)} ({len(content)} characters)")
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    
    return filenames, texts

# Step 2: Summarize documents using Hugging Face pipeline
def summarize_documents(texts, max_length=150, min_length=50):
    """Generate summaries using Hugging Face transformers"""
    print("\n📝 Initializing Hugging Face summarization pipeline...")
    print("   Model: facebook/bart-large-cnn (from Hugging Face)")
    print("   https://huggingface.co/facebook/bart-large-cnn")
    
    # Check if GPU is available
    device = 0 if torch.cuda.is_available() else -1
    print(f"   Using device: {'GPU' if device == 0 else 'CPU'}")
    
    summaries = []
    
    try:
        # Use the pipeline with the correct task name
        print("   Loading summarization pipeline (this may take a minute)...")
        summarizer = pipeline("summarization", 
                             model="facebook/bart-large-cnn",
                             device=device)
        
        for i, text in enumerate(texts):
            print(f"   Summarizing document {i+1}/{len(texts)}...")
            
            try:
                # Truncate long documents (BART has 1024 token limit)
                if len(text) > 3000:
                    text = text[:3000]
                    print(f"     (Document truncated to 3000 chars)")
                
                # Generate summary
                summary_result = summarizer(text, 
                                          max_length=max_length, 
                                          min_length=min_length,
                                          do_sample=False)
                
                summary = summary_result[0]['summary_text']
                summaries.append(summary)
                print(f"     ✅ Summary generated ({len(summary)} chars)")
                
            except Exception as e:
                print(f"     ❌ Error on document {i+1}: {e}")
                # Fallback: take first 500 chars
                summaries.append(text[:500] + "...")
    
    except Exception as e:
        print(f"❌ Error with pipeline: {e}")
        print("   Trying alternative method with AutoTokenizer and AutoModel...")
        
        try:
            # Alternative method: load tokenizer and model directly
            tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
            model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
            
            if device == 0:
                model = model.cuda()
            
            for i, text in enumerate(texts):
                print(f"   Summarizing document {i+1}/{len(texts)}...")
                
                try:
                    # Truncate long documents
                    if len(text) > 3000:
                        text = text[:3000]
                    
                    # Tokenize and generate
                    inputs = tokenizer.encode("summarize: " + text, 
                                            return_tensors="pt", 
                                            max_length=1024, 
                                            truncation=True)
                    
                    if device == 0:
                        inputs = inputs.cuda()
                    
                    summary_ids = model.generate(inputs, 
                                               max_length=max_length, 
                                               min_length=min_length,
                                               num_beams=4,
                                               early_stopping=True)
                    
                    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    summaries.append(summary)
                    print(f"     ✅ Summary generated ({len(summary)} chars)")
                    
                except Exception as e:
                    print(f"     ❌ Error: {e}")
                    summaries.append(text[:500] + "...")
        
        except Exception as e:
            print(f"❌ All summarization methods failed: {e}")
            print("   Using simple extractive summaries as final fallback...")
            for text in texts:
                summaries.append(text[:500] + "...")
    
    return summaries

# Step 3: Generate embeddings using Sentence Transformers (Hugging Face)
def generate_embeddings(texts, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    """Generate embeddings using sentence transformers from Hugging Face"""
    print(f"\n🔤 Loading embedding model from Hugging Face: {model_name}")
    print(f"   Model URL: https://huggingface.co/sentence-transformers/{model_name}")
    print("   (This downloads the model first time - may take a few minutes)")
    
    try:
        model = SentenceTransformer(model_name)
        print("   ✅ Model loaded successfully")
        
        print("   Generating embeddings...")
        embeddings = model.encode(texts, show_progress_bar=True)
        print(f"   ✅ Embeddings generated: {embeddings.shape}")
        
        return embeddings
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return None

# Step 4: Find optimal number of clusters
    if embeddings is None or len(embeddings) < 2:
        print("⚠️ Not enough documents for clustering")
        return 2
    
    silhouette_scores = []
    k_range = range(2, min(max_clusters, len(embeddings)))
    
    print(f"\n📊 Finding optimal clusters (trying {len(k_range)} options)...")
    
    for n_clusters in k_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"   Clusters: {n_clusters}, Score: {silhouette_avg:.3f}")
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Optimal Number of Clusters')
    plt.grid(True, alpha=0.3)
    plt.savefig('optimal_clusters.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    optimal_clusters = k_range[np.argmax(silhouette_scores)]
    print(f"\n✅ Optimal number of clusters: {optimal_clusters}")
    return optimal_clusters


# OVERRIDE: Use 4 clusters as requested
    optimal_k = 4
    print(f"\n✅ Using fixed number of clusters: {optimal_k}")
    


# Step 5: Perform clustering
def perform_clustering(embeddings, n_clusters, filenames):
    """Perform K-means clustering and visualize results"""
    print(f"\n🔮 Performing K-means clustering with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Reduce dimensionality for visualization
    print("   Creating t-SNE visualization...")
    perplexity = min(30, len(embeddings)-1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Create scatter plot
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=cluster_labels, cmap='viridis', s=200, alpha=0.7)
    
    # Add labels for each point
    for i, filename in enumerate(filenames):
        display_name = filename.replace('.txt', '')[:20]
        if len(filename) > 20:
            display_name += '...'
        plt.annotate(display_name, 
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=8, alpha=0.8, 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'ESG Document Clusters using Hugging Face Models\n{n_clusters} clusters, {len(filenames)} documents', 
              fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('clusters_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cluster_labels, embeddings_2d

# Step 6: Analyze clusters
def analyze_clusters(filenames, cluster_labels):
    """Print analysis of each cluster"""
    df_results = pd.DataFrame({
        'filename': filenames,
        'cluster': cluster_labels
    })
    
    print("\n" + "="*60)
    print("📊 CLUSTER ANALYSIS")
    print("="*60)
    
    for cluster_id in sorted(df_results['cluster'].unique()):
        cluster_files = df_results[df_results['cluster'] == cluster_id]['filename'].tolist()
        print(f"\n🔵 Cluster {cluster_id} ({len(cluster_files)} documents):")
        for i, file in enumerate(cluster_files, 1):
            print(f"   {i}. {file}")
    
    # Save to CSV
    df_results.to_csv('clustering_results.csv', index=False)
    print(f"\n💾 Results saved to 'clustering_results.csv'")
    
    return df_results

# Main execution
def main():
    print("="*60)
    print("🌍 ESG REPORT CLUSTERING PIPELINE")
    print("   Using Hugging Face Models")
    print("="*60)
    print("\n📌 Models used:")
    print("   1. Summarization: facebook/bart-large-cnn")
    print("      🔗 https://huggingface.co/facebook/bart-large-cnn")
    print("   2. Embeddings: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print("      🔗 https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print("="*60)
    
    # Check current directory
    print(f"\n📁 Current working directory: {os.getcwd()}")
    
    # Load documents
    filenames, texts = load_text_files("data")
    
    if not texts:
        print("\n❌ No text files found! Please make sure:")
        print("  1. You're running this script from your esg_project folder")
        print("  2. There's a 'data' folder in the current directory")
        print("  3. The data folder contains .txt files")
        return
    
    print(f"\n📄 Loaded {len(texts)} documents for processing")
    
    # Summarize documents
    print("\n" + "="*60)
    print(" STEP 1: SUMMARIZING DOCUMENTS")
    print("="*60)
    summaries = summarize_documents(texts, max_length=150, min_length=50)
    
    # Generate embeddings from summaries
    print("\n" + "="*60)
    print(" STEP 2: GENERATING EMBEDDINGS")
    print("="*60)
    summary_embeddings = generate_embeddings(summaries)
    
    if summary_embeddings is None:
        return
    
    # Find optimal clusters
    print("\n" + "="*60)
    print(" STEP 3: FINDING OPTIMAL CLUSTERS")
    print("="*60)
    optimal_k = find_optimal_clusters(summary_embeddings)
    
    # Perform clustering
    print("\n" + "="*60)
    print(" STEP 4: PERFORMING CLUSTERING")
    print("="*60)
    cluster_labels, _ = perform_clustering(summary_embeddings, optimal_k, filenames)
    
    # Analyze results
    print("\n" + "="*60)
    print(" STEP 5: ANALYZING RESULTS")
    print("="*60)
    df_results = analyze_clusters(filenames, cluster_labels)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    print("\n📁 Files created:")
    print("   - optimal_clusters.png (silhouette score plot)")
    print("   - clusters_visualization.png (t-SNE visualization)")
    print("   - clustering_results.csv (cluster assignments)")
    print("\n📊 Quick cluster summary:")
    print(df_results['cluster'].value_counts().sort_index())
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
