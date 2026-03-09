#!/usr/bin/env python3
"""
ESG Report Clustering Pipeline with Ward's Method
Using Hugging Face transformers for summarization and embeddings
"""

import os
import glob
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
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
                # Truncate long documents
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
                summaries.append(text[:500] + "...")
    
    except Exception as e:
        print(f"❌ Error with pipeline: {e}")
        print("   Using simple extractive summaries as fallback...")
        for text in texts:
            summaries.append(text[:500] + "...")
    
    return summaries

# Step 3: Generate embeddings using Sentence Transformers
def generate_embeddings(texts, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    """Generate embeddings using sentence transformers from Hugging Face"""
    print(f"\n🔤 Loading embedding model from Hugging Face: {model_name}")
    print(f"   Model URL: https://huggingface.co/sentence-transformers/{model_name}")
    
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

# Step 4: Perform Ward's hierarchical clustering and create dendrogram
def ward_clustering(embeddings, filenames):
    """Perform hierarchical clustering using Ward's method"""
    print("\n🌳 Performing hierarchical clustering with Ward's method...")
    
    # Compute linkage matrix using Ward's method
    linkage_matrix = linkage(embeddings, method='ward', metric='euclidean')
    
    # Create dendrogram
    plt.figure(figsize=(15, 8))
    plt.title('Hierarchical Clustering Dendrogram (Ward\'s Method)')
    plt.xlabel('Document Index')
    plt.ylabel('Distance')
    
    # Plot dendrogram with truncated labels
    dendrogram(linkage_matrix, 
               labels=[f"{i+1}:{f[:10]}..." for i, f in enumerate(filenames)],
               leaf_rotation=90,
               leaf_font_size=8)
    
    plt.tight_layout()
    plt.savefig('wards_dendrogram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Cut the dendrogram to get clusters (you can adjust the threshold)
    # Let's try different cluster numbers and show the dendrogram first
    print("\n📊 Based on the dendrogram, you can choose number of clusters")
    print("   Common choices: cut at distance 10-15 for fewer clusters")
    print("                   cut at distance 5-10 for more clusters")
    
    # Ask user for number of clusters
    while True:
        try:
            n_clusters = int(input("\nEnter number of clusters based on dendrogram: "))
            if 2 <= n_clusters <= len(filenames):
                break
            else:
                print(f"Please enter a number between 2 and {len(filenames)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Perform hierarchical clustering with chosen number of clusters
    clustering = AgglomerativeClustering(n_clusters=n_clusters, 
                                        metric='euclidean', 
                                        linkage='ward')
    cluster_labels = clustering.fit_predict(embeddings)
    
    return cluster_labels, linkage_matrix, n_clusters

# Step 5: Visualize clusters with t-SNE
def visualize_clusters(embeddings, cluster_labels, filenames, n_clusters):
    """Visualize clusters using t-SNE"""
    print("\n🎨 Creating t-SNE visualization...")
    
    # Reduce dimensionality for visualization
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
    plt.title(f'ESG Document Clusters (Ward\'s Method) - {n_clusters} clusters', 
              fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('wards_clusters_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

# Step 6: Analyze clusters
def analyze_clusters(filenames, cluster_labels):
    """Print analysis of each cluster"""
    df_results = pd.DataFrame({
        'filename': filenames,
        'cluster': cluster_labels
    })
    
    print("\n" + "="*60)
    print("📊 CLUSTER ANALYSIS (Ward's Method)")
    print("="*60)
    
    for cluster_id in sorted(df_results['cluster'].unique()):
        cluster_files = df_results[df_results['cluster'] == cluster_id]['filename'].tolist()
        print(f"\n🔵 Cluster {cluster_id} ({len(cluster_files)} documents):")
        for i, file in enumerate(cluster_files, 1):
            print(f"   {i}. {file}")
    
    # Save to CSV
    df_results.to_csv('wards_clustering_results.csv', index=False)
    print(f"\n💾 Results saved to 'wards_clustering_results.csv'")
    
    return df_results

# Step 7: Save summaries
def save_summaries(filenames, summaries):
    """Save summaries to files"""
    print("\n💾 Saving summaries...")
    
    # Save to CSV
    df_summaries = pd.DataFrame({
        'filename': filenames,
        'summary': summaries,
        'summary_length': [len(s) for s in summaries]
    })
    df_summaries.to_csv('document_summaries.csv', index=False)
    
    # Save individual text files
    summary_folder = "summaries"
    os.makedirs(summary_folder, exist_ok=True)
    
    for filename, summary in zip(filenames, summaries):
        txt_filename = filename.replace('.txt', '_summary.txt')
        with open(os.path.join(summary_folder, txt_filename), 'w', encoding='utf-8') as f:
            f.write(f"Original file: {filename}\n")
            f.write("="*50 + "\n")
            f.write(summary)
    
    print(f"   ✅ Summaries saved to '{summary_folder}/' folder and 'document_summaries.csv'")

# Main execution
def main():
    print("="*60)
    print("🌍 ESG REPORT CLUSTERING WITH WARD'S METHOD")
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
        print("\n❌ No text files found!")
        return
    
    print(f"\n📄 Loaded {len(texts)} documents for processing")
    
    # Summarize documents
    print("\n" + "="*60)
    print(" STEP 1: SUMMARIZING DOCUMENTS")
    print("="*60)
    summaries = summarize_documents(texts, max_length=150, min_length=50)
    
    # Save summaries
    save_summaries(filenames, summaries)
    
    # Generate embeddings from summaries
    print("\n" + "="*60)
    print(" STEP 2: GENERATING EMBEDDINGS")
    print("="*60)
    summary_embeddings = generate_embeddings(summaries)
    
    if summary_embeddings is None:
        return
    
    # Perform Ward's clustering
    print("\n" + "="*60)
    print(" STEP 3: WARD'S HIERARCHICAL CLUSTERING")
    print("="*60)
    cluster_labels, linkage_matrix, n_clusters = ward_clustering(summary_embeddings, filenames)
    
    # Visualize clusters
    visualize_clusters(summary_embeddings, cluster_labels, filenames, n_clusters)
    
    # Analyze results
    print("\n" + "="*60)
    print(" STEP 4: ANALYZING RESULTS")
    print("="*60)
    df_results = analyze_clusters(filenames, cluster_labels)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    print("\n📁 Files created:")
    print("   - wards_dendrogram.png (hierarchical clustering tree)")
    print("   - wards_clusters_visualization.png (t-SNE visualization)")
    print("   - wards_clustering_results.csv (cluster assignments)")
    print("   - document_summaries.csv (all summaries)")
    print("   - summaries/ folder (individual summary files)")
    print("\n📊 Quick cluster summary:")
    print(df_results['cluster'].value_counts().sort_index())
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
