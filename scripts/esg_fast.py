#!/usr/bin/env python3
"""
Fast ESG Clustering (No Summarization)
"""

import os
import glob
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 FAST ESG CLUSTERING (NO SUMMARIZATION)")
print("="*60)

# Load files
txt_files = glob.glob("data/*.txt")
print(f"Found {len(txt_files)} files")

if len(txt_files) == 0:
    print("No files found in data folder!")
    exit()

texts = []
filenames = []
for file_path in txt_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
            filenames.append(os.path.basename(file_path))
        print(f"✅ Loaded: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"❌ Error loading {os.path.basename(file_path)}: {e}")

print(f"\n📄 Loaded {len(texts)} documents")

print("\n🔤 Loading embedding model from Hugging Face...")
print("   Model: paraphrase-multilingual-MiniLM-L12-v2")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("   Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)
print(f"   ✅ Embeddings generated: {embeddings.shape}")

# Find optimal clusters
print("\n📊 Finding optimal number of clusters...")
silhouette_scores = []
k_range = range(2, min(10, len(embeddings)))

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    silhouette_scores.append(score)
    print(f"   Clusters: {k}, Score: {score:.3f}")

optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\n✅ Optimal number of clusters: {optimal_k}")

# Final clustering
print("\n🔮 Performing K-means clustering...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings)

# Visualize
print("   Creating t-SNE visualization...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
embeddings_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(14, 10))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                     c=cluster_labels, cmap='viridis', s=200, alpha=0.7)

for i, fname in enumerate(filenames):
    short_name = fname.replace('.txt', '')[:15]
    plt.annotate(short_name, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                fontsize=8, alpha=0.8)

plt.colorbar(scatter, label='Cluster')
plt.title(f'ESG Document Clusters (Fast Mode) - {optimal_k} clusters')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.tight_layout()
plt.savefig('fast_clusters.png', dpi=300)
plt.show()

# Save results
df = pd.DataFrame({'filename': filenames, 'cluster': cluster_labels})
df.to_csv('fast_clustering_results.csv', index=False)

print("\n" + "="*60)
print("✅ COMPLETE!")
print("="*60)
print("\n📁 Files created:")
print("   - fast_clusters.png (visualization)")
print("   - fast_clustering_results.csv (cluster assignments)")
print("\n📊 Cluster summary:")
print(df['cluster'].value_counts().sort_index())
