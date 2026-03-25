#!/usr/bin/env python3
"""
ESG Report Clustering with Ward's Hierarchical Method
Using LED model for summarization + Ward's clustering
"""

import os
import glob
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')
import torch

# ============================================
# CONFIGURATION
# ============================================
SUMMARIZATION_MODEL = "pszemraj/led-base-book-summary"
SUMMARIZATION_MODEL_LINK = "https://huggingface.co/pszemraj/led-base-book-summary"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL_LINK = "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MAX_INPUT_TOKENS = 16000
CHUNK_OVERLAP = 200

# ============================================
# STEP 1: LOAD TEXT FILES
# ============================================
def load_text_files(data_folder="data"):
    """Load all .txt files from the data folder"""
    texts = []
    filenames = []
    
    txt_files = glob.glob(os.path.join(data_folder, "*.txt"))
    print(f"Found {len(txt_files)} text files")
    
    if len(txt_files) == 0:
        print(f"No .txt files found in {data_folder} folder!")
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

# ============================================
# STEP 2: SMART DOCUMENT CHUNKING
# ============================================
def chunk_document(text, tokenizer, max_tokens=MAX_INPUT_TOKENS, overlap=CHUNK_OVERLAP):
    """Intelligently chunk a long document into overlapping segments"""
    tokens = tokenizer.encode(text)
    
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += max_tokens - overlap
    
    print(f"   Split into {len(chunks)} chunks")
    return chunks

# ============================================
# STEP 3: SUMMARIZE WITH LED MODEL
# ============================================
def summarize_documents(texts, max_summary_length=200, min_summary_length=80):
    """Generate summaries using LED model for very long documents"""
    print("\n📝 Initializing LED summarization pipeline...")
    print(f"   Model: {SUMMARIZATION_MODEL}")
    print(f"   🔗 {SUMMARIZATION_MODEL_LINK}")
    print(f"   Max input tokens: {MAX_INPUT_TOKENS} (handles VERY long documents!)")
    
    device = 0 if torch.cuda.is_available() else -1
    print(f"   Using device: {'GPU' if device == 0 else 'CPU'}")
    
    try:
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
        
        print("   Loading LED model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL)
        
        if device == 0:
            model = model.cuda()
        
        summaries = []
        
        for i, text in enumerate(texts):
            print(f"\n   📄 Document {i+1}/{len(texts)}")
            
            try:
                chunks = chunk_document(text, tokenizer)
                
                if len(chunks) == 1:
                    print(f"   Processing as single chunk ({len(text)} chars)")
                    
                    inputs = tokenizer.encode(
                        "summarize: " + text,
                        return_tensors="pt",
                        max_length=MAX_INPUT_TOKENS,
                        truncation=True
                    )
                    
                    if device == 0:
                        inputs = inputs.cuda()
                    
                    global_attention_mask = torch.zeros_like(inputs)
                    global_attention_mask[:, 0] = 1
                    
                    summary_ids = model.generate(
                        inputs,
                        global_attention_mask=global_attention_mask,
                        max_length=max_summary_length,
                        min_length=min_summary_length,
                        num_beams=4,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        repetition_penalty=3.5
                    )
                    
                    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    summaries.append(summary)
                    print(f"     ✅ Summary generated ({len(summary)} chars)")
                    
                else:
                    print(f"   Processing {len(chunks)} chunks...")
                    chunk_summaries = []
                    
                    for j, chunk in enumerate(chunks):
                        print(f"     Chunk {j+1}/{len(chunks)}...")
                        
                        inputs = tokenizer.encode(
                            "summarize: " + chunk,
                            return_tensors="pt",
                            max_length=MAX_INPUT_TOKENS,
                            truncation=True
                        )
                        
                        if device == 0:
                            inputs = inputs.cuda()
                        
                        global_attention_mask = torch.zeros_like(inputs)
                        global_attention_mask[:, 0] = 1
                        
                        chunk_ids = model.generate(
                            inputs,
                            global_attention_mask=global_attention_mask,
                            max_length=150,
                            min_length=50,
                            num_beams=4,
                            early_stopping=True
                        )
                        
                        chunk_summary = tokenizer.decode(chunk_ids[0], skip_special_tokens=True)
                        chunk_summaries.append(chunk_summary)
                    
                    combined = " ".join(chunk_summaries)
                    
                    if len(combined) > 2000:
                        inputs = tokenizer.encode(
                            "summarize: " + combined,
                            return_tensors="pt",
                            max_length=MAX_INPUT_TOKENS,
                            truncation=True
                        )
                        
                        if device == 0:
                            inputs = inputs.cuda()
                        
                        global_attention_mask = torch.zeros_like(inputs)
                        global_attention_mask[:, 0] = 1
                        
                        final_ids = model.generate(
                            inputs,
                            global_attention_mask=global_attention_mask,
                            max_length=max_summary_length,
                            min_length=min_summary_length,
                            num_beams=4,
                            early_stopping=True
                        )
                        
                        final_summary = tokenizer.decode(final_ids[0], skip_special_tokens=True)
                        summaries.append(final_summary)
                        print(f"     ✅ Final summary ({len(final_summary)} chars)")
                    else:
                        summaries.append(combined)
                        print(f"     ✅ Combined summary ({len(combined)} chars)")
                
            except Exception as e:
                print(f"     ❌ Error: {e}")
                if len(text) > 2000:
                    mid_point = len(text) // 2
                    summaries.append(text[mid_point:mid_point+1000] + "...")
                else:
                    summaries.append(text[:500] + "...")
        
        return summaries
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        summaries = []
        for text in texts:
            paragraphs = text.split('\n\n')
            start_idx = max(1, int(len(paragraphs) * 0.2))
            end_idx = int(len(paragraphs) * 0.8)
            middle_paragraphs = paragraphs[start_idx:end_idx]
            
            if middle_paragraphs:
                summaries.append('\n\n'.join(middle_paragraphs)[:1500])
            else:
                summaries.append(text[:1000])
        
        return summaries

# ============================================
# STEP 4: GENERATE EMBEDDINGS
# ============================================
def generate_embeddings(texts):
    """Generate embeddings using sentence transformers"""
    print(f"\n🔤 Loading embedding model...")
    print(f"   Model: {EMBEDDING_MODEL}")
    print(f"   🔗 {EMBEDDING_MODEL_LINK}")
    
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        print("   ✅ Model loaded")
        
        print("   Generating embeddings...")
        embeddings = model.encode(texts, show_progress_bar=True)
        print(f"   ✅ Embeddings: {embeddings.shape}")
        
        return embeddings
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# ============================================
# STEP 5: WARD'S HIERARCHICAL CLUSTERING
# ============================================
def wards_clustering(embeddings, filenames):
    """Perform Ward's hierarchical clustering with dendrogram visualization"""
    print("\n🌳 Performing Ward's hierarchical clustering...")
    
    # Compute linkage matrix using Ward's method
    print("   Computing linkage matrix...")
    linkage_matrix = linkage(embeddings, method='ward', metric='euclidean')
    
    # Create dendrogram
    plt.figure(figsize=(15, 10))
    plt.title('Hierarchical Clustering Dendrogram (Ward\'s Method)', fontsize=14)
    plt.xlabel('Documents', fontsize=12)
    plt.ylabel('Distance (Euclidean)', fontsize=12)
    
    # Create truncated labels
    labels = []
    for f in filenames:
        label = f.replace('.txt', '')[:15]
        labels.append(label)
    
    # Plot dendrogram
    dendrogram(linkage_matrix, 
               labels=labels,
               leaf_rotation=90,
               leaf_font_size=8,
               color_threshold=0.7 * max(linkage_matrix[:, 2]),
               above_threshold_color='gray')
    
    plt.tight_layout()
    plt.savefig('wards_dendrogram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Get distance range from dendrogram
    max_distance = max(linkage_matrix[:, 2])
    print(f"\n📊 Dendrogram Analysis:")
    print(f"   Maximum distance in dendrogram: {max_distance:.2f}")
    print(f"   Recommended cuts:")
    print(f"   - Cut at distance {max_distance*0.3:.2f}: ~2-3 clusters")
    print(f"   - Cut at distance {max_distance*0.5:.2f}: ~4-6 clusters")
    print(f"   - Cut at distance {max_distance*0.7:.2f}: ~7-10 clusters")
    
    # Ask user for number of clusters
    while True:
        try:
            n_clusters = int(input("\n🔢 Enter number of clusters based on dendrogram: "))
            if 2 <= n_clusters <= len(filenames):
                break
            else:
                print(f"Please enter a number between 2 and {len(filenames)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Perform hierarchical clustering with chosen number
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='euclidean',
        linkage='ward'
    )
    cluster_labels = clustering.fit_predict(embeddings)
    
    return cluster_labels, linkage_matrix, n_clusters

# ============================================
# STEP 6: VISUALIZE CLUSTERS WITH t-SNE
# ============================================
def visualize_clusters(embeddings, cluster_labels, filenames, n_clusters):
    """Visualize clusters using t-SNE"""
    print("\n🎨 Creating t-SNE visualization...")
    
    perplexity = min(30, len(embeddings)-1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(14, 10))
    
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=cluster_labels, cmap='viridis', s=200, alpha=0.7)
    
    for i, filename in enumerate(filenames):
        display_name = filename.replace('.txt', '')[:20]
        if len(filename) > 20:
            display_name += '...'
        plt.annotate(display_name, 
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=8, alpha=0.8, 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'ESG Document Clusters (Ward\'s Method)\n{n_clusters} clusters, {len(filenames)} documents', 
              fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('wards_clusters_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return embeddings_2d

# ============================================
# STEP 7: ANALYZE AND SAVE RESULTS
# ============================================
def analyze_clusters(filenames, cluster_labels, summaries=None):
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
    
    # Save summaries if provided
    if summaries:
        summary_df = pd.DataFrame({
            'filename': filenames,
            'summary': summaries,
            'cluster': cluster_labels
        })
        summary_df.to_csv('document_summaries_with_clusters.csv', index=False)
        print(f"💾 Summaries saved to 'document_summaries_with_clusters.csv'")
    
    return df_results

# ============================================
# MAIN EXECUTION
# ============================================
def main():
    print("="*60)
    print("🌍 ESG REPORT CLUSTERING WITH WARD'S METHOD")
    print("   Hierarchical Clustering + LED Summarization")
    print("="*60)
    print("\n📌 Models used:")
    print(f"   1. Summarization: {SUMMARIZATION_MODEL}")
    print(f"      🔗 {SUMMARIZATION_MODEL_LINK}")
    print(f"   2. Embeddings: {EMBEDDING_MODEL}")
    print(f"      🔗 {EMBEDDING_MODEL_LINK}")
    print(f"   Context Window: {MAX_INPUT_TOKENS} tokens")
    print("="*60)
    
    # Check current directory
    print(f"\n📁 Current directory: {os.getcwd()}")
    
    # Load documents
    filenames, texts = load_text_files("data")
    
    if not texts:
        print("\n❌ No text files found!")
        return
    
    print(f"\n📄 Loaded {len(texts)} documents")
    
    # Summarize documents
    print("\n" + "="*60)
    print(" STEP 1: SUMMARIZING WITH LED")
    print("="*60)
    summaries = summarize_documents(texts, max_summary_length=200, min_summary_length=80)
    
    # Generate embeddings
    print("\n" + "="*60)
    print(" STEP 2: GENERATING EMBEDDINGS")
    print("="*60)
    embeddings = generate_embeddings(summaries)
    
    if embeddings is None:
        return
    
    # Ward's hierarchical clustering
    print("\n" + "="*60)
    print(" STEP 3: WARD'S HIERARCHICAL CLUSTERING")
    print("="*60)
    cluster_labels, linkage_matrix, n_clusters = wards_clustering(embeddings, filenames)
    
    # Visualize with t-SNE
    visualize_clusters(embeddings, cluster_labels, filenames, n_clusters)
    
    # Analyze results
    print("\n" + "="*60)
    print(" STEP 4: ANALYZING RESULTS")
    print("="*60)
    df_results = analyze_clusters(filenames, cluster_labels, summaries)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    print("\n📁 Files created:")
    print("   - wards_dendrogram.png (hierarchical clustering tree)")
    print("   - wards_clusters_visualization.png (t-SNE visualization)")
    print("   - wards_clustering_results.csv (cluster assignments)")
    print("   - document_summaries_with_clusters.csv (summaries + clusters)")
    print("\n📊 Quick cluster summary:")
    print(df_results['cluster'].value_counts().sort_index())
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
