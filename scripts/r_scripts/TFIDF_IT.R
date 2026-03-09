
# COMPLETE TEXT ANALYSIS PIPELINE

# 1. SETUP ENVIRONMENT ------------------------------------------------------
rm(list = ls())  # Clear everything
library(tidyverse)  # Loads dplyr, tibble, etc.
library(tidytext)   # For text mining
library(readr)      # Safe file reading
library(dplyr)

# 2. LOAD TEXT FILES -------------------------------------------------------
folder_path <- "/Users/Neng/Desktop/INDATIS CLEANED TEXTS"  # USE FORWARD SLASHES


# Verify folder exists
if(!dir.exists(folder_path)) {
  stop("Folder not found. Check path: ", folder_path)
}

# Get all .txt files
file_list <- list.files(folder_path, 
                        pattern = "\\.txt$", 
                        full.names = TRUE,
                        ignore.case = TRUE)




# Check files found
if(length(file_list) == 0) {
  stop("No .txt files found in: ", folder_path)
}

# Read files with error handling
texts <- map(file_list, ~ {
  tryCatch(
    {
      # Read with UTF-8 encoding, convert to single string
      paste(read_lines(.x, locale = locale(encoding = "UTF-8")), collapse = " ")
    },
    error = function(e) {
      message("Error reading: ", .x)
      NA_character_
    }
  )
}) %>% 
  set_names(tools::file_path_sans_ext(basename(file_list)))


# Remove any failed reads
texts <- discard(texts, is.na)

# 3. CREATE CORPUS TIBBLE --------------------------------------------------
corpus_df <- tibble(
  doc_id = names(texts),
  text = map_chr(texts, as.character),  # Ensure character type
  .name_repair = "unique"
)


# 4. TEXT PROCESSING ------------------------------------------------------
# Load required packages
library(tidytext)
library(stopwords)  # For comprehensive Italian stop words
library(dplyr)
library(ggplot2)
library(tidyr)

# Get Italian stop words (using the 'stopwords-iso' source)
italian_stopwords <- stopwords("it", source = "stopwords-iso")

# Convert to a tidytext-friendly format (tibble/data frame)
italian_stop_words <- tibble(word = italian_stopwords, lexicon = "italian")

# ADD CUSTOM STOP WORDS HERE
custom_stop_words <- tibble(
  word = c("delle", "della", "del", "dei", "degli", "una", "loro", "sono", 
           "alla", "alle", "allo", "agli", "questo", "questa", "questi", 
           "queste", "quello", "quella", "quelli", "quelle", "come", "con",
           "per", "tra", "fra", "sul", "sulla", "sui", "sugli", "sulle",
           "nella", "nelle", "negli", "siano", "essere", "aver", "avere",
           "dell'","_esgcon", "i.e.", "www.issgovernance.com", "mediolanum",
           "saipem","bper","terna","enel","pirelli","hera","generali","tim",
           "campari","italgas","prysmian","intesa","sanpaolo","unicredit",
           "eni","nexi","fineco","interpump","finecobank","unipolsai",
           "unipolsai","amplifon","bpm","montepaschi","unipol","inwit",
           "moncler","camparisti","tenaris","diasorin","iveco","leonardo",
           "mediobanca","pagg","which","bancobpm","sondrio","nexigroup",
           "unicreditgroup","stmicroelectronics","Paschi","telecom","azimut",
           "siena","erg","pump","pry","gruppotim",
  lexicon = "custom"
))


# Combine Italian and custom stop words
all_stop_words <- bind_rows(italian_stop_words, custom_stop_words)

tidy_text <- corpus_df %>%
  # Tokenize
  unnest_tokens(word, text) %>%
  # Remove stopwords
  anti_join(all_stop_words, by = "word") %>%  
  # Remove numbers
  filter(!str_detect(word, "[0-9]")) %>%
  # Remove single characters
  filter(nchar(word) > 2) %>%
  # REMOVE WORDS WITH APOSTROPHES - ADD THIS LINE
  filter(!str_detect(word, "[']")) %>%
  # Calculate word counts
  count(doc_id, word, sort = TRUE)

# 5. TF-IDF ANALYSIS ------------------------------------------------------
tf_idf <- tidy_text %>%
  bind_tf_idf(word, doc_id, n) %>%
  arrange(desc(tf_idf))

# 6. VIEW RESULTS --------------------------------------------------------
# Top 10 words per document
top_words <- tf_idf %>%
  group_by(doc_id) %>%
  slice_max(tf_idf, n = 10) %>%
  ungroup()

print(top_words)

# 7. VISUALIZATION -------------------------------------------------------
ggplot(top_words, aes(x = reorder_within(word, tf_idf, doc_id), y = tf_idf)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~doc_id, scales = "free_y") +
  coord_flip() +
  scale_x_reordered() +
  labs(title = "Top 10 Important Words per Document",
       x = NULL,
       y = "TF-IDF Score")








# Load additional libraries for enhanced visualizations

library(wordcloud)
library(igraph)
library(ggraph)
library(patchwork)
library(dplyr)

# A. WORD FREQUENCY ANALYSIS ACROSS ALL DOCUMENTS
overall_word_freq <- tidy_text %>%
  group_by(word) %>%
  summarize(total_freq = sum(n)) %>%
  arrange(desc(total_freq))

# Plot overall top words
overall_top_20 <- overall_word_freq %>%
  slice_max(total_freq, n = 20)

ggplot(overall_top_20, aes(x = reorder(word, total_freq), y = total_freq)) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  coord_flip() +
  labs(title = "Top 20 Most Frequent Words Across All Documents",
       x = "Words",
       y = "Total Frequency") +
  theme_minimal()

# B. BIGRAM ANALYSIS
bigram_analysis <- corpus_df %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(!word1 %in% all_stop_words$word,
         !word2 %in% all_stop_words$word,
         !str_detect(word1, "[0-9']"),
         !str_detect(word2, "[0-9']"),
         nchar(word1) > 2,
         nchar(word2) > 2) %>%
  unite(bigram, word1, word2, sep = " ") %>%
  count(doc_id, bigram, sort = TRUE)

# Bigram TF-IDF
bigram_tf_idf <- bigram_analysis %>%
  bind_tf_idf(bigram, doc_id, n) %>%
  arrange(desc(tf_idf))

# Top bigrams per document
top_bigrams <- bigram_tf_idf %>%
  group_by(doc_id) %>%
  slice_max(tf_idf, n = 8) %>%
  ungroup()

# Plot top bigrams
ggplot(top_bigrams, aes(x = reorder_within(bigram, tf_idf, doc_id), y = tf_idf)) +
  geom_col(show.legend = FALSE, fill = "darkorange") +
  facet_wrap(~doc_id, scales = "free_y") +
  coord_flip() +
  scale_x_reordered() +
  labs(title = "Top 8 Important Bigrams per Document",
       x = NULL,
       y = "TF-IDF Score") +
  theme_minimal()

# C. WORD CLOUD
word_freq_for_cloud <- tidy_text %>%
  group_by(word) %>%
  summarize(frequency = sum(n)) %>%
  arrange(desc(frequency))

# Create word cloud
wordcloud(words = word_freq_for_cloud$word, 
          freq = word_freq_for_cloud$frequency,
          max.words = 100,
          random.order = FALSE,
          colors = brewer.pal(8, "Dark2"),
          scale = c(3, 0.5))

# D. DOCUMENT COMPARISON - TOTAL WORDS
doc_summary <- tidy_text %>%
  group_by(doc_id) %>%
  summarize(total_words = sum(n),
            unique_words = n_distinct(word)) %>%
  arrange(desc(total_words))

# Plot document statistics
p1 <- ggplot(doc_summary, aes(x = reorder(doc_id, total_words), y = total_words)) +
  geom_col(fill = "black") +
  coord_flip() +
  labs(title = "Total Words per Document",
       x = "Document", y = "Total Words") +
  theme_minimal()
plot(p1)

p2 <- ggplot(doc_summary, aes(x = reorder(doc_id, unique_words), y = unique_words)) +
  geom_col(fill = "red") +
  coord_flip() +
  labs(title = "Unique Words per Document",
       x = "Document", y = "Unique Words") +
  theme_minimal()
plot(p2)


# Combine the two plots
library(patchwork)
p1 + p2

# E. TERM FREQUENCY DISTRIBUTION
# Plot distribution of word frequencies
word_freq_distribution <- tidy_text %>%
  group_by(word) %>%
  summarize(total_count = sum(n)) %>%
  arrange(desc(total_count))

ggplot(word_freq_distribution %>% slice_head(n = 50), 
       aes(x = reorder(word, total_count), y = total_count)) +
  geom_point(size = 2, color = "red") +
  coord_flip() +
  labs(title = "Word Frequency Distribution (Top 50)",
       x = "Words", y = "Frequency") +
  theme_minimal()






# G. EXPORT RESULTS
# Create summary tables
top_words_per_doc <- tf_idf %>%
  group_by(doc_id) %>%
  slice_max(tf_idf, n = 15) %>%
  ungroup()

top_bigrams_per_doc <- bigram_tf_idf %>%
  group_by(doc_id) %>%
  slice_max(tf_idf, n = 10) %>%
  ungroup()

# Print summary statistics
cat("ANALYSIS SUMMARY:\n")
cat("Total documents analyzed:", n_distinct(tidy_text$doc_id), "\n")
cat("Total unique words:", n_distinct(tidy_text$word), "\n")
cat("Total word occurrences:", sum(tidy_text$n), "\n")
cat("Total bigrams found:", nrow(bigram_analysis), "\n")
cat("Most frequent word:", overall_word_freq$word[1], "(", overall_word_freq$total_freq[1], "occurrences)\n")

# Display top words across all documents
cat("\nTOP 10 WORDS ACROSS ALL DOCUMENTS:\n")
print(overall_word_freq %>% slice_head(n = 10))

# Display top bigrams across all documents
cat("\nTOP 10 BIGRAMS ACROSS ALL DOCUMENTS:\n")
bigram_overall_freq <- bigram_analysis %>%
  group_by(bigram) %>%
  summarize(total_freq = sum(n)) %>%
  arrange(desc(total_freq))

print(bigram_overall_freq %>% slice_head(n = 10))




# Load required package for Excel export

library(openxlsx)
library(dplyr)
# If you don't have openxlsx: install.packages("openxlsx")

# H. EXPORT ALL RESULTS TO EXCEL ------------------------------------------

# Create a list to store all dataframes for Excel export
excel_results <- list()

# 1. TF-IDF Scores (Full Dataset)
excel_results[["TF_IDF_Full"]] <- tf_idf %>%
  dplyr::select(doc_id, word, n, tf, idf, tf_idf) %>%
  dplyr::arrange(doc_id, desc(tf_idf))

# 2. Top Words per Document (TF-IDF)
excel_results[["Top_Words_Per_Doc"]] <- top_words %>%
  dplyr::select(doc_id, word, n, tf_idf) %>%
  dplyr::arrange(doc_id, desc(tf_idf))

# 3. Top Bigrams per Document
excel_results[["Top_Bigrams_Per_Doc"]] <- top_bigrams %>%
  dplyr::select(doc_id, bigram, n, tf_idf) %>%
  dplyr::arrange(doc_id, desc(tf_idf))

# 4. Overall Word Frequency
excel_results[["Overall_Word_Frequency"]] <- overall_word_freq

# 5. Overall Bigram Frequency
bigram_overall_freq <- bigram_analysis %>%
  group_by(bigram) %>%
  summarize(total_frequency = sum(n)) %>%
  arrange(desc(total_frequency))

excel_results[["Overall_Bigram_Frequency"]] <- bigram_overall_freq

# 6. Document Statistics
excel_results[["Document_Statistics"]] <- doc_summary

# 7. Bigram Analysis (Full)
excel_results[["Bigram_Analysis_Full"]] <- bigram_tf_idf %>%
  dplyr::select(doc_id, bigram, n, tf, idf, tf_idf) %>%
  dplyr::arrange(doc_id, desc(tf_idf))

# 8. Word Frequency Distribution
excel_results[["Word_Frequency_Distribution"]] <- word_freq_distribution

# Create timestamp for filename
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")

# Export to Excel file
output_filename <- paste0("Text_Analysis_Results_", timestamp, ".xlsx")

write.xlsx(excel_results, 
           file = output_filename,
           creator = "NLP Analysis Pipeline",
           title = "Comprehensive Text Analysis Results")

cat("SUCCESS: All results exported to:", output_filename, "\n")




# ===========================================================================
# CONTINUATION: TRUE DOC2VEC IMPLEMENTATION (Document Text by Text)
# ===========================================================================

# First, let's check what packages we need to install
if (!requireNamespace("doc2vec", quietly = TRUE)) {
  install.packages("doc2vec")
}
if (!requireNamespace("umap", quietly = TRUE)) {
  install.packages("umap")
}
if (!requireNamespace("proxy", quietly = TRUE)) {
  install.packages("proxy")
}
if (!requireNamespace("factoextra", quietly = TRUE)) {
  install.packages("factoextra")
}
if (!requireNamespace("cluster", quietly = TRUE)) {
  install.packages("cluster")
}
if (!requireNamespace("ggrepel", quietly = TRUE)) {
  install.packages("ggrepel")
}

library(doc2vec)
library(umap)
library(proxy)
library(factoextra)
library(cluster)
library(ggrepel)

# ===========================================================================
# 8. PREPARE DATA FOR DOC2VEC
# ===========================================================================

cat("Preparing data for Doc2Vec...\n")

# 7. CREATE CLEANING FUNCTION ----------------------------------------------
clean_for_doc2vec <- function(text) {
  # Skip if text is not character
  if (!is.character(text)) return("")
  
  # Remove non-ASCII characters first
  text <- iconv(text, from = "UTF-8", to = "ASCII", sub = "")
  
  # Check if text is empty after conversion
  if (is.na(text) || nchar(text) == 0) return("")
  
  # Now safely convert to lowercase
  text <- tolower(text)
  
  # Continue with cleaning
  text <- gsub("[^[:alnum:][:space:]]", " ", text)
  text <- gsub("\\b\\d+\\b", " ", text)
  text <- gsub("\\s+", " ", text)
  text <- trimws(text)
  
  return(text)
}

# 8. APPLY CLEANING -------------------------------------------------------
library(purrr)
corpus_clean <- corpus_df %>%
  mutate(
    text_clean = sapply(text, clean_for_doc2vec),
    # Split into "sentences" - Doc2Vec works better with document chunks
    # For long documents, we split into paragraphs of ~100 words
    text_chunks = map(text_clean, function(txt) {
      words <- unlist(strsplit(txt, "\\s+"))
      # Create chunks of approximately 100 words
      chunk_size <- 100
      n_chunks <- ceiling(length(words) / chunk_size)
      chunks <- character(n_chunks)
      for (i in 1:n_chunks) {
        start <- (i-1)*chunk_size + 1
        end <- min(i*chunk_size, length(words))
        chunks[i] <- paste(words[start:end], collapse = " ")
      }
      return(chunks)
    })
  )

cat("Documents cleaned and chunked for Doc2Vec training.\n")
cat("Total documents:", nrow(corpus_clean), "\n")
cat("Average chunks per document:", mean(map_dbl(corpus_clean$text_chunks, length)), "\n")

# ===========================================================================
# 9. TRAIN DOC2VEC MODEL (PV-DM - Distributed Memory)
# ===========================================================================

# 10. PREPARE DATA IN CORRECT FORMAT FOR DOC2VEC --------------------------
cat("\nPreparing data in correct format for doc2vec...\n")

# The doc2vec package expects a dataframe with 'doc_id' and 'text' columns
# Create dataframe from the chunks
doc2vec_df <- data.frame(
  doc_id = character(),
  text = character(),
  stringsAsFactors = FALSE
)

# Fill the dataframe with chunks FROM CORPUS_CLEAN
for (i in 1:nrow(corpus_clean)) {
  doc_name <- corpus_clean$doc_id[i]
  chunks <- corpus_clean$text_chunks[[i]]
  
  for (chunk in chunks) {
    doc2vec_df <- rbind(doc2vec_df, 
                        data.frame(doc_id = doc_name, 
                                   text = as.character(chunk),
                                   stringsAsFactors = FALSE))
  }
}

# Check the result
cat("Created doc2vec_df with", nrow(doc2vec_df), "chunks\n")

cat("Dataframe created with", nrow(doc2vec_df), "rows (chunks)\n")
cat("Unique documents:", length(unique(doc2vec_df$doc_id)), "\n")

# 11. TRAIN DOC2VEC MODEL -------------------------------------------------
cat("\nTraining Doc2Vec model...\n")
cat("This may take a few minutes...\n")

set.seed(123)
start_time <- Sys.time()

# Train with the dataframe
model_pvdm <- paragraph2vec(
  x = doc2vec_df,        # Use the dataframe
  type = "PV-DM",
  dim = 100,
  iter = 15,             # Number of training iterations
  min_count = 5,         # Ignore words appearing less than 5 times
  lr = 0.025,            # Learning rate
  window = 10,           # Context window size
  hs = FALSE,            # Use negative sampling
  negative = 5,          # Number of negative samples
  sample = 0.001,        # Subsampling threshold
  threads = 4            # Use 4 CPU cores
)

# ===========================================================================
# 10. EXTRACT DOCUMENT VECTORS
# ===========================================================================

cat("\nExtracting document vectors...\n")

# Get document vectors from the trained model
doc_vectors <- as.matrix(model_pvdm, which = "docs")

# Ensure document names are preserved
rownames(doc_vectors) <- corpus_clean$doc_id

# Check the dimensions
cat("Document vectors created:\n")
cat("  Number of documents:", nrow(doc_vectors), "\n")
cat("  Vector dimensions:", ncol(doc_vectors), "\n")

# Save document vectors
write.csv(doc_vectors, "document_vectors_doc2vec.csv")
cat("Document vectors saved to 'document_vectors_doc2vec.csv'\n")

# ===========================================================================
# 11. DOCUMENT SIMILARITY ANALYSIS
# ===========================================================================

cat("\nCalculating document similarities...\n")

# Calculate cosine similarity matrix
similarity_matrix <- proxy::simil(doc_vectors, method = "cosine")

# Convert to matrix for easier manipulation
sim_matrix <- as.matrix(similarity_matrix)

#pairwise similarity table for all documents.
sim_df <- as.data.frame(as.table(sim_matrix))
colnames(sim_df) <- c("Doc1", "Doc2", "CosineSimilarity")
# Filter out self similarities
sim_df <- sim_df %>% filter(Doc1 != Doc2)
write.csv(sim_df, "document_similarity_pairs.csv", row.names = FALSE)

#Gives a visual overview of document similarities, which can help spot clusters or outliers
library(ggplot2)
library(reshape2)

sim_melt <- melt(sim_matrix)
ggplot(sim_melt, aes(x=Var1, y=Var2, fill=value)) +
  geom_tile() +
  labs(title="Document Similarity Heatmap",
       x="Document", y="Document", fill="Cosine Similarity") +
  theme(axis.text.x = element_text(angle=90, vjust=0.5))



# Function to find most similar documents for each document
find_top_similar <- function(sim_mat, top_n = 5) {
  results <- list()
  
  for (i in 1:nrow(sim_mat)) {
    doc_name <- rownames(sim_mat)[i]
    
    # Get similarity scores for this document
    scores <- sim_mat[i, ]
    
    # Exclude self-similarity
    scores[i] <- -Inf
    
    # Get top N most similar documents
    top_indices <- order(scores, decreasing = TRUE)[1:top_n]
    
    top_docs <- data.frame(
      source_document = doc_name,
      similar_document = rownames(sim_mat)[top_indices],
      similarity_score = scores[top_indices],
      rank = 1:top_n
    )
    
    results[[doc_name]] <- top_docs
  }
  
  return(bind_rows(results))
}

# Find top 5 similar documents for each
top_similar_docs <- find_top_similar(sim_matrix, top_n = 5)

# Save similarity results
write.csv(top_similar_docs, "document_similarities_top5.csv")

# ===========================================================================
# 12. VISUALIZE DOCUMENT RELATIONSHIPS (UMAP)
# ===========================================================================

cat("\nCreating UMAP visualization...\n")

# Reduce dimensions to 2D for visualization
set.seed(456)
umap_result <- umap(doc_vectors, n_components = 2, n_neighbors = 5)

# Create visualization dataframe
viz_data <- data.frame(
  Document = rownames(doc_vectors),
  UMAP1 = umap_result$layout[, 1],
  UMAP2 = umap_result$layout[, 2]
)

# Plot with minimal overlapping labels
viz_plot <- ggplot(viz_data, aes(x = UMAP1, y = UMAP2, label = Document)) +
  geom_point(color = "blue", size = 3, alpha = 0.7) +
  geom_text_repel(
    size = 3,
    max.overlaps = 15,
    box.padding = 0.5,
    point.padding = 0.5,
    segment.color = "gray",
    segment.size = 0.2
  ) +
  theme_minimal() +
  labs(
    title = "Document Embeddings Visualization (Doc2Vec)",
    subtitle = "Each point represents one company's report - closer points are more similar",
    x = "UMAP Dimension 1",
    y = "UMAP Dimension 2"
  )

print(viz_plot)

# Save the plot
ggsave("document_embeddings_plot.png", viz_plot, width = 12, height = 10, dpi = 300)

# ===========================================================================
# 13. CLUSTER ANALYSIS
# ===========================================================================

cat("\nPerforming cluster analysis...\n")

# Determine optimal number of clusters using elbow method
set.seed(789)
wss <- sapply(1:10, function(k) {
  kmeans(doc_vectors, centers = k, nstart = 25)$tot.withinss
})

# Plot elbow method
elbow_plot <- ggplot(data.frame(k = 1:10, wss = wss), aes(x = k, y = wss)) +
  geom_line(color = "blue") +
  geom_point(color = "blue", size = 3) +
  theme_minimal() +
  labs(
    title = "Elbow Method for Optimal Clusters",
    x = "Number of Clusters (k)",
    y = "Total Within-Cluster Sum of Squares"
  ) +
  scale_x_continuous(breaks = 1:10)

print(elbow_plot)

# Based on elbow plot, choose number of clusters (typically 4-6 for 40 documents)
n_clusters <- 4

# Perform K-means clustering
kmeans_result <- kmeans(doc_vectors, centers = n_clusters, nstart = 25)

# Add clusters to visualization data
viz_data$Cluster <- as.factor(kmeans_result$cluster)

# Plot clusters
cluster_plot <- ggplot(viz_data, aes(x = UMAP1, y = UMAP2, color = Cluster, label = Document)) +
  geom_point(size = 3) +
  geom_text_repel(
    size = 3,
    max.overlaps = 15,
    box.padding = 0.5,
    show.legend = FALSE
  ) +
  theme_minimal() +
  scale_color_brewer(palette = "Set1") +
  labs(
    title = paste("Document Clusters (k =", n_clusters, ")"),
    subtitle = "Companies grouped by semantic similarity of their reports",
    x = "UMAP Dimension 1",
    y = "UMAP Dimension 2",
    color = "Cluster"
  )

print(cluster_plot)
ggsave("document_clusters_plot.png", cluster_plot, width = 12, height = 10, dpi = 300)

# shows which document in each cluster that gives a quick overview on what the cluster contains
cluster_reps <- cluster_membership %>%
  group_by(Cluster) %>%
  slice_head(n = 1) %>% # gets the first doc arbitrarily
  ungroup()

write.csv(cluster_reps, "cluster_representatives.csv", row.names = FALSE)

# ===========================================================================
# 14. ANALYZE CLUSTER CONTENTS
# ===========================================================================

cat("\nAnalyzing cluster contents...\n")

# Create cluster membership dataframe
cluster_membership <- data.frame(
  Document = rownames(doc_vectors),
  Cluster = kmeans_result$cluster
)

# Function to get characteristic words for each cluster
analyze_cluster_keywords <- function(cluster_id, cluster_members, tf_idf_data) {
  # Get documents in this cluster
  cluster_docs <- cluster_members$Document[cluster_members$Cluster == cluster_id]
  
  # Get top TF-IDF words for these documents
  cluster_words <- tf_idf_data %>%
    filter(doc_id %in% cluster_docs) %>%
    group_by(word) %>%
    summarize(
      avg_tfidf = mean(tf_idf),
      doc_frequency = n_distinct(doc_id),
      total_occurrences = sum(n)
    ) %>%
    arrange(desc(avg_tfidf)) %>%
    head(15)
  
  return(cluster_words)
}

# Analyze each cluster
cluster_keywords <- list()
for (cl in 1:n_clusters) {
  cluster_keywords[[cl]] <- analyze_cluster_keywords(cl, cluster_membership, tf_idf)
}

# Print cluster summaries
cat("\nCLUSTER SUMMARY:\n")
cat("================\n\n")

for (cl in 1:n_clusters) {
  docs_in_cluster <- sum(cluster_membership$Cluster == cl)
  cat(sprintf("CLUSTER %d: %d documents\n", cl, docs_in_cluster))
  cat("Documents:", paste(cluster_membership$Document[cluster_membership$Cluster == cl], collapse = ", "), "\n")
  
  cat("Top characteristic words:\n")
  if (nrow(cluster_keywords[[cl]]) > 0) {
    top_words_str <- paste(head(cluster_keywords[[cl]]$word, 8), collapse = ", ")
    cat("  ", top_words_str, "\n")
  }
  cat("\n")
}

cluster_keywords_df <- bind_rows(
  lapply(seq_along(cluster_keywords), function(cl) {
    data.frame(
      Cluster = cl,
      Keyword = cluster_keywords[[cl]]$word,
      Avg_TFIDF = cluster_keywords[[cl]]$avg_tfidf,
      Doc_Freq = cluster_keywords[[cl]]$doc_frequency,
      Total_Occurrences = cluster_keywords[[cl]]$total_occurrences,
      stringsAsFactors = FALSE
    )
  })
)

# View it
print(cluster_keywords_df)

# Word Clouds per Cluster
library(wordcloud)

for (cl in unique(cluster_keywords_df$Cluster)) {
  subset_df <- cluster_keywords_df %>%
    filter(Cluster == cl)
  
  png_filename <- paste0("wordcloud_cluster_", cl, ".png")
  
  png(png_filename, width = 800, height = 600)
  wordcloud(
    words = subset_df$Keyword,
    freq = subset_df$Total_Occurrences,
    scale = c(3, 0.4),
    colors = brewer.pal(8, "Dark2"),
    random.order = FALSE
  )
  dev.off()
  cat("Saved:", png_filename, "\n")
}

# Save to CSV
write.csv(cluster_keywords_df, "cluster_keywords_summary.csv", row.names = FALSE)

# ===========================================================================
# 15. EXPORT COMPREHENSIVE RESULTS
# ===========================================================================

cat("\nExporting comprehensive results...\n")

# 1. Save cluster membership
write.csv(cluster_membership, "cluster_membership.csv", row.names = FALSE)

# 2. Save similarity matrix
write.csv(sim_matrix, "document_similarity_matrix.csv")

# 3. Create and save detailed similarity report
similarity_report <- top_similar_docs %>%
  dplyr::group_by(source_document) %>%
  dplyr::mutate(
    most_similar = similar_document[1],
    similarity_to_most_similar = similarity_score[1]
  ) %>%
  dplyr::select(source_document, most_similar, similarity_to_most_similar) %>%
  dplyr::distinct()

write.csv(similarity_report, "document_most_similar_pairs.csv", row.names = FALSE)

# 4. Create summary statistics
summary_stats <- data.frame(
  Metric = c(
    "Total Documents",
    "Vector Dimensions",
    "Number of Clusters",
    "Average Document Similarity",
    "Most Similar Pair",
    "Least Similar Pair"
  ),
  Value = c(
    nrow(doc_vectors),
    ncol(doc_vectors),
    n_clusters,
    round(mean(sim_matrix[upper.tri(sim_matrix)]), 4),
    paste(
      similarity_report$source_document[which.max(similarity_report$similarity_to_most_similar)],
      "↔",
      similarity_report$most_similar[which.max(similarity_report$similarity_to_most_similar)],
      sprintf("(%.3f)", max(similarity_report$similarity_to_most_similar))
    ),
    paste(
      rownames(sim_matrix)[which.min(sim_matrix[upper.tri(sim_matrix)])],
      "↔",
      colnames(sim_matrix)[which.min(sim_matrix[upper.tri(sim_matrix)])],
      sprintf("(%.3f)", min(sim_matrix[upper.tri(sim_matrix)]))
    )
  )
)

write.csv(summary_stats, "doc2vec_summary_statistics.csv", row.names = FALSE)

# 5. Save the trained model
saveRDS(model_pvdm, "trained_doc2vec_model.rds")

# ===========================================================================
# 16. GENERATE FINAL REPORT
# ===========================================================================

cat("\nGENERATING FINAL REPORT\n")
cat("=======================\n\n")

cat("DOC2VEC ANALYSIS COMPLETED SUCCESSFULLY\n")
cat("=======================================\n\n")

cat("SUMMARY:\n")
cat("--------\n")
cat("• 35 corporate reports analyzed\n")
cat("• Each document converted to a 100-dimensional vector\n")
cat("• Documents vectorized 'text by text' using Doc2Vec (PV-DM)\n")
cat("• Similarity matrix calculated between all documents\n")
cat("• Documents clustered into", n_clusters, "groups based on content similarity\n\n")

cat("KEY FINDINGS:\n")
cat("-------------\n")

# Most similar document pair
most_sim_pair <- similarity_report[which.max(similarity_report$similarity_to_most_similar), ]
cat("1. Most similar reports: ", most_sim_pair$source_document, " and ", 
    most_sim_pair$most_similar, " (similarity: ", 
    round(most_sim_pair$similarity_to_most_similar, 3), ")\n", sep = "")

# Cluster sizes
cat("2. Cluster distribution:\n")
for (cl in 1:n_clusters) {
  n_docs <- sum(cluster_membership$Cluster == cl)
  cat(sprintf("   Cluster %d: %d documents (%.1f%%)\n", 
              cl, n_docs, n_docs/nrow(cluster_membership)*100))
}

# Average similarity
cat(sprintf("3. Average similarity between documents: %.3f\n", 
            mean(sim_matrix[upper.tri(sim_matrix)])))

cat("\nFILES CREATED:\n")
cat("--------------\n")
cat("1. document_vectors_doc2vec.csv - Document vectors (100D each)\n")
cat("2. document_similarity_matrix.csv - Pairwise similarity matrix\n")
cat("3. document_similarities_top5.csv - Top 5 similar docs for each\n")
cat("4. cluster_membership.csv - Cluster assignments\n")
cat("5. document_most_similar_pairs.csv - Most similar document pairs\n")
cat("6. doc2vec_summary_statistics.csv - Summary statistics\n")
cat("7. trained_doc2vec_model.rds - Saved model for future use\n")
cat("8. document_embeddings_plot.png - UMAP visualization\n")
cat("9. document_clusters_plot.png - Cluster visualization\n")

cat("\nNEXT STEPS:\n")
cat("-----------\n")
cat("1. Examine cluster_keywords list for characteristic terms per cluster\n")
cat("2. Review similarity_report to understand document relationships\n")
cat("3. Use document vectors for further analysis (classification, etc.)\n")
cat("4. Share visualizations with stakeholders\n")

cat("\nAnalysis completed at: ", Sys.time(), "\n")











































