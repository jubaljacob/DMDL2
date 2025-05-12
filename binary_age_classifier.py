# Binary Age Classifier using Logistic Regression for BNC2014 Corpus
# This script implements a binary age classifier (Young vs Old) using logistic regression

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import lxml.etree as ET
from wordcloud import WordCloud

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_classif

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Set path to dataset
path = 'Dataset'
dir_corpus = os.path.join(path, 'spoken', 'tagged')
dir_meta = os.path.join(path, 'spoken', 'metadata')

print("Loading speaker metadata...")
# Load speaker metadata
fields_s = pd.read_csv(
    os.path.join(dir_meta, 'metadata-fields-speaker.txt'),
    sep='\t', skiprows=1, index_col=0
)

# Load the speaker metadata
df_speakers_meta = pd.read_csv(
    os.path.join(dir_meta, 'bnc2014spoken-speakerdata.tsv'),
    sep='\t', names=fields_s['XML tag'], index_col=0
)

print(f"Loaded metadata for {len(df_speakers_meta)} speakers")

# Function to map BNC age ranges to binary categories
def map_to_binary_age(age_range):
    """
    Map BNC age ranges to binary categories:
    Young (0-29) vs Old (30+)
    """
    if pd.isna(age_range):
        return np.nan
    
    # Handle different formats in the age range field
    try:
        # Extract the upper bound of the age range
        if '_' in str(age_range):
            ages = str(age_range).split('_')
        elif '-' in str(age_range):
            ages = str(age_range).split('-')
        else:
            return np.nan
        
        # Parse the upper bound
        if ages[1] == 'plus':
            upper = 100  # Arbitrarily high for '60_plus'
        else:
            upper = int(ages[1])
        
        # Classify as young or old
        if upper <= 29:
            return "Young"
        else:
            return "Old"
    except Exception as e:
        return np.nan

# Apply the binary age classification to speaker metadata
df_speakers_meta['binary_age'] = df_speakers_meta['agerange'].apply(map_to_binary_age)

# Display the counts for each binary age group
binary_age_counts = df_speakers_meta['binary_age'].value_counts()
print("Distribution of speakers by binary age classification:")
print(binary_age_counts)
print(f"Percentage: {100 * binary_age_counts / binary_age_counts.sum():.1f}%")

print("\nProcessing corpus data...")
# Process tagged corpus files to extract word and linguistic feature data
# We'll limit to 30 files to keep processing time reasonable
file_limit = 30  # Increase for more data

tagged_rows = []
try:
    # Load a subset of corpus files
    for file_count, fname in enumerate(sorted(os.listdir(dir_corpus))[:file_limit]):
        if file_count % 5 == 0:
            print(f"Processing file {file_count+1}/{file_limit}: {fname}")
            
        fpath = os.path.join(dir_corpus, fname)
        xml = ET.parse(fpath)
        root = xml.getroot()
        text_id = root.get('id')
        
        for u in root.findall('.//u'):
            utt_id = u.get('n')
            spk = u.get('who')
            for w in u.findall('w'):
                tagged_rows.append({
                    'text_id': text_id,
                    'utterance_id': utt_id,
                    'speaker_id': spk,
                    'word': w.text,
                    'lemma': w.get('lemma'),
                    'pos': w.get('pos'),
                    'class': w.get('class'),
                    'usas': w.get('usas'),
                })
    
    # Create a DataFrame from the extracted data
    df_tagged = pd.DataFrame(tagged_rows)
    
    print(f"\nLoaded {len(df_tagged)} word tokens from {file_limit} files")
    print(f"Found {df_tagged['speaker_id'].nunique()} unique speakers in the processed data")
    
except Exception as e:
    print(f"Error loading corpus data: {e}")

# Count of speakers with valid age data
valid_age_speakers = set(df_speakers_meta[~df_speakers_meta['binary_age'].isna()].index)
tagged_speakers = set(df_tagged['speaker_id'].unique())
valid_speakers = valid_age_speakers.intersection(tagged_speakers)

print(f"\nOf {len(tagged_speakers)} speakers in the corpus data, {len(valid_speakers)} have valid age data")

# Filter to only include speakers with valid age data
df_tagged_valid = df_tagged[df_tagged['speaker_id'].isin(valid_speakers)]
print(f"Filtered corpus data contains {len(df_tagged_valid)} word tokens from {len(valid_speakers)} speakers")

print("\nCreating speaker-level features...")
# Create speaker-level features
def create_speaker_features(df_tagged, df_speakers_meta):
    """Create a dataframe of speaker-level features for classification"""
    # Dictionary to store features for each speaker
    speaker_features = {}
    
    # Group by speaker_id
    for speaker_id, speaker_data in df_tagged.groupby('speaker_id'):
        if speaker_id not in df_speakers_meta.index:
            continue
            
        # Extract words and texts for this speaker
        words = speaker_data['word'].tolist()
        lemmas = speaker_data['lemma'].fillna('').tolist()
        pos_tags = speaker_data['pos'].fillna('').tolist()
        word_classes = speaker_data['class'].fillna('').tolist()
        
        # Skip speakers with too few words (< 50)
        if len(words) < 50:
            continue
            
        # Dictionary for this speaker's features
        features = {}
        
        # 1. Basic counts
        features['total_words'] = len(words)
        features['unique_words'] = len(set([w.lower() for w in words if w]))
        features['unique_lemmas'] = len(set([l.lower() for l in lemmas if l]))
        
        # 2. Lexical diversity (type-token ratio)
        features['lexical_diversity'] = features['unique_words'] / features['total_words']
        features['lemma_diversity'] = features['unique_lemmas'] / features['total_words']
        
        # 3. Part-of-speech distributions
        pos_counter = Counter(pos_tags)
        total_pos = sum(pos_counter.values())
        
        # Add normalized POS features (as percentages)
        for pos, count in pos_counter.most_common():
            if pos:  # Skip empty POS tags
                features[f'pos_{pos}'] = count / total_pos
                
        # 4. Word class distributions
        class_counter = Counter(word_classes)
        total_classes = sum(class_counter.values())
        
        # Add normalized word class features
        for wclass, count in class_counter.most_common():
            if wclass:  # Skip empty class tags
                features[f'class_{wclass}'] = count / total_classes
        
        # 5. Calculate average word length
        word_lengths = [len(w) for w in words if w]
        features['avg_word_length'] = sum(word_lengths) / len(word_lengths) if word_lengths else 0
        
        # 6. Count specific linguistic markers
        
        # Filler words and discourse markers
        filler_words = ['um', 'uh', 'er', 'erm', 'like', 'you know', 'i mean', 'well', 'so']
        filler_count = sum(1 for w in words if w and w.lower() in filler_words)
        features['filler_ratio'] = filler_count / features['total_words']
        
        # Store features for this speaker
        speaker_features[speaker_id] = features
    
    # Create DataFrame from features dictionary
    features_df = pd.DataFrame.from_dict(speaker_features, orient='index')
    
    # Add binary age from metadata
    features_df = features_df.join(
        df_speakers_meta[['binary_age', 'gender', 'edqual']], 
        how='left'
    )
    
    # Drop rows with missing binary_age
    features_df = features_df.dropna(subset=['binary_age'])
    
    return features_df

# Create speaker features dataframe
speakers_df = create_speaker_features(df_tagged_valid, df_speakers_meta)

print(f"Created feature dataframe with {len(speakers_df)} speakers")
print(f"Feature dataframe has {speakers_df.shape[1]} columns")

# Create a text corpus for each speaker for bag-of-words features
speaker_texts = {}

for speaker_id, speaker_data in df_tagged_valid.groupby('speaker_id'):
    if speaker_id in speakers_df.index:
        words = speaker_data['lemma'].fillna('').tolist()
        # Convert to lowercase and join with spaces
        speaker_texts[speaker_id] = ' '.join([w.lower() for w in words if w])

# Create a Series with speaker_id as index and text as values
speaker_corpus = pd.Series(speaker_texts)

print(f"Created text corpus for {len(speaker_corpus)} speakers")

print("\nPreparing data for classification...")
# Encode the target variable (binary_age)
speakers_df['age_label'] = (speakers_df['binary_age'] == 'Old').astype(int)

# Verify encoding
print("Binary age encoding:")
print(speakers_df[['binary_age', 'age_label']].drop_duplicates())

# Get features and target
X_features = speakers_df.drop(['binary_age', 'age_label', 'gender', 'edqual'], axis=1)
y = speakers_df['age_label']

# Get the corpus for the speakers in our feature set
corpus = speaker_corpus[X_features.index]

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test, corpus_train, corpus_test = train_test_split(
    X_features, y, corpus, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

print(f"Training set: {len(X_train)} speakers")
print(f"Test set: {len(X_test)} speakers")

# Create TF-IDF vectorizer for the text corpus
tfidf = TfidfVectorizer(
    max_features=300,  # Use top 300 features
    min_df=5,          # Minimum document frequency
    max_df=0.7,        # Maximum document frequency (remove very common words)
    stop_words='english' # Remove English stopwords
)

# Fit and transform the training corpus
X_train_tfidf = tfidf.fit_transform(corpus_train)

# Transform the test corpus
X_test_tfidf = tfidf.transform(corpus_test)

print(f"TF-IDF matrix shape for training data: {X_train_tfidf.shape}")
print(f"TF-IDF matrix shape for test data: {X_test_tfidf.shape}")

# Select top 20 features based on ANOVA F-value
selector = SelectKBest(f_classif, k=20)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Get the names of the selected features
selected_indices = selector.get_support(indices=True)
selected_features = X_train.columns[selected_indices]

print("\nTop 20 selected engineered features:")
print(selected_features.tolist())

# Scale the selected features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Combine TF-IDF features with selected engineered features
from scipy.sparse import hstack

X_train_combined = hstack([X_train_tfidf, X_train_scaled])
X_test_combined = hstack([X_test_tfidf, X_test_scaled])

print(f"\nCombined feature matrix shape for training: {X_train_combined.shape}")
print(f"Combined feature matrix shape for testing: {X_test_combined.shape}")

print("\nTraining logistic regression model...")
# Train a logistic regression model
log_reg = LogisticRegression(
    C=1.0,               # Regularization strength (inverse)
    penalty='l1',        # L1 regularization
    solver='liblinear',  # Solver that supports L1
    random_state=RANDOM_SEED,
    max_iter=1000
)

# Train the model
log_reg.fit(X_train_combined, y_train)

# Get predictions
y_pred = log_reg.predict(X_test_combined)
y_pred_proba = log_reg.predict_proba(X_test_combined)[:, 1]

print("Logistic Regression model trained successfully")

print("\nModel evaluation:")
# Calculate performance metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Young', 'Old']))

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Extract and sort feature coefficients
# First for TF-IDF features
tfidf_feature_names = np.array(tfidf.get_feature_names_out())
tfidf_coefs = log_reg.coef_[0][:len(tfidf_feature_names)]

# Get indices of the most important TF-IDF features (both positive and negative)
tfidf_indices = np.argsort(np.abs(tfidf_coefs))[::-1][:30]  # Top 30 features

# Create a DataFrame for TF-IDF features
tfidf_importance = pd.DataFrame({
    'Feature': tfidf_feature_names[tfidf_indices],
    'Coefficient': tfidf_coefs[tfidf_indices],
    'Type': 'Word Usage'
})

# Now for engineered features
eng_feature_names = np.array(selected_features)
eng_coefs = log_reg.coef_[0][len(tfidf_feature_names):]

# Sort engineered features by absolute coefficient value
eng_indices = np.argsort(np.abs(eng_coefs))[::-1]

# Create a DataFrame for engineered features
eng_importance = pd.DataFrame({
    'Feature': eng_feature_names[eng_indices],
    'Coefficient': eng_coefs[eng_indices],
    'Type': 'Linguistic Feature'
})

# Combine both types of features
feature_importance = pd.concat([tfidf_importance, eng_importance])

# Sort by absolute coefficient value
feature_importance['Abs_Coefficient'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False).reset_index(drop=True)

# Display the top 20 most important features
print("\nTop 20 most important features for age classification:")
print(feature_importance.head(20))

# Save the results to a file
feature_importance.to_csv('age_classification_feature_importance.csv', index=False)
print("\nFeature importance saved to 'age_classification_feature_importance.csv'")

print("\nAnalysis complete!")
