<VSCode.Cell language="markdown">
## Dividing Speakers into Two Age Categories (Youngsters vs. Seniors)

Based on the research approach mentioned in the paper excerpt, we'll divide our speakers into just two age categories:
1. **Youngsters**: Combining the original "Youngsters (0-18)" and "Young Adults (19-29)" groups
2. **Seniors**: Combining the original "Adults (30-59)" and "Seniors (60+)" groups

This binary classification will allow us to examine broader age-related patterns in language use.
</VSCode.Cell>
<VSCode.Cell language="python">
# Create a new function to map age groups to a binary classification
def map_to_binary_age_group(age_group):
    """
    Map the existing four age groups into a binary classification:
    - Youngsters: includes original Youngsters (0-18) and Young Adults (19-29)
    - Seniors: includes original Adults (30-59) and Seniors (60+)
    
    Parameters:
    -----------
    age_group : str
        Original age group category
        
    Returns:
    --------
    str
        'Youngsters' or 'Seniors' classification
    """
    if pd.isna(age_group):
        return np.nan
        
    if "Youngsters" in str(age_group) or "Young Adults" in str(age_group):
        return "Youngsters"
    elif "Adults" in str(age_group) or "Seniors" in str(age_group):
        return "Seniors"
    else:
        return np.nan

# Apply the binary age group mapping to speaker_df
speaker_df['binary_age_group'] = speaker_df['age_group'].apply(map_to_binary_age_group)

# Check the distribution of speakers in each binary age group
binary_age_counts = speaker_df['binary_age_group'].value_counts()
print("Distribution of speakers by binary age classification:")
print(binary_age_counts)
print(f"Percentage: {100 * binary_age_counts / binary_age_counts.sum():.1f}%")

# Create separate dataframes for youngsters and seniors
youngsters_df = speaker_df[speaker_df['binary_age_group'] == 'Youngsters'].copy()
seniors_df = speaker_df[speaker_df['binary_age_group'] == 'Seniors'].copy()

print(f"\nYoungsters dataset: {len(youngsters_df)} speakers")
print(f"Seniors dataset: {len(seniors_df)} speakers")

# Display sample data from each group
print("\nSample of Youngsters data:")
print(youngsters_df[['binary_age_group', 'gender', 'total_words', 'vocab_size', 'lexical_diversity']].head(3))

print("\nSample of Seniors data:")
print(seniors_df[['binary_age_group', 'gender', 'total_words', 'vocab_size', 'lexical_diversity']].head(3))
</VSCode.Cell>
<VSCode.Cell language="markdown">
## Comparing Linguistic Features Between Age Groups

Let's explore differences in language use between Youngsters and Seniors, focusing on:
1. Lexical diversity
2. Word choice and vocabulary
3. Utterance length
4. Non-lexical vocalizations (as mentioned in the paper)
</VSCode.Cell>
<VSCode.Cell language="python">
# Visualize the differences in linguistic features between the two age groups
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1. Lexical diversity comparison
sns.boxplot(x='binary_age_group', y='lexical_diversity', data=speaker_df, 
            palette='viridis', ax=axes[0, 0])
axes[0, 0].set_title('Lexical Diversity by Age Group', fontsize=16)
axes[0, 0].set_xlabel('Age Group', fontsize=14)
axes[0, 0].set_ylabel('Lexical Diversity (Type-Token Ratio)', fontsize=14)

# Add statistical annotation
mean_lex_diversity = speaker_df.groupby('binary_age_group')['lexical_diversity'].mean()
for i, age_group in enumerate(mean_lex_diversity.index):
    axes[0, 0].text(i, mean_lex_diversity[age_group] + 0.01, 
                  f'Mean: {mean_lex_diversity[age_group]:.4f}', 
                  ha='center', fontsize=12)

# 2. Average utterance length comparison
sns.boxplot(x='binary_age_group', y='avg_utt_length', data=speaker_df, 
            palette='viridis', ax=axes[0, 1])
axes[0, 1].set_title('Average Utterance Length by Age Group', fontsize=16)
axes[0, 1].set_xlabel('Age Group', fontsize=14)
axes[0, 1].set_ylabel('Average Utterance Length (Words)', fontsize=14)

# Add statistical annotation
mean_utt_length = speaker_df.groupby('binary_age_group')['avg_utt_length'].mean()
for i, age_group in enumerate(mean_utt_length.index):
    axes[0, 1].text(i, mean_utt_length[age_group] + 0.5, 
                  f'Mean: {mean_utt_length[age_group]:.2f}', 
                  ha='center', fontsize=12)

# 3. Total vocabulary size comparison
sns.boxplot(x='binary_age_group', y='vocab_size', data=speaker_df, 
            palette='viridis', ax=axes[1, 0])
axes[1, 0].set_title('Vocabulary Size by Age Group', fontsize=16)
axes[1, 0].set_xlabel('Age Group', fontsize=14)
axes[1, 0].set_ylabel('Vocabulary Size (Unique Words)', fontsize=14)
axes[1, 0].set_yscale('log')

# Add statistical annotation
median_vocab_size = speaker_df.groupby('binary_age_group')['vocab_size'].median()
for i, age_group in enumerate(median_vocab_size.index):
    axes[1, 0].text(i, median_vocab_size[age_group] * 1.2, 
                  f'Median: {median_vocab_size[age_group]:.0f}', 
                  ha='center', fontsize=12)

# 4. Gender distribution within age groups
gender_age_counts = pd.crosstab(speaker_df['binary_age_group'], speaker_df['gender'])
gender_age_counts.plot(kind='bar', stacked=True, ax=axes[1, 1], colormap='viridis')
axes[1, 1].set_title('Gender Distribution by Age Group', fontsize=16)
axes[1, 1].set_xlabel('Age Group', fontsize=14)
axes[1, 1].set_ylabel('Number of Speakers', fontsize=14)
axes[1, 1].legend(title='Gender')

# Add count labels
for i, age_group in enumerate(gender_age_counts.index):
    total = gender_age_counts.loc[age_group].sum()
    axes[1, 1].text(i, total + 5, f'Total: {total}', ha='center', fontsize=12)

plt.tight_layout()
plt.savefig('binary_age_group_comparisons.png', dpi=300)
plt.show()
</VSCode.Cell>
<VSCode.Cell language="markdown">
## Analysis of Non-lexical Vocalizations and Turn-taking Features

As mentioned in the paper, we'll examine non-lexical vocalizations and turn-taking features that might differ between age groups, including:

1. Positive responses and continuers (e.g., "mm", "mhm", "uhuh")
2. Turn stalling expressions (e.g., "hmm", "hmmm")
3. Turn management tokens (e.g., "um", "er", "erm")
4. Repair initiators (e.g., "hm?")
5. Change-of-state tokens (e.g., "oh")
</VSCode.Cell>
<VSCode.Cell language="python">
# Function to identify non-lexical vocalizations in text
def analyze_vocalizations(text_series):
    """
    Analyze a series of texts for various non-lexical vocalizations.
    
    Parameters:
    -----------
    text_series : pandas.Series
        Series containing utterance texts
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with counts of different vocalization types
    """
    # Define regular expressions for different vocalization types
    vocalization_patterns = {
        'positive_response': r'\b(mm|mhm|uhu|uhuh|aha|uh huh)\b',
        'turn_stalling': r'\b(hmm|hmmm)\b',
        'turn_management': r'\b(um|er|erm)\b',
        'repair_initiator': r'\bhm\?\b',
        'change_of_state': r'\boh\b'
    }
    
    # Count occurrences for each speaker
    results = {}
    
    for pattern_name, regex in vocalization_patterns.items():
        # Count occurrences in each text
        results[pattern_name] = text_series.str.count(regex, flags=re.IGNORECASE).fillna(0)
    
    # Create DataFrame with results
    return pd.DataFrame(results)

# Apply analysis to the utterance texts for each speaker
vocalization_df = analyze_vocalizations(speaker_df['all_utterances'])

# Add binary age group column
vocalization_df['binary_age_group'] = speaker_df['binary_age_group'] 
vocalization_df['gender'] = speaker_df['gender']
vocalization_df['total_words'] = speaker_df['total_words']

# Calculate normalized frequencies (per 1000 words)
for col in ['positive_response', 'turn_stalling', 'turn_management', 'repair_initiator', 'change_of_state']:
    vocalization_df[f'{col}_per_1k'] = vocalization_df[col] / vocalization_df['total_words'] * 1000

# Aggregate results by age group
agg_by_age = vocalization_df.groupby('binary_age_group').agg({
    'positive_response_per_1k': 'mean',
    'turn_stalling_per_1k': 'mean',
    'turn_management_per_1k': 'mean',
    'repair_initiator_per_1k': 'mean',
    'change_of_state_per_1k': 'mean'
})

# Create comparative visualization
fig, ax = plt.subplots(figsize=(12, 8))

# Transpose to get features as x-axis
agg_by_age_t = agg_by_age.T

# Plot
agg_by_age_t.plot(kind='bar', ax=ax, width=0.7)
ax.set_title('Non-lexical Vocalizations by Age Group (per 1000 words)', fontsize=16)
ax.set_ylabel('Frequency per 1000 words', fontsize=14)
ax.set_xlabel('Vocalization Type', fontsize=14)
ax.set_xticklabels([x.replace('_per_1k', '').replace('_', ' ').title() for x in agg_by_age_t.index])

# Add value labels
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', fontsize=10)

plt.legend(title='Age Group')
plt.tight_layout()
plt.savefig('vocalization_by_binary_age.png', dpi=300)
plt.show()

# Also analyze by gender within each age group
agg_by_age_gender = vocalization_df.groupby(['binary_age_group', 'gender']).agg({
    'positive_response_per_1k': 'mean',
    'turn_stalling_per_1k': 'mean', 
    'turn_management_per_1k': 'mean',
    'repair_initiator_per_1k': 'mean',
    'change_of_state_per_1k': 'mean'
}).reset_index()

# Filter to only include M/F genders for clarity
agg_by_age_gender = agg_by_age_gender[agg_by_age_gender['gender'].isin(['M', 'F'])]

# Create a figure for each vocalization type
vocalization_types = [col for col in agg_by_age_gender.columns if col.endswith('_per_1k')]

fig, axes = plt.subplots(len(vocalization_types), 1, figsize=(12, 4*len(vocalization_types)))

for i, v_type in enumerate(vocalization_types):
    # Create temporary DataFrame for plotting
    plot_df = agg_by_age_gender.pivot(index='binary_age_group', columns='gender', values=v_type)
    plot_df.plot(kind='bar', ax=axes[i], width=0.7)
    
    # Set titles and labels
    axes[i].set_title(f'{v_type.replace("_per_1k", "").replace("_", " ").title()} by Age Group and Gender', fontsize=14)
    axes[i].set_ylabel('Frequency per 1000 words', fontsize=12)
    axes[i].set_xlabel('Age Group', fontsize=12)
    
    # Add value labels
    for container in axes[i].containers:
        axes[i].bar_label(container, fmt='%.2f', fontsize=10)

plt.tight_layout()
plt.savefig('vocalization_by_age_and_gender.png', dpi=300)
plt.show()
</VSCode.Cell>
<VSCode.Cell language="markdown">
## Characteristic Words and Phrases by Age Group

Following the approach in the paper, we'll identify words and phrases that are characteristic of each age group using a frequency analysis.
</VSCode.Cell>
<VSCode.Cell language="python">
import numpy as np
from collections import Counter
import re

def get_characteristic_words(df_tagged, speaker_ids_by_group, min_count=10, top_n=20):
    """
    Find characteristic words for each group using a simple frequency analysis.
    
    Parameters:
    -----------
    df_tagged : pandas.DataFrame
        DataFrame containing the tagged words with speaker_id
    speaker_ids_by_group : dict
        Dictionary mapping group names to lists of speaker IDs
    min_count : int
        Minimum count for a word to be considered
    top_n : int
        Number of top words to return for each group
        
    Returns:
    --------
    dict
        Dictionary containing characteristic words for each group
    """
    # Define stopwords to exclude
    stopwords = set([
        'the', 'and', 'to', 'a', 'of', 'in', 'i', 'it', 'that', 'you', 'is', 
        'for', 'on', 'have', 'with', 'be', 'this', 'are', 'was', 'but', 'not', 
        'they', 'so', 'we', 'what', 'um', 'er', 'erm', 'eh', 'mm'
    ])
    
    # Collect words by group
    word_counts_by_group = {}
    total_words_by_group = {}
    
    for group, speaker_ids in speaker_ids_by_group.items():
        # Get words for all speakers in this group
        group_words = df_tagged[df_tagged['speaker_id'].isin(speaker_ids)]['word']
        
        # Clean and normalize words
        clean_words = []
        for word in group_words:
            if pd.notna(word):
                # Convert to lowercase and remove punctuation
                clean_word = re.sub(r'[^\w\s]', '', str(word).lower())
                if clean_word and clean_word not in stopwords and len(clean_word) > 1:
                    clean_words.append(clean_word)
        
        # Count words
        word_counts = Counter(clean_words)
        word_counts_by_group[group] = word_counts
        total_words_by_group[group] = len(clean_words)
    
    # Calculate scaled frequency for each word in each group
    characteristic_words = {}
    
    for group in speaker_ids_by_group.keys():
        scores = {}
        
        for word, count in word_counts_by_group[group].items():
            if count < min_count:
                continue
                
            # Calculate frequency in this group
            freq_in_group = count / total_words_by_group[group]
            
            # Calculate frequency in other groups
            other_groups = [g for g in speaker_ids_by_group.keys() if g != group]
            
            other_count = 0
            other_total = 0
            for other_group in other_groups:
                other_count += word_counts_by_group[other_group].get(word, 0)
                other_total += total_words_by_group[other_group]
            
            freq_in_others = other_count / other_total if other_total > 0 else 0
            
            # Calculate ratio of frequencies (smoothed to avoid division by zero)
            epsilon = 1e-10
            specificity = np.log2((freq_in_group + epsilon) / (freq_in_others + epsilon))
            
            # Store word with its specificity score and raw count
            scores[word] = {
                'specificity': specificity,
                'count': count,
                'frequency': freq_in_group * 1000  # per 1000 words
            }
        
        # Sort by specificity and get top N
        sorted_words = sorted(scores.items(), key=lambda x: x[1]['specificity'], reverse=True)[:top_n]
        characteristic_words[group] = {word: data for word, data in sorted_words}
    
    return characteristic_words

# Get speaker IDs for each binary age group
youngsters_ids = youngsters_df.index.tolist()
seniors_ids = seniors_df.index.tolist()

speaker_ids_by_group = {
    'Youngsters': youngsters_ids,
    'Seniors': seniors_ids
}

# Find characteristic words
characteristic_words = get_characteristic_words(df_tagged, speaker_ids_by_group)

# Print results
for group, words in characteristic_words.items():
    print(f"\nTop characteristic words for {group}:")
    print("-" * 50)
    print(f"{'Word':<15} {'Specificity':>12} {'Count':>8} {'Per 1000':>10}")
    print("-" * 50)
    
    for word, data in words.items():
        print(f"{word:<15} {data['specificity']:>12.2f} {data['count']:>8} {data['frequency']:>10.2f}")

# Visualize the results
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

for i, (group, words) in enumerate(characteristic_words.items()):
    # Extract data for plotting
    word_list = list(words.keys())
    specificity_scores = [data['specificity'] for data in words.values()]
    frequencies = [data['frequency'] for data in words.values()]
    
    # Sort by specificity
    sorted_indices = np.argsort(specificity_scores)
    word_list = [word_list[i] for i in sorted_indices]
    specificity_scores = [specificity_scores[i] for i in sorted_indices]
    frequencies = [frequencies[i] for i in sorted_indices]
    
    # Plot horizontal bar chart
    y_pos = np.arange(len(word_list))
    bars = axes[i].barh(y_pos, specificity_scores, align='center', 
                       color=plt.cm.viridis(np.array(frequencies) / max(frequencies)))
    
    # Add frequency as text
    for j, (bar, freq) in enumerate(zip(bars, frequencies)):
        axes[i].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                   f"{freq:.1f}/1k", va='center')
    
    axes[i].set_yticks(y_pos)
    axes[i].set_yticklabels(word_list)
    axes[i].invert_yaxis()  # Highest values at the top
    axes[i].set_xlabel('Specificity Score', fontsize=12)
    axes[i].set_title(f'Characteristic Words for {group}', fontsize=16)
    
    # Add a colorbar legend
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(frequencies), vmax=max(frequencies)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[i])
    cbar.set_label('Frequency per 1000 words', rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig('characteristic_words_by_binary_age.png', dpi=300)
plt.show()

# Create a combined visualization of the most characteristic words as word clouds
from wordcloud import WordCloud

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

for i, (group, words) in enumerate(characteristic_words.items()):
    # Create a dictionary of words and their specificity scores
    word_scores = {word: data['specificity'] for word, data in words.items()}
    
    # Generate a word cloud
    wordcloud = WordCloud(
        width=800, 
        height=800,
        background_color='white',
        colormap='viridis',
        max_words=100,
        contour_width=1,
        contour_color='steelblue',
        relative_scaling=0.8  # Influence of frequency on size
    ).generate_from_frequencies(word_scores)
    
    # Display the word cloud
    axes[i].imshow(wordcloud, interpolation='bilinear')
    axes[i].set_title(f'Characteristic Words for {group}', fontsize=16)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('wordclouds_characteristic_words_binary_age.png', dpi=300)
plt.show()
</VSCode.Cell>
<VSCode.Cell language="markdown">
## Summary of Binary Age Group Analysis

The analysis above explores linguistic differences between two age groups (Youngsters and Seniors) in the BNC2014 corpus, examining:

1. **Basic linguistic measures**:
   - Lexical diversity
   - Utterance length
   - Vocabulary size

2. **Non-lexical vocalizations and discourse markers**:
   - Positive responses and continuers (e.g., "mm", "mhm")
   - Turn management tokens (e.g., "er", "um")
   - Change-of-state tokens (e.g., "oh")

3. **Characteristic vocabulary**:
   - Words that are distinctively common in each age group
   - Distribution patterns across groups

These analyses help identify age-specific language patterns that could be useful for sociolinguistic research and applications like automated speaker demographic classification as mentioned in the paper.
</VSCode.Cell>
