import pandas as pd

# Load dataset
with open(r".\data\dataset.csv", 'r', encoding='Windows-1254', errors='replace') as f:
    tweet_df_2 = pd.read_csv(f)
    print(tweet_df_2.columns)

# Remove all rows that contain NaN values
tweet_df_2 = tweet_df_2.dropna()

# Keep only rows where the 'Language' column is 'en' (English)
tweet_df_2 = tweet_df_2[tweet_df_2['Language'] == 'en']

# Group by 'Label' column and count the number of unique values
grouped_labels = tweet_df_2.groupby('Label').size()

# Filter the rows where 'Label' is either 'negative', 'positive', or 'uncertainty'
filtered_tweet_df_2 = tweet_df_2[tweet_df_2['Label'].isin(['negative', 'positive', 'uncertainty'])]
print(filtered_tweet_df_2)

# Map 'positive' to 2, 'negative' to 0, and 'uncertainty' to 1 (for -1, modify this mapping)
tweet_df_2['Label'] = tweet_df_2['Label'].map({'negative': 0, 'uncertainty': 1, 'positive': 2})

# Remove duplicate entries based on 'Text' column
tweet_df_2.drop_duplicates(subset='Text', inplace=True)

# -----------------------------------------------------------------------------------------------------------------
# Ensure all entries are strings
tweet_df_2['Text'] = tweet_df_2['Text'].astype(str)

# Remove URLs and Mentions from tweets
tweet_df_2['Text'] = tweet_df_2['Text'].apply(lambda Text: re.sub(r"(?:\@|https?\://)\S+", "", Text))

# Remove extra spaces
tweet_df_2['Text'] = tweet_df_2['Text'].apply(lambda Text: re.sub("\s\s+", " ", Text))

# Filter out tweets that contain 5 consecutive question marks
tweet_df_2 = tweet_df_2[~tweet_df_2['Text'].apply(lambda x: '?????' in x)]

# Convert text to lowercase
tweet_df_2['Text'] = tweet_df_2['Text'].apply(lambda x: x.lower())

# -----------------------------------------------------------------------------------------------------------------
# Apply a function to split 'Text' column values and calculate the length of the resulting list
tweet_df_2['TextLen'] = tweet_df_2['Text'].apply(lambda x: len(x.split()))

# Using Seaborn to plot a distribution graph
plt.figure(figsize=(10, 5))
sns.histplot(tweet_df_2['TextLen'], kde=True)  # Plot a histogram with a kernel density estimate
plt.title('Distribution of Tweet Lengths')
plt.xlabel('Length of Tweet')
plt.ylabel('Frequency')
plt.show()

# -----------------------------------------------------------------------------------------------------------------
# Save the processed DataFrame to a CSV file
tweet_df_2.to_csv(r'./data/train.csv', encoding='utf-8')

