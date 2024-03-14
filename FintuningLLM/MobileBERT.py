import pandas as pd
from transformers import MobileBertTokenizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import TFMobileBertForSequenceClassification

# Read data
# Load training data from a CSV file.
tweet_df = pd.read_csv(r'./data/train.csv', encoding='utf-8')
# Drop the first column and rows with any missing values.
tweet_df = tweet_df.drop(tweet_df.columns[0], axis=1)
tweet_df = tweet_df.dropna(how='any')
# Convert the 'Label' column to integer type and filter rows based on 'TextLen' criteria.
tweet_df['Label'] = tweet_df['Label'].astype(int)
tweet_df = tweet_df[(tweet_df['TextLen'] >= 2) & (tweet_df['TextLen'] <= 60)]

# Using MobileBERT's Tokenizer
# Initialize the tokenizer for 'google/mobilebert-uncased'.
tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")

# Split data into training and validation sets
# Stratify split ensures training and validation have the same distribution of 'Label'.
train_df, val_df = train_test_split(tweet_df, test_size=0.1, random_state=42, stratify=tweet_df['Label'])

# Tokenize the training and validation datasets
# Tokenization converts text into a format that can be used by the model, with padding and truncation to ensure uniform length.
train_encodings = tokenizer(train_df['Text'].tolist(), truncation=True, padding=True, max_length=60, return_tensors='tf')
val_encodings = tokenizer(val_df['Text'].tolist(), truncation=True, padding=True, max_length=60, return_tensors='tf')

# Define the model
# Initialize a MobileBERT model for sequence classification with 3 sentiment labels assumed.
model = TFMobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=3)
print(model.summary())  # Print a summary of the model.

# Model compilation
# Configure the model for training with an optimizer, loss function, and evaluation metric.
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the model using tokenized training data, and validate with tokenized validation data.
model.fit(train_encodings['input_ids'], train_df['Label'].values,
          validation_data=(val_encodings['input_ids'], val_df['Label'].values),
          epochs=5, batch_size=32)

save_path = r'../results/MobileBERT/MobileBERT'
# Save the model weights
model.save_weights(save_path)
