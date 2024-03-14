import pandas as pd
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2ForSequenceClassification
from sklearn.model_selection import train_test_split

# Data loading and cleaning
# Load the dataset from a CSV file.
tweet_df = pd.read_csv(r'../data/train.csv', encoding='utf-8')
# Drop the first column and any rows with missing values.
tweet_df = tweet_df.drop(tweet_df.columns[0], axis=1)
tweet_df = tweet_df.dropna(how='any')
# Convert the 'Label' column to integer and filter rows based on text length.
tweet_df['Label'] = tweet_df['Label'].astype(int)
tweet_df = tweet_df[(tweet_df['TextLen'] >= 2) & (tweet_df['TextLen'] <= 60)]

# Tokenization using GPT-2's tokenizer
# Initialize the tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# GPT-2 requires a specific token to start the sequence, which for GPT-2 is <|endoftext|>.
# Ensure this token is added to each sequence.
tweet_df['Text'] = tweet_df['Text'].apply(lambda x: tokenizer.bos_token + x + tokenizer.eos_token)
# Apply tokenization to the text. This converts text to a format the model can understand.
input_data = tokenizer(tweet_df['Text'].tolist(), padding=True, truncation=True, max_length=60, return_tensors="tf")

# Split the dataset into training and validation sets.
train_df, val_df = train_test_split(tweet_df, test_size=0.1, random_state=42, stratify=tweet_df['Label'])
train_labels = train_df['Label'].values
val_labels = val_df['Label'].values

# Define the model
# Initialize a GPT-2 model for sequence classification.
# Note that GPT2ForSequenceClassification is a custom or modified class, as GPT-2 is not natively designed for classification.
model = TFGPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3)  # Assuming there are 3 sentiment categories.

# Define the optimizer, loss function, and metric
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Train the model
# Note: We directly pass the `input_ids` and `attention_mask` from the tokenization step.
model.fit({"input_ids": input_data["input_ids"][train_df.index], "attention_mask": input_data["attention_mask"][train_df.index]}, train_labels,
          validation_data=({"input_ids": input_data["input_ids"][val_df.index], "attention_mask": input_data["attention_mask"][val_df.index]}, val_labels),
          epochs=3, batch_size=32)

# Save the model
save_path = r'../results/GPT2/GPT2'
model.save_pretrained(save_path)
