import pandas as pd
from transformers import RobertaTokenizer
import tensorflow as tf
from transformers import TFRobertaForSequenceClassification
from sklearn.model_selection import train_test_split

# Data loading and cleaning
# Load the data from a CSV file.
tweet_df = pd.read_csv(r'../data/train.csv', encoding='utf-8')
# Drop the first column and any rows with missing values.
tweet_df = tweet_df.drop(tweet_df.columns[0], axis=1)
tweet_df = tweet_df.dropna(how='any')
# Convert the 'Label' column to integer and filter rows based on text length.
tweet_df['Label'] = tweet_df['Label'].astype(int)
tweet_df = tweet_df[(tweet_df['TextLen'] >= 2) & (tweet_df['TextLen'] <= 60)]

# Tokenization using RoBERTa's tokenizer
# Initialize the tokenizer.
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# Apply tokenization to the text. The output will include `input_ids` and `attention_mask`.
tweet_df['input_data'] = tweet_df['Text'].apply(lambda x: tokenizer.encode_plus(x, add_special_tokens=True, max_length=60, padding='max_length', truncation=True))
tweet_df['input_ids'] = tweet_df['input_data'].apply(lambda x: x['input_ids'])
tweet_df['attention_mask'] = tweet_df['input_data'].apply(lambda x: x['attention_mask'])

# Split the dataset into training and validation sets.
train_df, val_df = train_test_split(tweet_df, test_size=0.1, random_state=42)

# Process data for training and validation sets.
# Convert lists of input_ids and attention_masks into tensors.
train_input_ids = tf.stack(train_df['input_ids'].to_list(), axis=0)
train_attention_masks = tf.stack(train_df['attention_mask'].to_list(), axis=0)
train_labels = train_df['Label'].values

val_input_ids = tf.stack(val_df['input_ids'].to_list(), axis=0)
val_attention_masks = tf.stack(val_df['attention_mask'].to_list(), axis=0)
val_labels = val_df['Label'].values

# Define the model
# Initialize a RoBERTa model for sequence classification with 3 sentiment categories.
model = TFRobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)

# Define the optimizer, loss function, and metric
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
print(model.summary())  # Print model summary for verification.

# Train the model
model.fit([train_input_ids, train_attention_masks], train_labels, 
          validation_data=([val_input_ids, val_attention_masks], val_labels), 
          epochs=5, batch_size=32)

# Save the model
save_path = r'../results/RoBERT/RoBERT'
model.save(save_path)
