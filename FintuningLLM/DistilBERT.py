import pandas as pd
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification

# Reading the data
tweet_df = pd.read_csv(r'../data/train.csv', encoding='utf-8')

# Drop the first column and any rows with missing values.
tweet_df = tweet_df.drop(tweet_df.columns[0], axis=1)
tweet_df = tweet_df.dropna(how='any')

# Convert the 'Label' column to integer type and filter rows based on 'TextLen'.
tweet_df['Label'] = tweet_df['Label'].astype(int)
tweet_df = tweet_df[(tweet_df['TextLen'] >= 2) & (tweet_df['TextLen'] <= 60)]

# Splitting the dataset
# Divide the dataset into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(tweet_df['Text'], tweet_df['Label'], test_size=0.1)

# Initialize the tokenizer
# Create a tokenizer object for 'distilbert-base-uncased' model.
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the text
# Apply the tokenizer to the training and validation text.
X_train_tokenized = X_train.apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
X_val_tokenized = X_val.apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

# Load the model and configuration
# Instantiate a DistilBERT model for sequence classification.
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Compile the model
# Define the optimizer, loss function, and metrics for the model.
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Print the model summary
print(model.summary())

# Padding the tokenized input
MAX_LEN = 60  # Maximum sequence length
# Pad the tokenized sequences to a fixed length for both training and validation sets.
X_train_padded = tf.keras.preprocessing.sequence.pad_sequences(X_train_tokenized, maxlen=MAX_LEN, padding='post', truncating='post')
X_val_padded = tf.keras.preprocessing.sequence.pad_sequences(X_val_tokenized, maxlen=MAX_LEN, padding='post', truncating='post')

# Train the model
# Fit the model to the training data, using the validation set for evaluation.
model.fit(X_train_padded, y_train, validation_data=(X_val_padded, y_val), epochs=5, batch_size=32)

# Save the model weights
save_path = r'../results/DistilBERT/DistilBERT'
model.save_weights(save_path + '/distilbert_weights.h5')

# Test
# Reload a pre-trained DistilBERT model and its saved weights.
# loaded_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
# load path = r'../results/DistilBERT/DistilBERT'
# loaded_model.load_weights(save_path + '/distilbert_weights.h5')
