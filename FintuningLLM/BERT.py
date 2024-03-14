import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification

# 1. Data Set Splitting
# Load data
tweet_df = pd.read_csv(r'../data/train.csv', encoding='utf-8')
# Drop the first column
tweet_df = tweet_df.drop(tweet_df.columns[0], axis=1)

# Remove rows with any missing values
tweet_df = tweet_df.dropna(how='any')

# Convert label column to integer type
tweet_df['Label'] = tweet_df['Label'].astype(int)

# Filter out tweets with text length less than 2 or greater than 60
tweet_df = tweet_df[(tweet_df['TextLen'] >= 2) & (tweet_df['TextLen'] <= 60)]

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(tweet_df, test_size=0.1, random_state=42)

# 2. Tokenization using BERT's Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_sentences(sentences):
    """
    Encode a list of sentences using BERT tokenizer.

    Parameters:
        sentences (list): List of input sentences to be encoded.

    Returns:
        dict: Encodings of the input sentences.
    """
    return tokenizer(sentences, padding=True, truncation=True, max_length=60, return_tensors="tf")

train_encodings = encode_sentences(train_df['Text'].tolist())
val_encodings = encode_sentences(val_df['Text'].tolist())

# 3. Building the BERT Model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
print(model.summary())

# 4. Training
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Create TensorFlow datasets for training and validation
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_df['Label'].tolist()
)).shuffle(1000).batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_df['Label'].tolist()
)).batch(32)

# Fit the model to the training data
model.fit(train_dataset, validation_data=val_dataset, epochs=5)

# 5. Saving Model Weights
save_path = r'../results/BERT/BERT'
model.save_weights(save_path)

# 6. Testing
# model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
# load_path = r'/results/BERT/BERT'
# model.load_weights(load_path)