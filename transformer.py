import tensorflow as tf
import pandas as pd
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
from tensorflow.keras import layers
from tensorflow.keras.layers import GaussianNoise
import gc
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizerFast
from tensorflow.keras import layers, preprocessing

# Check for GPU availability and set up the first detected GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        gpu_device_details = tf.config.experimental.get_device_details(gpu)
        print(gpu_device_details)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
else:
    print("No GPU detected.")

################################################## - Data Preparing - ##################################################
# 
# Tokenization; Dataset Split; Padding; Padding Mask; One-hot Encoding
#
################################################## - -------------- - ##################################################
#
# Purpose of padding
# Models typically require fixed-length inputs
# 
# Sentence 1: [2, 4]
# Sentence 2: [3, 5, 7]
# Sentence 3: [1, 6, 8, 9]
# After padding
# Sentence 1: [2, 4, 0, 0, 0]
# Sentence 2: [3, 5, 7, 0, 0]
# Sentence 3: [1, 6, 8, 9, 0]
# 
################################################## - -------------- - ##################################################
#
# On-hot encoding
# One-hot encoding and word embedding can be used simultaneously. In natural language processing tasks, especially when using sequence models such as RNNs or Transformers, 
# one-hot encoding and embeddings may be employed in different parts or stages of the model. For instance, the input layer might utilize one-hot encoding to represent individual characters, 
# while embeddings are used at higher levels to represent words or phrases.
#
################################################## - -------------- - ##################################################

# Preprocessing data
tweet_df = pd.read_csv(r'E:\TF_GPU\data\Transformer\train.csv', encoding='utf-8')
tweet_df = tweet_df.drop(tweet_df.columns[0], axis=1)
tweet_df = tweet_df.dropna(how='any')
tweet_df['Label'] = tweet_df['Label'].astype(int)

# Filtering tweets based on their text length
tweet_df = tweet_df[(tweet_df['TextLen'] >= 2) & (tweet_df['TextLen'] <= 60)]

# Tokenizing tweets using BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
tweet_df['Token'] = tweet_df['Text'].apply(lambda txt: tokenizer.encode(txt, max_length=512, truncation=True))
tweet_df['TokenLen'] = tweet_df['Token'].apply(lambda _list: len(_list))
tweet_df = tweet_df[(tweet_df['TokenLen'] >= 2) & (tweet_df['TokenLen'] <= 75)]

MaxTokenLen = tweet_df['TokenLen'].max()

# Shuffling and splitting the dataset
tweet_df = tweet_df.sample(frac=1).reset_index(drop=True)
X_train, X_valid, y_train, y_valid = train_test_split(
    np.array(tweet_df['Token']), 
    np.array(tweet_df['Label']), 
    test_size=0.1, 
    stratify=tweet_df['Label'], 
    random_state=43
)

# Padding token sequences for consistent sequence length
X_train_tokens = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=MaxTokenLen, padding='post')
X_valid_tokens = tf.keras.preprocessing.sequence.pad_sequences(X_valid, maxlen=MaxTokenLen, padding='post')
# X_test_tokens = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=MaxTokenLen, padding='post')

def create_attention_masks(token_sequences: np.ndarray) -> np.ndarray:
    """Creates attention masks for token sequences.
    
    Args:
        token_sequences: A NumPy array of token sequences.
        
    Returns:
        A NumPy array representing the attention masks.
    """
    masks = np.array(token_sequences) > 0
    return masks.astype(float)

train_attention_masks = create_attention_masks(X_train_tokens)
valid_attention_masks = create_attention_masks(X_valid_tokens)

# One-hot encoding labels
ohe = preprocessing.OneHotEncoder()
y_train = ohe.fit_transform(y_train.reshape(-1, 1)).toarray()
y_valid = ohe.fit_transform(y_valid.reshape(-1, 1)).toarray()
# y_test = ohe.fit_transform(y_test.reshape(-1, 1)).toarray()

# Delete unused data or variables
del tweet_df, X_train, X_valid

# Manually trigger garbage collection
gc.collect()

################################################## - Modeling - ##################################################
# 
# Self-attention Block; Calculate K, Q, V Matrix; Multihead-attention; Transformers Block
#
################################################## - -------- - ##################################################
#
# embedding_dim and num_heads:
# embedding_dim (the dimension of the model embeddings) should be a multiple of num_heads. This is because in the multi-head self-attention mechanism, each "head" should have its own separate Q, K, V representations. The embedding_dim is evenly divided among num_heads, with each portion being embedding_dim / num_heads in size.
# For example, if you have an embedding_dim of 64, and you use num_heads of 8, then each head's size is 8.
# ff_dim (Feed Forward Network dimension):
#
# ff_dim (dimension of the Feed Forward Network) can be any value, but it is typically larger than embedding_dim. The purpose of this is to increase the model's representational capacity. However, this is not a strict requirement.
# Other parameters:
#
# num_blocks: You can stack multiple Transformer blocks as needed, which is completely independent of other parameters. Increasing the number of blocks can enhance the model's depth and representational ability but may also increase the risk of overfitting.
# noise_stddev: This is a hyperparameter for a noise layer, and it does not need to be proportional to other parameters. It should be adjusted based on the performance on the validation set.
# rate: The dropout rate is also an independent hyperparameter, and adjusting it based on the validation set's performance is a good strategy.
# Additional suggestions:
#
# In your model, you used a look_ahead_mask, but this is more commonly used in decoder scenarios in self-attention modules, such as in machine translation, to ensure that the predicted position can only consider previous positions. For sentence classification tasks, this may not be necessary.
# If you want to add positional encoding to your model, then you need to consider this additionally. Transformers require positional information to understand the order in sequences, which is typically achieved through positional encoding.
# Ensure your training and validation data have undergone appropriate preprocessing, including tokenization, padding, etc.
#
################################################## - -------- - ##################################################
#
# Before embedding, the input dimension is: (batch_size, input_length) (where input_length is the maximum length)
# After embedding, the input dimension becomes: (batch_size, input_length, embedding_dim) (embedding_dim is added)
# vocab_size = tokenizer.vocab_size: This represents the size of the vocabulary, indicating how many unique words or tokens there are. tokenizer.vocab_size retrieves the vocabulary size of the BERT tokenizer.
# E = M * V
# E represents the matrix of embedding results, with shape (input_length, embedding_dim)
# M represents the input sentence/sequence, with shape (input_length,)
# V represents the embedding matrix, with shape (vocab_size, embedding_dim). Each row represents the embedding vector of a word in the vocabulary.
# The weights of embedding_dim can also be updated during the training process.
#
# Additional
# In the Transformer, we have three weight matrices: W_Q, W_K, W_V, used to generate queries (Q), keys (K), and values (V).
# The shapes of these three matrices are typically: (embedding_dim, depth), where depth is the specific dimension set for Q, K, V.
# When input data is multiplied by these three weight matrices, we obtain the Q, K, and V matrices, each with the shape: (batch_size, input_length, depth)
# Thus, when we say Q, K, V are three-dimensional, we mean their shapes are (batch_size, input_length, depth), where depth is the dimensionality of Q, K, V.
#
# Let d_model be the embedding dimension, which is also the original depth of Q, K, V.
# Let num_heads be the number of heads you choose.
# Then, the depth (or dimension) per head is d_k = d_model / num_heads. That is, the depth of Q, K, V for each head is d_k.
# For example, if your embedding dimension d_model is 256, and you chose 8 heads, then the depth d_k for each head is 32.
#
################################################## - -------- - ##################################################

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_size, num_heads):
        """
        Initializes the MultiHeadSelfAttention layer.
        
        Args:
            embed_size (int): The size of the embedding vectors.
            num_heads (int): The number of attention heads.
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads  # Number of attention heads
        self.embed_size = embed_size  # Size of the embedding vectors
        assert embed_size % self.num_heads == 0  # Ensure the embedding size is divisible by the number of heads
        self.projection_dim = embed_size // num_heads  # Dimension of each head
        
        # Dense layers for queries, keys, and values
        self.query_dense = layers.Dense(embed_size)  # Queries
        self.key_dense = layers.Dense(embed_size)    # Keys
        self.value_dense = layers.Dense(embed_size)  # Values
        
        self.combine_heads = layers.Dense(embed_size)  # Layer to combine the heads' outputs


    def get_config(self):
        """
        Returns the configuration of the layer.
        """
        config = super().get_config().copy()
        config.update({
            'num_heads': self.num_heads,
            'embed_size': self.embed_size
        })
        return config

    def attention(self, query, key, value, mask=None):
        """
        Computes the attention scores and outputs.
        
        Args:
            query, key, value: The query, key, and value matrices.
            mask (optional): An optional mask to nullify the effect of padding tokens.
        
        Returns:
            The output after applying attention and the attention weights.
        """
        score = tf.matmul(query, key, transpose_b=True)  # Dot product of queries and keys
        if mask is not None:
            score += (mask * -1e9)  # Apply the mask to scores
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)  # Scale the scores
        weights = tf.nn.softmax(scaled_score, axis=-1)  # Softmax to get attention weights
        output = tf.matmul(weights, value)  # Weighted sum of values
        return output, weights

    def separate_heads(self, x, batch_size):
        """
        Prepares the input matrix x for multi-head attention.
        
        Args:
            x: Input matrix.
            batch_size: Current batch size.
        
        Returns:
            The input matrix reshaped and transposed for multi-head attention.
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None, training=False):
        """
        The forward pass for the layer.
        
        Args:
            inputs: Input tensor.
            mask (optional): An optional mask for padding tokens.
            training (bool): Whether the layer is in training mode.
        
        Returns:
            The output tensor of the MultiHeadSelfAttention layer.
        """
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        # Prepare the inputs for each head
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        
        # Compute the attention
        attention, weights = self.attention(query, key, value, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_size))
        output = self.combine_heads(concat_attention)
        return output

class TransformerBlock(layers.Layer):
    def __init__(self, embed_size, num_heads, ff_dim, rate=0.1):
        """
        Initializes a Transformer block layer consisting of multi-head self-attention and position-wise feed-forward network.
        
        Args:
            embed_size (int): The size of the embedding vectors.
            num_heads (int): The number of attention heads.
            ff_dim (int): The dimension of the feed-forward network.
            rate (float): Dropout rate.
        """
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_size, num_heads)  # The multi-head self-attention layer
        self.ffn = tf.keras.Sequential([  # Position-wise feed-forward network
            layers.Dense(ff_dim, activation="ReLU"),
            layers.Dense(embed_size),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)  # Layer normalization before the addition and after self-attention
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)  # Layer normalization before the addition and after feed-forward network
        self.dropout1 = layers.Dropout(rate)  # Dropout after self-attention
        self.dropout2 = layers.Dropout(rate)  # Dropout after feed-forward network

    def get_config(self):
        """
        Returns the configuration of the layer.
        """
        config = super().get_config().copy()
        config.update({
            'embed_size': self.att.embed_size,
            'num_heads': self.att.num_heads,
            'ff_dim': self.ffn.layers[0].units,
            'rate': self.dropout1.rate
        })
        return config


    def call(self, inputs, mask=None, training=False):
        """
        The forward pass for the Transformer block.
        
        Args:
            inputs: Input tensor.
            mask (optional): An optional mask for padding tokens.
            training (bool): Whether the layer is in training mode.
        
        Returns:
            The output tensor of the Transformer block.
        """
        attn_output = self.att(inputs, mask=mask, training=training)  # Apply multi-head self-attention
        attn_output = self.dropout1(attn_output, training=training)  # Apply dropout
        out1 = self.layernorm1(inputs + attn_output)  # Apply layer normalization
        ffn_output = self.ffn(out1)  # Apply the feed-forward network
        ffn_output = self.dropout2(ffn_output, training=training)  # Apply dropout
        return self.layernorm2(out1 + ffn_output)  # Apply layer normalization and return the output

def create_look_ahead_mask(size: int) -> tf.Tensor:
    """
    Creates a mask to hide future tokens for a sequence.
    
    Args:
        size (int): Size of the sequence.
        
    Returns:
        tf.Tensor: A 2D mask tensor where the positions on and above the diagonal are `True` (masked).
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # Shape: (seq_len, seq_len)

################################################## - Initialization - ##################################################
#
# embedding; prepare different input; transformers layers
#
################################################## - -------------- - ##################################################

# Define hyperparameters for the Transformer model
hyperparameters = {
    "embedding_dim": 8,  # Dimension of the embedding vector
    "num_heads": 2,      # Number of attention heads
    "ff_dim": 512,       # Dimensionality of the feed-forward network
    "num_blocks": 1,     # Number of Transformer blocks
    "dropout_rate": 0.1, # Dropout rate
    "noise_stddev": 0.1  # Standard deviation of Gaussian noise
}

# Parameters derived from hyperparameters for easy access
embedding_dim = hyperparameters["embedding_dim"]
vocab_size = tokenizer.vocab_size  # Vocabulary size from the tokenizer
input_length = MaxTokenLen         # Maximum length of input sequences
num_heads = hyperparameters["num_heads"]
ff_dim = hyperparameters["ff_dim"]
dropout_rate = hyperparameters["dropout_rate"]
num_blocks = hyperparameters["num_blocks"]

# Define inputs for the model: tokens and their corresponding masks
token_inputs = tf.keras.Input(shape=(input_length,), dtype=tf.int32, name='token_inputs')
mask_inputs = tf.keras.Input(shape=(input_length,), dtype=tf.int32, name='mask_inputs')
look_ahead_mask = create_look_ahead_mask(input_length)  # Create a look-ahead mask to prevent attending to future tokens

# Embedding layer to transform token IDs into dense vectors of fixed size
embedding_seq = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=input_length)(token_inputs)

# Apply masking to ignore padding tokens (assumed to be 0) in the sequence
masked_embedding = tf.keras.layers.Masking(mask_value=0)(embedding_seq)

# Build the Transformer blocks
x = masked_embedding
for _ in range(num_blocks):
    transformer_block = TransformerBlock(embedding_dim, num_heads, ff_dim)
    x = transformer_block(x, mask=look_ahead_mask, training=True)  # Apply Transformer block with look-ahead masking

# Pooling layer to reduce the dimensionality of the sequence data
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dropout(dropout_rate)(x)  # Apply dropout for regularization

# A dense layer for further processing
x = tf.keras.layers.Dense(128, activation="ReLU")(x)
x = tf.keras.layers.Dropout(dropout_rate)(x)  # Apply another dropout layer

# Output layer: A dense layer with softmax activation to classify the sentiment into 3 categories
outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

################################################## - Training - ##################################################
#
# compile model; training; save model; Visualize Loss
#
################################################## - -------- - ##################################################

# Compile the model specifying the optimizer, loss function, and metrics to monitor
model = tf.keras.Model(inputs=[token_inputs, mask_inputs], outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the preprocessed dataset
history = model.fit(
    [X_train_tokens, train_attention_masks], 
    y_train, 
    validation_data=([X_valid_tokens, valid_attention_masks], y_valid), 
    epochs=4, 
    batch_size=32
)

# Save the model weights
# save_path = r'.\results\transformers\transformers\path_to_weights.h5'
# model.save_weights(save_path)

# import matplotlib.pyplot as plt
#
# Visualize the training and validation loss and accuracy
# plt.figure(figsize=(12, 6))

# Plot training and validation loss
# plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# # Plot training and validation accuracy
# plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.tight_layout()  # Adjust the layout to make room for all the elements
# plt.show() 

################################################## - Testing - ##################################################
#
# compile model; training; save model
#
################################################## - ------- - ##################################################

# load_path = r'.\results\transformers\transformers\path_to_weights.h5'
# model.load_weights(load_path)
#
# Loading test data
# test_df = pd.read_excel(r'.\data\test.csv')
#
# Process the test data using the tokenizer
# test_df['Token'] = test_df['Text'].apply(lambda txt: tokenizer.encode(txt, max_length=60, truncation=True))
#
# Generate attention masks for the test data
# test_tokens = tf.keras.preprocessing.sequence.pad_sequences(test_df['Token'], maxlen=MaxTokenLen, padding='post')
# test_attention_masks = create_attention_masks(test_tokens)
#
# Make predictions using the model
# predictions = model.predict([test_tokens, test_attention_masks])
#
# Decode the predicted labels to sentiments
# predicted_labels = np.argmax(predictions, axis=1)
#
# 0, 1, and 2 correspond to 'Negative', 'Neutral', and 'Positive', respectively
# sentiments = ['Negative', 'Neutral', 'Positive']
# test_df['PredictedSentiment'] = [sentiments[label] for label in predicted_labels]
#
# Save the predictions to an Excel file
# test_df.to_excel(r'.\results\transformers\PredictedTestData.xlsx', index=False)
