import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('Kannada_Documents_For_Classification.xlsx')
df.head()

df.shape

print(df['LABEL'].value_counts())

def calculate_length(text):
    return len(text.split())    

df['length'] = df['DOCUMENT'].apply(calculate_length)

# Plot each document with respect to its length
plt.figure(figsize=(10, 6))
plt.scatter(range(len(df)), df['length'], alpha=0.5)
plt.title('Document Lengths')
plt.xlabel('Document Index')
plt.ylabel('Length (words)')
plt.show()

#print the dataframe row with the maximum length
print(df.loc[df['length'].idxmax()])

#print the dataframe with the minimum length
print(df.loc[df['length'].idxmin()])

#drop the row with the maximum length
df = df.drop(df['length'].idxmax())
df.shape

# Plot each document with respect to its length
plt.figure(figsize=(10, 6))
plt.scatter(range(len(df)), df['length'], alpha=0.5)
plt.title('Document Lengths')
plt.xlabel('Document Index')
plt.ylabel('Length (words)')
plt.show()

import binascii
import pandas as pd
from Crypto.Cipher import AES, Blowfish, DES3, DES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad
from sklearn.utils import shuffle
import numpy as np
from collections import Counter
from math import log2

# Function to encode Kannada text to bytes
def to_bytes(text):
    return text.encode('utf-8')

# AES encryption
def generate_aes_ciphertext(key, plaintext):
    cipher = AES.new(key, AES.MODE_ECB)
    padded_text = pad(plaintext, AES.block_size)
    ciphertext = cipher.encrypt(padded_text)
    return ciphertext

# DES encryption
def generate_des_ciphertext(key, plaintext):
    cipher = DES.new(key, DES.MODE_ECB)
    padded_text = pad(plaintext, DES.block_size)
    ciphertext = cipher.encrypt(padded_text)
    return ciphertext

# Blowfish encryption
def generate_blowfish_ciphertext(key, plaintext):
    cipher = Blowfish.new(key, Blowfish.MODE_ECB)
    padded_text = pad(plaintext, Blowfish.block_size)
    ciphertext = cipher.encrypt(padded_text)
    return ciphertext

# 3DES encryption
def generate_3des_ciphertext(key, plaintext):
    cipher = DES3.new(key, DES3.MODE_ECB)
    padded_text = pad(plaintext, DES3.block_size)
    ciphertext = cipher.encrypt(padded_text)
    return ciphertext

# Function to compute byte frequency
def compute_byte_frequency(ciphertext):
    byte_counts = Counter(ciphertext)
    total_bytes = len(ciphertext)
    byte_freq = np.zeros(256)
    for byte_val, count in byte_counts.items():
        byte_freq[byte_val] = count / total_bytes  # Normalized frequency
    return byte_freq

# Function to compute entropy of ciphertext
def compute_entropy(ciphertext):
    byte_counts = Counter(ciphertext)
    total_bytes = len(ciphertext)
    entropy = 0
    for count in byte_counts.values():
        prob = count / total_bytes
        entropy -= prob * log2(prob)
    return entropy

# Load your documents (assuming they're in a list of strings)
documents = df['DOCUMENT'].tolist()

# Encryption keys setup
des_key = get_random_bytes(8)  # DES requires 8-byte key
aes_key = get_random_bytes(16)  # AES requires a 16-byte key
blowfish_key = get_random_bytes(16)  # Blowfish key can vary but 16 bytes is common
des3_key = get_random_bytes(24)  # 3DES requires a 24-byte key

# Prepare a list to store the encrypted documents and their features
features_data = []

# Encrypt each document with all four algorithms and extract features
for doc in documents:
    plaintext = to_bytes(doc)
    
    # DES encryption and features
    des_cipher = generate_des_ciphertext(des_key, plaintext)
    des_byte_freq = compute_byte_frequency(des_cipher)
    des_entropy = compute_entropy(des_cipher)
    features_data.append((doc, binascii.hexlify(des_cipher).decode(), des_byte_freq, des_entropy, 'DES'))
    
    # AES encryption and features
    aes_cipher = generate_aes_ciphertext(aes_key, plaintext)
    aes_byte_freq = compute_byte_frequency(aes_cipher)
    aes_entropy = compute_entropy(aes_cipher)
    features_data.append((doc, binascii.hexlify(aes_cipher).decode(), aes_byte_freq, aes_entropy, 'AES'))
    
    # Blowfish encryption and features
    blowfish_cipher = generate_blowfish_ciphertext(blowfish_key, plaintext)
    blowfish_byte_freq = compute_byte_frequency(blowfish_cipher)
    blowfish_entropy = compute_entropy(blowfish_cipher)
    features_data.append((doc, binascii.hexlify(blowfish_cipher).decode(), blowfish_byte_freq, blowfish_entropy, 'Blowfish'))
    
    # 3DES encryption and features
    des3_cipher = generate_3des_ciphertext(des3_key, plaintext)
    des3_byte_freq = compute_byte_frequency(des3_cipher)
    des3_entropy = compute_entropy(des3_cipher)
    features_data.append((doc, binascii.hexlify(des3_cipher).decode(), des3_byte_freq, des3_entropy, '3DES'))

# Convert the features data into a pandas DataFrame
df_features = pd.DataFrame(features_data, columns=['Document', 'Encrypted_Text', 'Byte_Frequency', 'Entropy', 'Algorithm'])

# Shuffle the dataset to mix the encrypted data
df_features = shuffle(df_features).reset_index(drop=True)

# Show the first few rows of the shuffled dataset
df_features.head()

df_features.to_csv("encrypted_kannada_dataset_with_features.csv", index=False)  

print("Feature extraction complete!")

# Load the new data
data = pd.read_csv('encrypted_kannada_dataset_with_features.csv')
data.head()

# Split the features - Byte_Frequency and Entropy
X = data[['Byte_Frequency','Entropy']]
X.shape

y = data['Algorithm']
y.shape

# Encode the target variable
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

#split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print the shapes of the new X objects and save the shapes
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#save the scaled data and the label encoded data
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

# Load the data from numpy arrays
X_train = np.load('X_train.npy',allow_pickle=True)
y_train = np.load('y_train.npy',allow_pickle=True)
X_test = np.load('X_test.npy',allow_pickle=True)
y_test = np.load('y_test.npy',allow_pickle=True)

#print the shape of the data
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

import numpy as np

def convert_to_float(X):
    # Step 1: Clean the string (remove newlines, commas, and extra spaces)
    clean_string = X.replace('\n', ' ').replace('[', '').replace(']', '').replace(',', '').strip()

    # Step 2: Split the string into individual elements
    string_elements = clean_string.split()

    # Step 3: Convert the list of string elements to a list of floats
    float_array = np.array([float(i) for i in string_elements])
    
    return float_array

# Assuming X_train is a list of arrays or strings that need to be converted to floats.
for i in range(len(X_train)):
    # Apply the conversion to the first column of each entry in X_train
    X_train[i][0] = convert_to_float(X_train[i][0])

# Assuming X_test is a list of arrays or strings that need to be converted to floats.
for i in range(len(X_test)):
    # Apply the conversion to the first column of each entry in X_test
    X_test[i][0] = convert_to_float(X_test[i][0])

X_train_flattened = np.array([np.concatenate((x[0], [x[1]])) for x in X_train])
X_test_flattened = np.array([np.concatenate((x[0], [x[1]])) for x in X_test])

print(X_train_flattened.shape)
print(X_test_flattened.shape)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_flattened = sc.fit_transform(X_train_flattened)
X_test_flattened = sc.transform(X_test_flattened)

from sklearn.preprocessing import OneHotEncoder

# OneHotEncode the y values
enc = OneHotEncoder()
y_train_encoded = enc.fit_transform(y_train.reshape(-1,1)).toarray()
y_test_encoded = enc.transform(y_test.reshape(-1,1)).toarray()

#print the shape
print(y_train_encoded.shape)
print(y_test_encoded.shape)



import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam

def create_ann_model(input_shape):
    # Define the input layer
    inputs = Input(shape=(input_shape,))
    
    # Define hidden layer
    x = Dense(64, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)

    # Define the output layer for 4 classes

    outputs = Dense(4, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    #optimizer
    optimizer = Adam(learning_rate=0.00045)
    # Compile the model with categorical crossentropy for multi-class classification
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create the model
input_shape = X_train_flattened.shape[1]  # Use correct feature shape (2 in this case)
model1 = create_ann_model(input_shape)

# Train the model

model1.summary()

history1 = model1.fit(X_train_flattened, y_train_encoded, epochs=10, batch_size=16, validation_split=0.3)


#plot the acuuracies and losses 
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate the model
loss, accuracy = model1.evaluate(X_test_flattened, y_test_encoded)


#plot the confusion matrix
from sklearn.metrics import confusion_matrix

y_pred = model1.predict(X_test_flattened)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test_encoded, axis=1)

cm = confusion_matrix(y_test, y_pred)
#in seaborn heatmap, the x-axis is the predicted labels and the y-axis is the true labels
import seaborn as sns
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

#plot the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


#save the model
model1.save('model1.h5')