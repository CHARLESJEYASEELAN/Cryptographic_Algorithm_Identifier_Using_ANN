import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X_train = np.load('X_train.npy',allow_pickle=True)
y_train = np.load('y_train.npy',allow_pickle=True)
X_test = np.load('X_test.npy',allow_pickle=True)
y_test = np.load('y_test.npy',allow_pickle=True)

#check the shape of the data
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

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Fit PCA on the training data
pca = PCA().fit(X_train_flattened)

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(cumulative_variance, marker='o', linestyle='-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA: Cumulative Explained Variance vs. Number of Components')
plt.grid(True)
plt.show()

# Find the number of components that explain 90-95% of the variance
explained_variance_threshold = 0.95
n_components = np.argmax(cumulative_variance >= explained_variance_threshold) + 1

print(f"Number of components to explain {explained_variance_threshold*100}% variance: {n_components}")

# Refit PCA with optimal number of components
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_flattened)
X_test_pca = pca.transform(X_test_flattened)

print(X_train_pca.shape)
print(X_test_pca.shape)


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
input_shape = X_train_pca.shape[1]  # Use correct feature shape (2 in this case)
model1 = create_ann_model(input_shape)

# Train the model

model1.summary()

history1 = model1.fit(X_train_pca, y_train_encoded, epochs=15, batch_size=16, validation_split=0.3)

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
loss, accuracy = model1.evaluate(X_test_pca, y_test_encoded)

#save the model
model1.save('model1_with_PCA.h5')

#plot the confusion matrix
from sklearn.metrics import confusion_matrix

y_pred = model1.predict(X_test_pca)
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

# Convert the history.history dictionary to a DataFrame
history_df = pd.DataFrame(history1.history)

# Save the DataFrame to a CSV file
history_df.to_csv('training_history.csv', index=False)

# You can then use this CSV file for plotting learning curves and validation curves
print("Training history saved to 'training_history_with_PCA.csv'")

#calculate the precisionm, recall and f1-score, specificity and sensitivity

from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Calculate the specificity and sensitivity
def specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[1, 1] / (cm[1, 0] + cm[1, 1])

def sensitivity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[0, 0] / (cm[0, 0] + cm[0, 1])

specificity_score = specificity(y_test, y_pred)
sensitivity_score = sensitivity(y_test, y_pred)

print(f"Specificity: {specificity_score:.4f}")
print(f"Sensitivity: {sensitivity_score:.4f}")

