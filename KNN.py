import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import os
from skimage.feature import hog
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
import joblib

def convert_folder_to_grayscale(folder_path):
    """
    Converts all images in a folder to grayscale and replaces the original images.

    Args:
        folder_path (str): Path to the folder containing the images.

    Returns:
        None
    """
    try:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_image_path = os.path.join(folder_path, filename)
                image = Image.open(input_image_path)
                grayscale_image = image.convert('L')
                grayscale_image.save(input_image_path)  # Overwrite the original image
                print(f"Converted {filename} to grayscale and replaced the original.")
    except Exception as e:
        print(f"Error converting images: {e}")
def preprocess_images(image_directory, target_size=(400, 400)):
    image_files = os.listdir(image_directory)
    image_data = []
    features = []
    for image_file in image_files:
        # Open the image file.
        with Image.open(os.path.join(image_directory, image_file)) as img:
            # Resize the image to the target size.
            img_resized = img.resize(target_size)
            # Convert the image data to a NumPy array.
            img_array = np.array(img_resized)
            image_data.append(img_array.flatten())
            # Compute HOG features
            hog_features = hog(img_array, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), visualize=False)
            features.append(hog_features)

    return np.array(image_data), np.array(features)
from sklearn.multioutput import MultiOutputClassifier

def train_knn_classification(features, labels, n_neighbors=5,model_path='model.KNN'):
    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train the KNN model
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(x_train, y_train)
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")
    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Calculate the Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f"The Mean Squared Error of the model is {mse:.2f}")

    # Calculate the Root Mean Squared Error
    rmse = np.sqrt(mse)
    print(f"The Root Mean Squared Error of the model is {rmse:.2f}")

    return y_test, y_pred, model
    # Print the accuracy of the model

folder_path = 'posture'  # Replace with the actual folder path
convert_folder_to_grayscale(folder_path)
image_directory='posture'
image_data,features=preprocess_images(image_directory)
df = pd.read_excel('rula.xlsx')
# Save as a CSV file
df.to_csv('labels.csv', index=False)
labels = pd.read_csv('labels.csv')
# Train the model


# Make a prediction on a new image

# Train the decision tree
train_knn_classification(features, labels)
# Calculate the accuracy of the model
print(features.shape,labels.shape)

