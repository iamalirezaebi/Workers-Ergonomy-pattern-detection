# Workers-Ergonomy-pattern-detection

Project Description
This project aims to analyze and predict the risk of posture-related problems among workers in different companies using images. The dataset comprises 2100 photos of workers in various postures. The main output is a 2D Excel file that provides the minimum and maximum risk of posture issues. The project successfully processes images and makes predictions using a saved machine learning model.

Project Structure
The repository contains the following files:

KNN.py: Implementation of the K-Nearest Neighbors algorithm.
MPL.py: Implementation of a Multi-layer Perceptron algorithm.
Newimage.py: Script for processing and predicting a single image using the saved model.
Nonlinearregression.py: Implementation of a Non-linear Regression algorithm.
README.md: This file, providing an overview of the project.
decisiontree.py: Implementation of a Decision Tree algorithm.
image_data.txt: Text file containing image data information.
labels.csv: CSV file containing labels for the dataset.
linearregression.py: Implementation of a Linear Regression algorithm.
rula.xlsx: Excel file with the final results showing the risk assessment of posture problems.
Installation
To run this project, you'll need to have Python installed along with the necessary libraries. You can install the required libraries using the following command:


pip install -r requirements.txt
(Note: Create a requirements.txt file listing all the dependencies such as scikit-learn, numpy, pandas, etc.)

Usage
Processing Images and Training Models
KNN.py: Run this script to use the K-Nearest Neighbors algorithm for posture risk prediction.
MPL.py: Run this script to use the Multi-layer Perceptron for posture risk prediction.
Nonlinearregression.py: Run this script to use Non-linear Regression for posture risk prediction.
decisiontree.py: Run this script to use the Decision Tree algorithm for posture risk prediction.
linearregression.py: Run this script to use Linear Regression for posture risk prediction.
Predicting with a New Image
Newimage.py: This script processes a new image and predicts the risk using the saved model.

Copy code
python Newimage.py --image_path path_to_your_image
Data Files
image_data.txt: Contains detailed information about the images used in the dataset.
labels.csv: Contains the labels corresponding to each image in the dataset.
rula.xlsx: Excel file with the final posture risk assessment results.
Results
The output is a 2D Excel file (rula.xlsx) that provides the minimum and maximum risk of posture-related problems for workers based on the images provided.

Challenges
The main and most significant challenge was processing the images accurately to ensure reliable predictions. However, the project was successful in providing full predictions on single images using the saved model.

Conclusion
This project demonstrates the application of various machine learning algorithms to predict the risk of posture problems in workers based on image data. The comprehensive approach ensures that the predictions are robust and reliable, offering valuable insights into worker health and safety.

Contributing
Fateme hamzeian from shahid beheshti university
