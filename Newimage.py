from decisiontree import convert_folder_to_grayscale,preprocess_images
import joblib

img_path= 'img_new'
convert_folder_to_grayscale(img_path)
features,image_data= preprocess_images(img_path)
# Specify the path to your saved model
model_path = 'model.linear'
# Load the model
model = joblib.load(model_path)
risk_prediction= model.predict(image_data)
print(f'The predicted risk for the image is: {risk_prediction}')
