# Fruits-Recognition
 This repository contains a project for fruit recognition using the Fruits-360 dataset. The project leverages computer vision techniques, image segmentation, color histogram extraction, and machine learning classifiers to classify fruits into different categories.

## Features

   **Data Preprocessing** : Data is preprocessed by extracting color histograms from fruit images. This includes segmenting fruit from the background using color thresholds and enhancing the segmented images.
   
   **Segmentation**: The images are segmented to isolate the fruit based on predefined color ranges, using the HSV color space. A mask is created to separate the fruit from the white background.
   
   **Image Enhancement**: The segmented images are enhanced using brightness/contrast adjustment, histogram equalization, denoising, and sharpening.
   
   **Feature Extraction**: Color histograms are extracted for each image using the HSV color space to represent the distribution of color components (Hue, Saturation, Value).
   
   **Machine Learning Models** : Multiple machine learning classifiers, including SVM, KNN, and Random Forest, are applied to classify the fruits based on the extracted color histogram features.
   
   **Model Evaluation**: The models are evaluated on their classification performance using accuracy, precision, recall, F1-score, and confusion matrices.
   
## Results

The Random Forest model outperforms the SVM and KNN models in terms of accuracy. The final evaluation on the test set provides an accuracy of approximately **95%**.

## Flask Web Application

After training the machine learning model, we have integrated it into a Flask-based web application to provide real-time fruit recognition. The web app allows users to upload fruit images and get predictions based on the trained model.

### Features of the Web Application:

    Real-Time Prediction: Users can upload an image of a fruit, and the model will predict the type of fruit in real-time.
    
    User-Friendly Interface: The frontend is built using HTML, Bootstrap, and Tailwind CSS to ensure a responsive and easy-to-navigate interface.
    
    Image Preview: After selecting an image for upload, the app displays a preview of the image before making predictions.
    
    Cross-Origin Requests (CORS): CORS is enabled to allow secure communication between the frontend and backend, even from different origins.
    
    Error Handling: The app includes error handling to manage issues such as missing files or failed predictions, ensuring smooth user experience.

### How It Works:

    Uploading the Image: Users can upload an image via a form on the web page.
    
    Image Preprocessing: Once the image is uploaded, it is preprocessed by extracting color histograms and resizing it for model prediction.
    
    Prediction: The processed image is passed to the trained Random Forest model to classify the fruit.
    
    Displaying Results: The model’s prediction (fruit category) is returned and displayed to the user in real-time.

### Technologies Used:

    Backend: Flask for serving the machine learning model and handling HTTP requests.
    
    Frontend: HTML, Bootstrap, and Tailwind CSS for building the user interface.
    
    Machine Learning: The trained Random Forest model, which was built using color histogram features of fruit images.
    
    Python: Python was used for all the machine learning model development, image preprocessing, and backend server functionality.
    
    Google Colab: Google Colab was used for training and evaluating the machine learning models. It provided an easy environment for experimenting with different classifiers and optimizing model performance.

## Web Application
![Screenshot 2024-12-24 193943](https://github.com/user-attachments/assets/f05cfd21-aa9d-4124-8fcf-d11cc14a2b17)
![Screenshot 2024-12-24 193959](https://github.com/user-attachments/assets/3f303dab-6b6d-4816-9b1a-e4f602bae8fe)
![Screenshot 2024-12-24 194024](https://github.com/user-attachments/assets/4dd5dd51-e6da-42ad-94b4-3f6acf94f0e0)
![Screenshot 2024-12-24 194128](https://github.com/user-attachments/assets/b5bec280-102b-4ac9-b711-bec67a351d29)

