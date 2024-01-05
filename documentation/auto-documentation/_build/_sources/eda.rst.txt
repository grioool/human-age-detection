Documentation
=============

This document contains documentation for the Jupyter notebook **exploratory_data_analytic.ipnyb** and the Python file **face_detection.py**.

I. Exploratory Data Analysis
=========================================

The notebook is designed to analyze and derive insights from 'metadata.csv' file and ultimately on analyzing the basic statistics of several .jpg images.

Data Structure
--------------

The data for model training should be placed in the `/train/data` directory, also referred to as `DATA_DIR` in `consts.py`. The structure of the data directory is as follows:

.. code-block:: none

    ├── data
    │   ├── imdb_crop
    │   │   ├── **
    │   │   │  ├── **.jpg
    │   │   ├── imdb.mat
    │   ├── wiki_crop
    │   │   ├── **
    │   │   │  ├── **.jpg
    │   │   ├── wiki.mat

Source:

1. imdb_crop: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar

2. wiki_crop: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar

To use this notebook:

1. Ensure that the data is structured as mentioned in the "Data Structure" section.
2. Run each cell in the notebook sequentially to perform exploratory data analysis.

Most important first steps - preparation for EDA
------------------------------------------------

1. Imports
----------

.. code-block:: python

    import os
    import csv
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import seaborn as sns
    import statsmodels.api as sm
    from scipy import stats
    from PIL import Image
    from tabulate import tabulate

This cell imports various libraries necessary for data analysis and visualization. Key libraries include:

- ``pandas`` and ``numpy`` for data manipulation,
- ``matplotlib`` and ``seaborn`` for data visualization,
- ``statsmodels`` and ``scipy`` for statistical analysis,
- ``PIL`` for image processing,
- ``tabulate`` for presenting tabular data.

2. Setting up Data Path
------------------------

.. code-block:: python

    path_to_metadatacsv = os.path.realpath('../data/metadata.csv')
    print(path_to_metadatacsv)

This cell calculates the real path of the 'metadata.csv' file, which is presumably the main dataset for analysis. The resolved path is printed for verification.

3. Reading CSV File
-------------------

.. code-block:: python

    rows = [] 
    with open(path_to_metadatacsv, 'r') as file:
        read_metadatacsv = csv.reader(file)
        column_names = next(read_metadatacsv)
        for row in read_metadatacsv:
            rows.append(row)
    print("Column names:", column_names)
    print("\nThe first few sample rows:\n")
    for row in rows[:10]:
        print(row)

In this cell, the notebook:

- Opens and reads the CSV file specified in the previous cell.
- Extracts and prints the column names of the dataset.
- Reads the first few rows of the dataset for a preliminary overview.

4. Loading CSV File into DataFrame
-----------------------------------

.. code-block:: python

    df_metadata = pd.read_csv(path_to_metadatacsv)
    df_metadata.head()

This cell loads the CSV file (referenced by `path_to_metadatacsv`) into a Pandas DataFrame named `df_metadata`. It then displays the first few rows of the DataFrame using the `head()` method. This is a common practice in data analysis to get a quick glimpse of the dataset structure and contents.

**General overview of Code Cells**
----------------------------------

The notebook includes various code cells for tasks such as data loading, cleaning, analysis, and visualization. Key code cells include:

1. **Imports** - explained above
2. **Data Path Setup** - explained above
3. **Data Loading** - explained above
4. **Preliminary Analysis**
5. **Exploratory Data Analysis**
6. **Image Analysis**

Preliminary Analysis
--------------------

The notebook provides analysis for subsequent data cleansing approaches, including:

- Collection size analysis.
- Checking whether NaN values are present.
- Analysis of outliers, including values impossible in biology.

Exploratory Data Analysis
-------------------------

This section covers:

- Techniques for uncovering patterns, trends, and relationships in the data.
- Statistical summaries and visualizations for understanding the data

Image Analysis
--------------

This section delves into:

- Loading and analyzing image data.
- Extracting basic statistics like format, size, and dimensions from images.
- Visualizing image data.

Conclusion
----------

**The most important thing** after analyzing this data set is that it contains **values that are impossible** for the ages of men and women. In the world of biology this is impossible. We set the upper possible age limit for a human to be 122 years, the lower one to 1, according to the oldest living person.

Article: https://en.wikipedia.or/wiki/List_of_the_verified_oldest_people


II. Face Detection
=========================================

This file allows you to detect human faces in the image and predicts their age.

To use this code:

1. Need to have OpenCV installed (cv2), 
2. Need to have the pre-trained Haar Cascade model for face detection (haarcascade_frontalface_default.xml), 
3. Need to have compatible age prediction model (`AgePredictionModel`).

See available models in the project:

.. code-block:: none

    ├── main
    │   ├── backend
    │   │   ├── core
    │   │   │  ├── models


For now (04-01-2024): ResNet-18 (Convolutional Neural Network that is 18 layers deep).

1. Imports
----------

.. code-block:: python

    import cv2
    from PIL import Image

This cell imports various libraries necessary for image processing.

- ``OpenCV`` and ``numpy`` for advanced image and video processing,
- ``pillow`` and ``seaborn`` for basic image operations such as resizing and format conversion.

2. Function
-----------

.. code-block:: python

    def detect_faces(img, model):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    faces = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        face_img = img[y:y+h, x:x+w]
        prediction = model.predict(Image.fromarray(face_img).convert('RGB'))
        cv2.putText(img, prediction, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    return {
        'faces': faces
    }

This function takes an input image and performs the following steps:

1. Converts the input image to grayscale for face detection. 
2. Uses a pre-trained Haar Cascade classifier model for face detection. 
3. Detects faces in the grayscale image using the Cascade Classifier. 
4. Draws green rectangles around each detected face on the original image. 
5. Predicts the age of each detected face using the provided age prediction model and displays it on the image. 

Parameters:
----------

**img** (numpy.ndarray): The input image on which face detection and age prediction should be performed. It should be in BGR format as per the standard convention in OpenCV.

**model** (AgePredictionModel): An instance of the age prediction model (selected AgePredictionModel) used for age prediction.

Returns:
-------

**dict**: A dictionary containing the results of face detection. The 'faces' key contains a list of detected faces, where each element in the list is a tuple (x, y, w, h) representing the coordinates and size of a detected face.

- `x` (int): The horizontal coordinate (from left to right) of the top-left corner of the bounding box around the detected face,
- `y` (int): The vertical coordinate (from top to bottom) of the top-left corner of the bounding box around the detected face,
- `w` (int): The width of the bounding box around the detected face,
- `h` (int): The height of the bounding box around the detected face.

Project GitHub
---------------

You can find the source code and more information about the project on GitHub:

[Human Age Detection on GitHub] (https://github.com/grioool/human-age-detection)