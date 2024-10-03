from django.shortcuts import render

# Create your views here.
import numpy as np
import pandas as pd
from django.shortcuts import render
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset and model
sonar_data = pd.read_csv('/Users/prajwal/praju/Project/SonarRockOrMinePrediction/sonar_classification/classify/sonarData.csv', header=None)

X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

def predict_sonar(request):
    if request.method == 'POST':
        input_data = request.POST.get('input_data')
        
        # Convert input to numpy array
        input_data = [float(i) for i in input_data.split(',')]
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshape = input_data_as_numpy_array.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data_reshape)[0]
        context = {'prediction': prediction}
        
        return render(request, 'result.html', context)
    
    return render(request, 'predict.html')
