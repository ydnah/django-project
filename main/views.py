from django.http import JsonResponse
from django.db import transaction
import json
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pickle

from .models import SearchLog
from flower_dict import class_to_flower
from torchvision.models import resnet50, ResNet50_Weights

from django.core.exceptions import ObjectDoesNotExist

# Load models
with open('trained_models/bankchurn.pkl', 'rb') as f:
    model = pickle.load(f)

# Load Image Classification Model
def load_model_ic():
    model_ic = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model_ic.fc = nn.Linear(model_ic.fc.in_features, 1000)
    model_ic.load_state_dict(torch.load('trained_models/oxford102flowers.pth', map_location=torch.device('cpu')))
    model_ic.eval()
    return model_ic

modelIC = load_model_ic()

# Initialize new model with adjusted weights
def initialize_new_model(modelIC):
    new_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    new_model.fc = nn.Linear(new_model.fc.in_features, 102)
    new_model.fc.weight.data = modelIC.fc.weight.data[:102]
    new_model.fc.bias.data = modelIC.fc.bias.data[:102]
    return new_model

new_model = initialize_new_model(modelIC)

# Create search log
def create_search_log(data):
    try:
        with transaction.atomic():
            search_log = SearchLog.objects.create(
                first_name=data.get('first_name'),
                last_name=data.get('last_name'),
                CreditScore=data.get('CreditScore'),
                Geography=data.get('Geography'),
                Gender=data.get('Gender'),
                Age=data.get('Age'),
                Tenure=data.get('Tenure'),
                Balance=data.get('Balance'),
                NumOfProducts=data.get('NumOfProducts'),
                HasCrCard=data.get('HasCrCard'),
                IsActiveMember=data.get('IsActiveMember'),
                EstimatedSalary=data.get('EstimatedSalary')
            )
    except Exception as e:
        print(f'Error: Failed to save data to the database: {str(e)}')

# Process and score JSON data
def score_json(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            create_search_log(data)
            df = pd.DataFrame({'x': data}).transpose().iloc[:, 2:]
            score = model.predict(df)
            return JsonResponse({'score': float(score)})
        except json.decoder.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

# Image classification
def classify_image(request):
    if request.method == 'POST':
        try:
            image_path = request.FILES["image"]
            image = Image.open(image_path)
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                output = modelIC(input_tensor)
            _, predicted_class = output.max(1)
            predicted_class_name = class_to_flower[predicted_class.item()]
            return JsonResponse({'predicted class': predicted_class_name})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
def returnData(request):
    if request.method == 'GET':
        name = request.GET.get('name', None)
        
        if not name:
            return JsonResponse({'error': 'Name parameter is missing'}, status=400)
        
        try:
            data_request = SearchLog.objects.filter(first_name=name).values('first_name', 'last_name', 'searchTime')
            data_request_list = list(data_request)
            return JsonResponse(data_request_list, safe=False)
        
        except ObjectDoesNotExist:
            return JsonResponse({'error': 'No data found for the specified name'}, status=404)
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)