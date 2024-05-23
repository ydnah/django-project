from django.test import TestCase
from django.urls import reverse
from .models import SearchLog
import json

# Create your tests here.

class CreateSearchLogTestCase(TestCase):
    
    def test_create_search_log_success(self):
        data = {
            'first_name': 'John',
            'last_name': 'Doe',
            'CreditScore': 700,
            'Geography': 0,
            'Gender': 1,
            'Age': 35,
            'Tenure': 5,
            'Balance': 5000,
            'NumOfProducts': 2,
            'HasCrCard': 1,
            'IsActiveMember': 0,
            'EstimatedSalary': 75000
        }
    
        response = self.client.post(reverse('Score Json'), json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(SearchLog.objects.filter(first_name='John', last_name='Doe').exists())