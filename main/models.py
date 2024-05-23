from django.db import models

# Create your models here.

class SearchLog(models.Model):
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    CreditScore = models.IntegerField()
    Geography = models.IntegerField()
    Gender = models.IntegerField()
    Age = models.IntegerField()
    Tenure = models.IntegerField()
    Balance = models.FloatField()
    NumOfProductsts = models.IntegerField()
    HasCrCard = models.IntegerField()
    IsActiveMember = models.IntegerField()
    EstimatedSalary = models.FloatField()
    searchTime = models.DateTimeField(auto_now=True)