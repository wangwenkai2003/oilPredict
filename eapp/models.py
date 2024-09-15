from django.db import models

# Create your models here.
class User(models.Model):
    object = models.Manager()
    name = models.CharField(max_length=15, unique=True)
    password = models.CharField(max_length=20)