from django.db import models

# Create your models here.

class Essay(models.Model):
    """ Essay to be submitted. """
    content = models.TextField(max_length=100000)
    score = models.IntegerField(null=True, blank=True)
    semantic = models.IntegerField(null=True, blank=True)
    tense = models.IntegerField(null=True, blank=True)
    accuracy = models.IntegerField(null=True, blank=True)
    wordcount = models.IntegerField(null=True, blank=True)