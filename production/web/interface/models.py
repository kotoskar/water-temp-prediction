from django.db import models

# Create your models here.
class Prediction(models.Model):
    date = models.DateField(unique=True, db_index=True)
    predicted_temperature = models.FloatField(default=None, null=True, blank=True)
    actual_temperature = models.FloatField(default=None, null=True, blank=True)

    def __str__(self):
        return f'{self.date}: {self.predicted_temperature}, {self.actual_temperature}'
    
    class Meta:
        verbose_name = 'Prediction'
        verbose_name_plural = 'Predictions'