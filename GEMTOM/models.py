from django.db import models

class Document(models.Model):
    description = models.CharField(max_length=255, blank=True)
    document = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class Observation(models.Model):
    RA = models.FloatField()
    dec = models.FloatField()
    notes = models.TextField(blank=True)
    night = models.DateField()

    def __str__(self):
        return f"Observation on {self.night} at ({self.ra}, {self.dec})"
