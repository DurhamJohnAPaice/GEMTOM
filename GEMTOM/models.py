from django.db import models
from tom_targets.base_models import BaseTarget


# class UserDefinedTarget(BaseTarget):
#     example_bool = models.BooleanField(default=False, verbose_name='Example Boolean')
#     example_number = models.FloatField(null=True, blank=True, help_text='Pick a number.')
#
#     # Set Hidden Fields
#     example_bool.hidden = True
#
#     class Meta:
#         verbose_name = "target"
#         permissions = (
#             ('view_target', 'View Target'),
#             ('add_target', 'Add Target'),
#             ('change_target', 'Change Target'),
#             ('delete_target', 'Delete Target'),
#         )

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
