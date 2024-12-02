from django import forms
# from uploads.core.models import Document
from django.core.validators import RegexValidator
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

# class DocumentForm(forms.ModelForm):
#     class Meta:
#         model = Document
#         fields = ('description', 'document', )

class ToOForm(forms.Form):
    name        = forms.CharField(max_length=100, label='Name',         widget=forms.TextInput(attrs={'placeholder': 'Name',           'style': 'width: 300px;', 'class': 'form-control'}))
    email       = forms.CharField(label='Email',                        widget=forms.TextInput(attrs={'placeholder': 'name@email.com', 'style': 'width: 300px;', 'class': 'form-control'}))
    date_start  = forms.DateField(label="Start Date",
            widget=forms.DateInput(format="%Y-%m-%d", attrs={"type": "date", 'style': 'width: 300px;', 'class': 'form-control'}),
            input_formats=["%Y-%m-%d"])
    date_close  = forms.DateField(label="End Date",
            widget=forms.DateInput(format="%Y-%m-%d", attrs={"type": "date", 'style': 'width: 300px;', 'class': 'form-control'}),
            input_formats=["%Y-%m-%d"])
    telescope   = forms.CharField(max_length=100, label='Telescope',    widget=forms.TextInput(attrs={'placeholder': 'Telescope',   'style': 'width: 300px;', 'class': 'form-control'}))
    band        = forms.CharField(max_length=100, label='Band',         widget=forms.TextInput(attrs={'placeholder': 'Band',        'style': 'width: 300px;', 'class': 'form-control'}))
    notes       = forms.CharField(widget=forms.Textarea(attrs={'cols':'70', 'rows':'1', 'style': 'width: 300px;', 'class': 'form-control'}), label='Notes', required=False)
    # notes       = forms.CharField(widget=forms.Textarea, label='Notes')

class RegistrationForm(UserCreationForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name', 'password1', 'password2']
