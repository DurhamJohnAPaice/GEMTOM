from django import forms
# from uploads.core.models import Document
from django.core.validators import RegexValidator
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from .models import Observation
import astropy.coordinates as coords
from datetime import datetime

# class DocumentForm(forms.ModelForm):
#     class Meta:
#         model = Document
#         fields = ('description', 'document', )

# class ToOForm_Old(forms.Form):
#     name        = forms.CharField(max_length=100, label='Name',         widget=forms.TextInput(attrs={'placeholder': 'Name',           'style': 'width: 300px;', 'class': 'form-control'}))
#     email       = forms.CharField(label='Email',                        widget=forms.TextInput(attrs={'placeholder': 'name@email.com', 'style': 'width: 300px;', 'class': 'form-control'}))
#     date_start  = forms.DateField(label="Start Date",
#             widget=forms.DateInput(format="%Y-%m-%d", attrs={"type": "date", 'style': 'width: 300px;', 'class': 'form-control'}),
#             input_formats=["%Y-%m-%d"])
#     date_close  = forms.DateField(label="End Date",
#             widget=forms.DateInput(format="%Y-%m-%d", attrs={"type": "date", 'style': 'width: 300px;', 'class': 'form-control'}),
#             input_formats=["%Y-%m-%d"])
#     telescope   = forms.CharField(max_length=100, label='Telescope',    widget=forms.TextInput(attrs={'placeholder': 'Telescope',   'style': 'width: 300px;', 'class': 'form-control'}))
#     band        = forms.CharField(max_length=100, label='Band',         widget=forms.TextInput(attrs={'placeholder': 'Band',        'style': 'width: 300px;', 'class': 'form-control'}))
#     notes       = forms.CharField(widget=forms.Textarea(attrs={'cols':'70', 'rows':'1', 'style': 'width: 300px;', 'class': 'form-control'}), label='Notes', required=False)
#     # notes       = forms.CharField(widget=forms.Textarea, label='Notes')

band_choices = [
    ("Radio"         , "Radio"),
    ("Millimetre"    , "Millimetre"),
    ("Microwave"     , "Microwave"),
    ("Infrared"      , "Infrared"),
    ("Optical"       , "Optical"),
    ("Ultraviolet"   , "Ultraviolet"),
    ("X-Ray"         , "X-Ray"),
    ("Gamma"         , "Gamma"),
    ("Other"         , "Other"),
]

list_of_locations = [
    'Other',
    'ATST',
    'Anglo-Australian Observatory',
    'Apache Point Observatory',
    'Atacama Large Millimeter Array',
    'Beijing XingLong Observatory',
    'Big Bear Solar Observatory',
    'Black Moshannon Observatory',
    'CHARA',
    'CHIME',
    'Canada-France-Hawaii Telescope',
    'Catalina Observatory',
    'Cerro Pachon',
    'Cerro Paranal',
    'Cerro Tololo Interamerican Observatory',
    'Cima Ekar Observing Station',
    'Daniel K. Inouye Solar Telescope',
    'Discovery Channel Telescope',
    'Dominion Radio Astrophysical Observatory',
    'GEO',
    'Gemini North',
    'Gemini South',
    'Green Bank Telescope',
    'Hale Telescope',
    'Haleakala Observatories',
    'Happy Jack',
    'Indian Astronomical Observatory',
    'James Clerk Maxwell Telescope',
    'Jansky Very Large Array',
    'John Galt Telescope',
    'Kamioka Gravitational Wave Detector',
    'Keck Observatory',
    'Kitt Peak National Observatory',
    'LIGO Hanford Observatory',
    'LIGO Livingston Observatory',
    'La Silla Observatory',
    'Large Binocular Telescope',
    'Las Campanas Observatory',
    'Lick Observatory',
    'Lowell Observatory',
    'Manastash Ridge Observatory',
    'McDonald Observatory',
    'Medicina',
    'Michigan-Dartmouth-MIT Observatory',
    'Mount Graham International Observatory',
    'Mount Wilson Observatory',
    'Mt. Ekar 182 cm Telescope',
    'Mt. Stromlo Observatory',
    'Multiple Mirror Telescope',
    'Murchison Widefield Array',
    'NASA Infrared Telescope Facility',
    'NST',
    'National Observatory of Venezuela',
    'Noto',
    'Observatoire SIRENE',
    'Observatoire de Haute Provence',
    'Observatorio Astronomico Nacional, San Pedro Martir',
    'Observatorio Astronomico Nacional, Tonantzintla',
    'Owens Valley Radio Observatory',
    'Palomar',
    'Paranal Observatory',
    'Roque de los Muchachos',
    'Royal Observatory Greenwich',
    'SAAO',
    'Sacramento Peak Observatory',
    'Sardinia Radio Telescope',
    'Siding Spring Observatory',
    'Subaru Telescope',
    'Sutherland',
    'TUBITAK National Observatory',
    'The Hale Telescope',
    'United Kingdom Infrared Telescope',
    'VIRGO',
    'Vainu Bappu Observatory',
    'Very Large Array',
    'Whipple Observatory'
]

## Deal with Locations
# list_of_locations = zip(["Other"]+coords.EarthLocation.get_site_names()[:-63], ["Other"]+coords.EarthLocation.get_site_names()[:-63])
list_of_locations = zip(list_of_locations, list_of_locations)
# list_of_locations = {**{("(Other)", "(Other)")}, **list_of_locations}

class ToOForm(forms.Form):
    PI          = forms.CharField(max_length=100, label='PI',         widget=forms.TextInput(attrs={'placeholder': 'Name',           'style': 'width: 300px;', 'class': 'form-control'}))
    # email       = forms.CharField(label='Email',                        widget=forms.TextInput(attrs={'placeholder': 'name@email.com', 'style': 'width: 300px;', 'class': 'form-control'}))
    date_start  = forms.DateField(label="Starting Night",
            widget=forms.DateInput(format="%Y-%m-%d", attrs={"type": "date", 'style': 'width: 300px;', 'class': 'form-control'}),
            input_formats=["%Y-%m-%d"])
    date_close  = forms.DateField(label="Ending Night",
            widget=forms.DateInput(format="%Y-%m-%d", attrs={"type": "date", 'style': 'width: 300px;', 'class': 'form-control'}),
            input_formats=["%Y-%m-%d"])
    telescope   = forms.CharField(max_length=100, label='Telescope',    widget=forms.TextInput(attrs={'placeholder': 'Telescope',   'style': 'width: 300px;', 'class': 'form-control'}))
    location    = forms.ChoiceField(choices=list_of_locations)
    band        = forms.MultipleChoiceField(widget=forms.CheckboxSelectMultiple(), choices=band_choices)
    # band        = forms.CharField(max_length=100, label='Band',         widget=forms.TextInput(attrs={'placeholder': 'Band',        'style': 'width: 300px;', 'class': 'form-control'}))
    notes       = forms.CharField(widget=forms.Textarea(attrs={'cols':'70', 'rows':'1', 'style': 'width: 300px;', 'class': 'form-control'}), label='Notes', required=False)
    # notes       = forms.CharField(widget=forms.Textarea, label='Notes')

    def clean(self):
        cleaned_data = super(ToOForm, self).clean()
        # here all fields have been validated individually,
        # and so cleaned_data is fully populated
        date_start = cleaned_data.get('date_start')
        date_close = cleaned_data.get('date_close')
        # if datetime.strptime(date_start, '%m/%d/%Y %H:%M:%S') > datetime.strptime(date_close, '%m/%d/%Y %H:%M:%S'):
        if date_start > date_close:
            # my_date_time = (my_date + ' ' + my_time + ':00')
            # my_date_time = datetime.strptime(my_date_time, '%m/%d/%Y %H:%M:%S')
            # if datetime.now() <= my_date_time:
            raise ValidationError("End date cannot be before the start date.")
            # msg = u"Start time cannot be after End time"
            # self.add_error('date_start', msg)
            # self.add_error('date_start', msg)
        return cleaned_data


class RegistrationForm(UserCreationForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name', 'password1', 'password2']


class ObservationForm(forms.ModelForm):
    class Meta:
        model = Observation
        fields = ['RA', 'dec', 'notes', 'night']
        widgets = {
            'night': forms.DateInput(attrs={'type': 'date'}),
        }
