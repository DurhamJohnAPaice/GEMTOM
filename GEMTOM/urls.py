"""django URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path, include
from .views import *


urlpatterns = [
    # path("status_daily/", views.status_daily, name='status_daily'),
    path('targets/import/', TargetImportView.as_view(), name='import'),
    path('', include('tom_common.urls')),
    path('about/', AboutView.as_view(), name='about'),
    path('status/', StatusView.as_view(), name='status'),
    path('transients/', TransientsView.as_view(), name='transients'),
    path('status/<int:obs_date>/', NightView, name='night'),
    path('status_to_GEMTOM/', status_to_GEMTOM, name='status_to_GEMTOM'),
    path('ID_to_GEMTOM/', ID_to_GEMTOM, name='ID_to_GEMTOM'),
    path('handle_input/', handle_input, name='handle_input'),
    # path('dash/', include('django_plotly_dash.urls')),
    # path('status_daily/', status_daily, name='status_daily'),
    path('blackGEM/', BlackGEMView.as_view(), name='blackGEM'),
    path('django_plotly_dash/', include('django_plotly_dash.urls')),
    path('ztf_upload/', UpdateZTFView.as_view(), name='update-ZTF-data'),
    path('blackgem_upload/', UpdateBlackGEMView.as_view(), name='update-BlackGEM-data'),
    # path('upload/', UploadView.as_view(), name='upload'),
]
