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
from django.conf.urls.static import static



urlpatterns = [
    # path("status_daily/", views.status_daily, name='status_daily'),
    path('targets/import/', TargetImportView.as_view(), name='import'),
    path('', include('tom_common.urls')),
    # path('GEMTOM/', include('tom_common.urls')),
    # path('about/', AboutView.as_view(), name='about'),

    ## Accounts
    path('accounts/', include('django.contrib.auth.urls')),
    path('accounts/signup/', sign_up, name='signup'),
    path('authentication/', authentication, name='authentication'),


    path('live_feed/', LiveFeed.as_view(), name='live_feed'),
    path('search_BGEM_ID_for_live_feed/', search_BGEM_ID_for_live_feed, name='search_BGEM_ID_for_live_feed'),
    path('live_feed/<int:bgem_id>/', LiveFeed_BGEM_ID_View, name='live_feed_bgem_id'),
    path('update_latest_BlackGEM_Field/', update_latest_BlackGEM_Field, name='update_latest_BlackGEM_Field'),
    path('update_latest_BlackGEM_Field_small/', update_latest_BlackGEM_Field_small, name='update_latest_BlackGEM_Field_small'),
    path('update_time_in_la_silla/', update_time_in_la_silla, name='update_time_in_la_silla'),

    path('update_classification/', update_classification, name='update_classification'),

    # path('recent_transients/', TransientsView.as_view(), name='recent_transients'),
    # path('old_transient/', TransientSearchView.as_view(), name='old_transient'),
    # path('plot/', plot_graph_view, name='plot_graph'),
    path('search_BGEM_ID/', search_BGEM_ID, name='search_BGEM_ID'),
    path('search_TNS_ID/', search_TNS_ID, name='search_TNS_ID'),
    path('search_fuzzy_iauname/', search_fuzzy_iauname, name='search_fuzzy_iauname'),
    path('search_GAIA_ID/', search_GAIA_ID, name='search_GAIA_ID'),
    path('search_BGEM_RA_Dec/', search_BGEM_RA_Dec, name='search_BGEM_RA_Dec'),
    path('search_skytiles_from_RA_Dec_orig/', search_skytiles_from_RA_Dec_orig, name='search_skytiles_from_RA_Dec_orig'),
    path('transients/<int:bgem_id>/', BGEM_ID_View, name='bgem_id'),

    path('transients/', UnifiedTransientsView.as_view(), name='transients'),
    path('transients/orphans/', OrphanedTransientsView.as_view(), name='orphaned_transients'),

    path('history/', HistoryView.as_view(), name='history'),
    path('history/<int:obs_date>/', NightView, name='night'),
    path('history_to_GEMTOM/', history_to_GEMTOM, name='history_to_GEMTOM'),
    path('rate_target/', rate_target, name='rate_target'),
    # path('TNS_to_GEMTOM/', TNS_to_GEMTOM, name='TNS_to_GEMTOM'),
    path('update_history/', manually_update_history, name='update_history'),
    path('url_to_GEMTOM/<int:bgem_id>/', url_to_GEMTOM, name='url_to_GEMTOM'),

    # path('ID_to_GEMTOM/', ID_to_GEMTOM, name='ID_to_GEMTOM'),
    path('handle_input/', handle_input, name='handle_input'),
    # path('dash/', include('django_plotly_dash.urls')),
    # path('status_daily/', status_daily, name='status_daily'),
    path('blackGEM/', BlackGEMView.as_view(), name='blackGEM'),
    path('django_plotly_dash/', include('django_plotly_dash.urls')),
    path('ztf_upload/', UpdateZTFView.as_view(), name='update-ZTF-data'),
    path('blackgem_upload/', UpdateBlackGEMView.as_view(), name='update-BlackGEM-data'),

    path('comingsoon/', ComingSoonView.as_view(), name='comingsoon'),

    path('ToOs/', ToOView.as_view(), name='ToOs'),
    path('delete_telescopetime/', delete_telescopetime, name='delete_telescopetime'),

    path('download_lightcurve/', download_lightcurve, name='download_lightcurve'),
    path('download_possible_CVs/', download_possible_CVs, name='download_possible_CVs'),

    # path('Observations/', ObservationNightView.as_view(), name='Observations'),
    # path('submit_observation/', submit_observation, name='submit_observation'),
    # path('Observations/', observation_view.as_view(), name='Observations'),
    path('set_observed/', set_observed, name='set_observed'),
    path('delete_observation/', delete_observation, name='delete_observation'),
    path('Observations/', submit_observation, name='Observations'),

    # path('ToOs/', too_view, name='ToOs'),
    # path('upload/', UploadView.as_view(), name='upload'),
]

# from django.contrib.staticfiles.urls import staticfiles_urlpatterns
# urlpatterns += patterns('', (
#     r'^GEMTOM/static/(?P<path>.*)$',
#     'django.views.static.serve',
#     {'document_root': settings.STATIC_ROOT}
# ))

# urlpatterns = [path(r'GEMTOM/', include(urlpatterns))]


# if settings.DEBUG:
#     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
