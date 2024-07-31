# from .forms import UploadFileForm
# from django.core.files.uploadedfile import SimpleUploadedFile

import os
import sys
import pandas as pd
from astropy.coordinates import SkyCoord, Galactocentric
from astropy import units as u
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback, dash_table
from io import StringIO

from django.views.generic import TemplateView, FormView
from django.http import HttpResponse, HttpResponseRedirect, FileResponse
from django.urls import reverse
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from django.utils.safestring import mark_safe
from django.core.files.storage import FileSystemStorage
from guardian.shortcuts import assign_perm, get_objects_for_user

## For importing targets
from .BlackGEM_to_GEMTOM import *
from tom_targets.utils import import_targets

from . import plotly_test
from . import plotly_app
from ztfquery import lightcurve

## For the Status View
import requests
from django.template import loader
from astropy.time import Time
from bs4 import BeautifulSoup
from datetime import datetime, date, timedelta
import numpy as np

## For the Recent Transients View
from django_plotly_dash import DjangoDash
import dash_ag_grid as dag
import json

## For the Transient View
from pathlib import Path
import plotly.graph_objs as go
from plotly.offline import plot

## BlackGEM Stuff
from blackpy import BlackGEM
from blackpy.catalogs.blackgem import TransientsCatalog

from tom_common.hooks import run_hook
from tom_observations.models import Target
from tom_common.hints import add_hint
from tom_dataproducts.exceptions import InvalidFileFormatException

## Data Products
from processors.ztf_processor import ZTFProcessor
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic.base import RedirectView
from data_processor import run_data_processor
from tom_dataproducts.models import DataProduct, DataProductGroup, ReducedDatum
from tom_dataproducts.forms import DataProductUploadForm


class TargetImportView(LoginRequiredMixin, TemplateView):
    """
    View that handles the import of targets from a CSV. Requires authentication.
    """
    template_name = 'tom_targets/target_import.html'

    def post(self, request):
        """
        Handles the POST requests to this view. Creates a StringIO object and passes it to ``import_targets``.

        NEW - can also handle BlackGEM-specific targets!

        :param request: the request object passed to this view
        :type request: HTTPRequest
        """

        ## Are we uploading in the BlackGEM format, or the TOM-specific format?
        ## If BlackGEM, we need to process the targets first, so...
        if 'process_targets' in request.POST:

            ## Load them in and interpret them as a pandas dataframe
            csv_file = request.FILES['file']
            csv_stream = StringIO(csv_file.read().decode('utf-8'), newline=None)
            data = pd.read_csv(csv_stream)

            ## Process them using the BlackGEM_to_GEMTOM code
            processed_file = GEM_to_TOM(data, request)
            processed_file.to_csv("./Data/processed_file.csv", index=False)

            ## Output them into a StringIO format for the import_targets function
            csv_stream = StringIO(open(os.getcwd()+"/Data/processed_file.csv", "rb").read().decode('utf-8'), newline=None)

        ## If they're TOM-specific...
        if 'import_targets' in request.POST:
            csv_file = request.FILES['target_csv']
            csv_stream = StringIO(csv_file.read().decode('utf-8'), newline=None)

        ## And finally, read them in!
        result = import_targets(csv_stream)
        for target in result['targets']:
            target.give_user_access(request.user)
        messages.success(
            request,
            'Targets created: {}'.format(len(result['targets']))
        )

        # ## Lastly, get the BlackGEM lightcurve...
        # print(data)
        # print(request)

        for error in result['errors']:
            messages.warning(request, error)
        return redirect(reverse('tom_targets:list'))


## =============================================================================
## ---------------------------- General Functions ------------------------------

def list_files(url):
    '''
    Gets a list of files from an online directory.
    '''

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all links on the page
        links = soup.find_all('a')

        # Extract file names from the links
        files = [link.get('href') for link in links if link.get('href')]

        # Filter out the parent directory link (if present) and directories
        files = [file for file in files if not file.endswith('/') and file != '../']

        return files
    else:
        print(f"Failed to access {url}")
        return []


def get_lightcurve(source_id):
    '''
    Fetches the lightcurve from Hugo's server. Returns blank if no lightcurve exists.
    '''
    url = "http://xmm-ssc.irap.omp.eu/claxson/BG_images/lcrequests/" + source_id + "_lc.jpg"
    r = requests.get(url)
    if r.status_code != 404:
        return url
    else:
        return ""


def add_to_GEMTOM(id, name, ra, dec):

    # get_lightcurve(id)

    gemtom_dataframe = pd.DataFrame({
        'name' : [name],
        'ra' : [ra],
        'dec' : [dec],
        'BlackGEM ID' : [int(id)],
        'type' : ['SIDEREAL'],
        'public' : ['Public']
    })

    gemtom_dataframe = gemtom_dataframe.reindex(gemtom_dataframe.index)

    gemtom_dataframe.to_csv("./Data/processed_file.csv", index=False)
    csv_stream = StringIO(open(os.getcwd()+"/Data/processed_file.csv", "rb").read().decode('utf-8'), newline=None)

    ## And finally, read them in!
    result = import_targets(csv_stream)

    return redirect(reverse('tom_targets:list'))


def plot_BGEM_lightcurve(df_bgem_lightcurve, df_limiting_mag):

    ## Lightcurve
    filters = ['u', 'g', 'q', 'r', 'i', 'z']
    colors = ['darkviolet', 'forestgreen', 'darkorange', 'orangered', 'crimson', 'dimgrey']
    symbols = ['triangle-up', 'diamond-wide', 'circle', 'diamond-tall', 'pentagon', 'star']

    fig = go.Figure()
    # print(df_bgem_lightcurve['x.mag_zogy'])
    # print(df_bgem_lightcurve['x.magerr_zogy'])
    print(df_bgem_lightcurve.columns)
    print(df_limiting_mag.columns)

    for f in filters:
        df_2 = df_bgem_lightcurve.loc[df_bgem_lightcurve['i.filter'] == f]
        df_limiting_mag_2 = df_limiting_mag.loc[df_limiting_mag['filter'] == f]

        print(f+":")
        print(df_2['x.mag_zogy'])
        print(df_2['x.magerr_zogy'])

        fig.add_trace(go.Scatter(
                    x               = df_2['i."mjd-obs"'],
                    y               = df_2['x.mag_zogy'],
                    error_y         = dict(
                                        type='data',
                                        array = df_2['x.magerr_zogy'],
                                        thickness=1,
                                        width=3,
                                      ),
                    mode            = 'markers',
                    marker_color    = colors[filters.index(f)],
                    name            = filters[filters.index(f)],
                    hovertemplate   =
                        'MJD: %{x:.3f}<br>' +
                        'Mag: %{y:.3f} ± %{customdata[0]:.3f}<br>' +
                        'Flux: %{customdata[1]:.3f} ± %{customdata[2]:.3f} µJy',

                    customdata = [(df_2['x.magerr_zogy'].iloc[i], df_2['x.flux_zogy'].iloc[i], df_2['x.fluxerr_zogy'].iloc[i]) for i in range(len(df_2['x.fluxerr_zogy']))]
        ))
        fig.add_trace(go.Scatter(
                    x               = df_limiting_mag_2['mjd'],
                    y               = df_limiting_mag_2['limiting_mag'],
                    mode            = 'markers',
                    marker          = dict(symbol='arrow-wide', angle=180, size=12),
                    marker_color    = colors[filters.index(f)],
                    opacity         = 0.2,
                    name            = filters[filters.index(f)],
                    hovertemplate   =
                        '<i>MJD: %{x:.3f}</i><br>' +
                        '<i>Limit: %{y:.3f}</i>',
                    hoverlabel      = dict(bgcolor="white")

        ))

    fig.update_layout(height=600)
    fig.update_layout(hovermode="x", xaxis=dict(tickformat ='d'),
    # fig.update_layout(xaxis=dict(tickformat ='d'),
        title="Lightcurves",
        xaxis_title="MJD",
        yaxis_title="Magnitude",)
    fig.update_yaxes(autorange="reversed")

    return fig


def plot_BGEM_location_on_sky(df_bgem_lightcurve):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_bgem_lightcurve['x.ra_psf_d'], y=df_bgem_lightcurve['x.dec_psf_d'], mode='markers', name='Line and Marker'))
    fig.update_layout(width=400, height=400)

    return fig


class BlackGEMView(TemplateView):
    template_name = 'blackGEM.html'

    def get_context_data(self, **kwargs):
        return {'targets': Target.objects.all()}





## =============================================================================
## ------------------------ Codes for the Target pages -------------------------

# def update_classification(request, target, field_name="Classification", value="SN"):
#     print(Target)
#     target = Target.objects.get(name='target_name')
#     target.save(extras={field_name: value})

def update_target_field(target_name, field_name, field_value):

    target = Target.objects.get(name=target_name)
    target.save(extras={field_name: field_value})


# def update_classification(request):
#     '''
#     Updates the classification of a target
#     '''
#
#     target_id   = request.GET.get('id')
#     target_name = request.GET.get('name')
#     field_name  = "Classification"
#     field_value = request.GET.get('field_value')
#
#     update_target_field(target_name, field_name, field_value)
#
#     return redirect(f'/targets/{target_id}')

def update_classification(request):
    '''
    Updates the classification of a target
    '''

    print("\n\n\n\n")
    print(request.POST)
    print(request.POST['id'])
    print("\n\n\n\n")

    target_id   = request.POST['id']
    target_name = request.POST['name']
    field_name  = "Classification"
    field_value = request.POST['dropdown']

    update_target_field(target_name, field_name, field_value)

    return redirect(f'/targets/{target_id}')



# class update_classification(LoginRequiredMixin, RedirectView):
#     """
#     View that handles the updating of BlackGEM data. Requires authentication.
#     """
#
#     def get(self, request, *args, **kwargs):
#         """
#         Method that handles the GET requests for this view. Calls the management command to update the reduced data and
#         adds a hint using the messages framework about automation.
#         """
#         target_id = request.POST.get('id')
#         target_name = request.POST.get('name')
#         field_name = "Classification"
#         field_value = "SN"
#
#         update_target_field(target_name, field_name, field_value)
#
#         return redirect(form.get('referrer', '/'))



## =============================================================================
## ------------------------ Codes for the Status pages -------------------------



def get_recent_blackgem_history(days_since_last_update):

    obs_date = date.today() - timedelta(1)
    extended_obs_date = obs_date.strftime("%Y-%m-%d")
    obs_date = obs_date.strftime("%Y%m%d")
    mjd = int(Time(extended_obs_date + "T00:00:00.00", scale='utc').mjd)

    ## Get previous history:
    previous_history = blackgem_history()

    dates           = []
    mjds            = []
    observed        = []
    transients      = []
    gaia            = []
    extagalactic    = []

    if days_since_last_update > 10:
        days_since_last_update = 10

    for this_mjd in np.arange(mjd,mjd-days_since_last_update,-1):

        time_mjd        = Time(this_mjd, format='mjd')
        time_isot       = time_mjd.isot

        extended_date   = time_isot[0:10]
        obs_date      = extended_date[0:4] + extended_date[5:7] + extended_date[8:10]

        dates.append(extended_date)
        mjds.append(this_mjd)

        base_url = 'http://xmm-ssc.irap.omp.eu/claxson/BG_images/'

        ## Get the list of files from Hugo's server
        files = list_files(base_url + obs_date)

        ## Check to find the transients file. It could be under one of two names...
        if extended_date+"_gw_BlackGEM_transients.csv" in files:
            transients_filename = base_url + obs_date + "/" + extended_date + "_gw_BlackGEM_transients.csv"
            data = pd.read_csv(transients_filename)
        elif extended_date+"_BlackGEM_transients.csv" in files:
            transients_filename = base_url + obs_date + "/" + extended_date + "_BlackGEM_transients.csv"
            data = pd.read_csv(transients_filename)
        else:
            ## If it doesn't exist, assume BlackGEM didn't observe any transients that night.
            # return "0", "0", "0", ["","","","", ""], "", "", "", ""
            observed.append("No")
            transients.append("0")
            gaia.append("0")
            extagalactic.append("0")
            continue

        ## Check to find the gaia crossmatched file. It could be under one of two names...
        if extended_date+"_gw_BlackGEM_transients_gaia.csv" in files:
            gaia_filename = base_url + obs_date + "/"+extended_date+"_gw_BlackGEM_transients_gaia.csv"
            data_gaia = pd.read_csv(gaia_filename)
            num_in_gaia = str(len(data_gaia))
        elif extended_date+"_BlackGEM_transients_gaia.csv" in files:
            gaia_filename = base_url + obs_date + "/"+extended_date+"_BlackGEM_transients_gaia.csv"
            data_gaia = pd.read_csv(gaia_filename)
            num_in_gaia = str(len(data_gaia))
        else:
            ## If it doesn't exist, assume no gaia crossmatches were found.
            gaia_filename = ""
            num_in_gaia = "0"

        ## Check to find the extragalactic file. It could be under one of two names...
        if extended_date+"_gw_BlackGEM_transients_selected.csv" in files:
            extragalactic_filename = base_url + obs_date + "/"+extended_date+"_gw_BlackGEM_transients_selected.csv"
            extragalactic_data = pd.read_csv(extragalactic_filename)
        elif extended_date+"_BlackGEM_transients_selected.csv" in files:
            extragalactic_filename = base_url + obs_date + "/"+extended_date+"_BlackGEM_transients_selected.csv"
            extragalactic_data = pd.read_csv(extragalactic_filename)
        else:
            ## If it doesn't exist, assume no extragalactic sources were found.
            extragalactic_filename = ""

        extragalactic_sources_id = []
        ## For each image file...
        for file in files:
            if ".png" in file:
                ## If we haven't got the data yet...
                if file[2:10] not in extragalactic_sources_id:
                    ## Save the ID...
                    extragalactic_sources_id.append(file[2:10])

        print(extended_date + " (MJD " + str(this_mjd) + "): BlackGEM found " + str(len(data)) + " transients (" + str(len(data_gaia)) + " in Gaia, " + str(len(extragalactic_sources_id)) + " extragalactic).")
        observed.append("Yes")
        transients.append(len(data))
        gaia.append(len(data_gaia))
        extagalactic.append(len(extragalactic_sources_id))

    fileOut = "./Data/Recent_BlackGEM_History.csv"
    new_history = pd.DataFrame({'Date' : dates, 'MJD' : mjds, 'Observed' : observed, 'Number_Of_Transients' : transients, 'Number_of_Gaia_Crossmatches' : gaia, 'Number_Of_Extragalactic' : extagalactic})

    output = pd.concat([new_history,previous_history.iloc[:(10-days_since_last_update)]]).reset_index(drop=True)
    output.to_csv(fileOut, index=False)


def blackgem_history():
    '''
    Fetches BlackGEM's history and returns as a pandas dataframe
    '''

    # get_recent_blackgem_history()

    # history = pd.read_csv("./data/BlackGEM_History.csv")
    history = pd.read_csv("./data/Recent_BlackGEM_History.csv")
    # print(history)

    return history

def manually_update_history(request):
    # Call your function with the hidden value
    update_history(10)
    # Redirect to the desired page (e.g., 'home' view)
    return redirect('status')

def update_history(days_since_last_update):
    '''
    Fetches BlackGEM's history and returns as several lists, in order to make a table
    '''

    get_recent_blackgem_history(days_since_last_update)

    history = pd.read_csv("./data/Recent_BlackGEM_History.csv")
    # print(history)

    return redirect('status')  # Redirect to the original view if no input



class StatusView(TemplateView):
    template_name = 'status.html'

    def post(self, request, **kwargs):
        # date = '20240424'
        obs_date  = request.POST['obs_date']

        extended_date = obs_date[:4] + "-" + obs_date[4:6] + "-" + obs_date[6:]

        try:
            mjd = int(Time(extended_date + "T00:00:00.00", scale='utc').mjd)
        except:
            raise RuntimeError("Inputted date is not valid!")

        print("")
        print("Hello! Welcome to the BlackGEM Transient fetcher.")
        print("Looking for data from ", extended_date, "...", sep="")
        print("")

        try:
            data_length, num_in_gaia, extragalactic_sources_length, extragalactic_sources, images_urls_sorted, \
                transients_filename, gaia_filename, extragalactic_filename = get_blackgem_stats(obs_date)


            if data_length == "1": data_length_plural = ""; data_length_plural_2 = "s"
            else: data_length_plural = "s"; data_length_plural_2 = "ve"
            if extragalactic_sources_length == "1": extragalactic_sources_plural = ""
            else: extragalactic_sources_plural = "s"

            extragalactic_sources_string = ""
            lightcurve_urls = []
            for source in extragalactic_sources[0]:
                extragalactic_sources_string += source + ", "


            images_urls_string = ""
            for this_source in images_urls_sorted:
                for image in this_source:
                    images_urls_string += "<a href=\"" + image + "\">" + image + "</a><br>"

            return HttpResponse("On " + extended_date + " (MJD " + str(mjd) + "), BlackGEM observed " + data_length + " transient" + data_length_plural + ", which ha" + data_length_plural_2 + " " + num_in_gaia + " crossmatches in Gaia (radius 1 arcsec). <br>" +
             "BlackGEM recorded pictures of the following " + extragalactic_sources_length + " possible extragalactic transient" + extragalactic_sources_plural + ": <br> " +
             extragalactic_sources_string + "<br>" +
             images_urls_string)


        except Exception as e:
            if '404' in str(e):
                print("BlackGEM did not observe on " + extended_date + " (MJD " + str(mjd) + ").")
                return HttpResponse("BlackGEM did not observe on " + extended_date + " (MJD " + str(mjd) + ").")

            else:
                return HttpResponse(e)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['status_daily_text_1'], \
            context['status_daily_text_2'], \
            context['status_daily_text_3'], \
            context['status_daily_text_4'], \
            context['images_daily_text_1'], \
            context['extragalactic_sources_id'], \
            context['transients_filename'], \
            context['gaia_filename'], \
            context['extragalactic_filename'] = status_daily()
        history = blackgem_history()
        dates = list(history.Date)
        dates = [this_date[0:4] + this_date[5:7] + this_date[8:] for this_date in dates]
        # print(dates[0])

        ## --- Check to see if the recent history is up to date. If not, update.
        ## Get yesterday's date...
        yesterday_date = datetime.today() - timedelta(1)
        yesterday_date = yesterday_date.replace(hour=0, minute=0, second=0, microsecond=0)

        ## Get the most recent date...
        most_recent_date = datetime.strptime(dates[0], "%Y%m%d")

        ## Find the difference...
        difference = yesterday_date - most_recent_date
        days_since_last_update = difference.days
        print("Days since last update:", days_since_last_update)

        yesterday_date_string = yesterday_date.strftime("%Y%m%d")

        if days_since_last_update > 0:
            update_history(days_since_last_update)
            history = blackgem_history()
            dates = list(history.Date)
            dates = [this_date[0:4] + this_date[5:7] + this_date[8:] for this_date in dates]

        context['history'] = zip(dates, list(history.Date), list(history.MJD), list(history.Observed), list(history.Number_Of_Transients), list(history.Number_of_Gaia_Crossmatches), list(history.Number_Of_Extragalactic))
        # print(context['history'])
        # context['images_daily_text_1'], \
        #     context['images_daily_text_2']  = images_daily()
        return context


def handle_input(request):
    '''
    Redirects to a status page about a certain date.
    '''
    user_input = request.GET.get('user_input')
    print(user_input)
    if user_input:
        return redirect(f'/status/{user_input}')
    return redirect('status')  # Redirect to the original view if no input


def search_BGEM_ID(request):
    '''
    Redirects to a status page about a certain date.
    '''
    user_input = request.GET.get('user_input')
    print(user_input)
    if user_input:
        return redirect(f'/transient/{user_input}')
    return redirect('transient')  # Redirect to the original view if no input



def get_blackgem_stats(obs_date):
    '''
    Gets details about a night's observations from its observation date.
    Querys Hugo's server.
    '''

    ## Get date in different formats
    extended_date = obs_date[:4] + "-" + obs_date[4:6] + "-" + obs_date[6:]
    mjd = int(Time(extended_date + "T00:00:00.00", scale='utc').mjd)

    base_url = 'http://xmm-ssc.irap.omp.eu/claxson/BG_images/'

    ## Get the list of files from Hugo's server
    files = list_files(base_url + obs_date)

    ## Check to find the transients file. It could be under one of two names...
    if extended_date+"_gw_BlackGEM_transients.csv" in files:
        transients_filename = base_url + obs_date + "/" + extended_date + "_gw_BlackGEM_transients.csv"
        data = pd.read_csv(transients_filename)
    elif extended_date+"_BlackGEM_transients.csv" in files:
        transients_filename = base_url + obs_date + "/" + extended_date + "_BlackGEM_transients.csv"
        data = pd.read_csv(transients_filename)
    else:
        ## If it doesn't exist, assume BlackGEM didn't observe any transients that night.
        return "0", "0", "0", ["","","","", ""], "", "", "", ""

    ## Check to find the gaia crossmatched file. It could be under one of two names...
    if extended_date+"_gw_BlackGEM_transients_gaia.csv" in files:
        gaia_filename = base_url + obs_date + "/"+extended_date+"_gw_BlackGEM_transients_gaia.csv"
        data_gaia = pd.read_csv(gaia_filename)
        num_in_gaia = str(len(data_gaia))
    elif extended_date+"_BlackGEM_transients_gaia.csv" in files:
        gaia_filename = base_url + obs_date + "/"+extended_date+"_BlackGEM_transients_gaia.csv"
        data_gaia = pd.read_csv(gaia_filename)
        num_in_gaia = str(len(data_gaia))
    else:
        ## If it doesn't exist, assume no gaia crossmatches were found.
        gaia_filename = ""
        num_in_gaia = "0"

    ## Check to find the extragalactic file. It could be under one of two names...
    if extended_date+"_gw_BlackGEM_transients_selected.csv" in files:
        extragalactic_filename = base_url + obs_date + "/"+extended_date+"_gw_BlackGEM_transients_selected.csv"
        extragalactic_data = pd.read_csv(extragalactic_filename)
    elif extended_date+"_BlackGEM_transients_selected.csv" in files:
        extragalactic_filename = base_url + obs_date + "/"+extended_date+"_BlackGEM_transients_selected.csv"
        extragalactic_data = pd.read_csv(extragalactic_filename)
    else:
        ## If it doesn't exist, assume no extragalactic sources were found.
        extragalactic_filename = ""

    ## --- Find the details of each extragalactic source ---
    images_urls                 = []
    extragalactic_sources       = []
    extragalactic_sources_id    = []
    extragalactic_sources_name  = []
    extragalactic_sources_ra    = []
    extragalactic_sources_dec   = []
    extragalactic_sources_check = []

    ## For each image file...
    for file in files:
        if ".png" in file:
            ## Save the URL...
            images_urls.append(base_url + obs_date + "/" + file[2:])

            ## If we haven't got the data yet...
            if file[2:10] not in extragalactic_sources_id:

                ## Save the ID...
                extragalactic_sources_id.append(file[2:10])

                ## And if there's extragalactic data...
                if extragalactic_filename:
                    runcat_id_list = list(extragalactic_data['runcat_id'])

                    ## And this source is in that data...
                    if int(file[2:10]) in runcat_id_list:
                        ## Save the name, RA, dec, and look for a lightcurve.
                        row_number = runcat_id_list.index(int(file[2:10]))
                        extragalactic_sources_name.append(extragalactic_data['iauname'][row_number])
                        extragalactic_sources_ra.append(  extragalactic_data['ra'][row_number])
                        extragalactic_sources_dec.append( extragalactic_data['dec'][row_number])
                        extragalactic_sources_check.append(True)
                        # extragalactic_sources_jpg.append(get_lightcurve(file[2:10]))
                    else:
                        ## If it's not, state they're all unknown.
                        extragalactic_sources_name.append("Unknown")
                        extragalactic_sources_ra.append("(Unknown)")
                        extragalactic_sources_dec.append("(Unknown)")
                        extragalactic_sources_check.append(False)
                        # extragalactic_sources_jpg.append("")

    ## Combine these together.
    # extragalactic_sources = [extragalactic_sources_id, extragalactic_sources_name, extragalactic_sources_ra, extragalactic_sources_dec, extragalactic_sources_jpg]
    extragalactic_sources = [extragalactic_sources_id, extragalactic_sources_name, extragalactic_sources_ra, extragalactic_sources_dec, extragalactic_sources_check]
    # print(extragalactic_sources)
    # print(extragalactic_sources_name)
    # print(extragalactic_sources_ra)
    # print(extragalactic_sources_dec)


    ## Sort the images into a list, separated into each source
    images_urls_sorted = []
    for this_source in extragalactic_sources[0]:
        matching = [url for url in images_urls if this_source in url]
        images_urls_sorted.append(matching)

    num_new_transients  = str(len(data))
    num_extragalactic   = str(len(extragalactic_sources[0]))
    extragalactic_urls  = images_urls_sorted

    return num_new_transients, num_in_gaia, num_extragalactic, extragalactic_sources, extragalactic_urls, \
        transients_filename, gaia_filename, extragalactic_filename


## Function for checking last night's BlackGEM status.
def status_daily():
    '''
    Specifically checks last night's observation.
    '''

    yesterday = date.today() - timedelta(1)
    yesterday_date = yesterday.strftime("%Y%m%d")
    extended_yesterday_date = yesterday.strftime("%Y-%m-%d")
    mjd = int(Time(extended_yesterday_date + "T00:00:00.00", scale='utc').mjd)

    url = 'http://xmm-ssc.irap.omp.eu/claxson/BG_images/' + yesterday_date + "/"

    # print(url)
    r = requests.get(url)
    if r.status_code != 404:
        result = "BlackGEM observed last night!"
        status_daily_text_1 = "Yes!"

        data_length, num_in_gaia, extragalactic_sources_length, extragalactic_sources, images_urls_sorted, \
            transients_filename, gaia_filename, extragalactic_filename = get_blackgem_stats(yesterday_date)

        ## If there was no data, assume BlackGEM didn't observe.
        if (data_length == "0") and (num_in_gaia == "0") and (extragalactic_sources_length == "0"):
            extragalactic_sources_id    = ""
            status_daily_text_1         = "BlackGEM did not observe last night (" + extended_yesterday_date + ")"
            status_daily_text_2         = ""
            status_daily_text_3         = ""
            status_daily_text_4         = ""
            images_daily_text_1         = zip([], ["BlackGEM did not observe last night."])

        else:
            if data_length == "1": data_length_plural = ""; data_length_plural_2 = "s"
            else: data_length_plural = "s"; data_length_plural_2 = "ve"
            if extragalactic_sources_length == "1": extragalactic_sources_plural = ""
            else: extragalactic_sources_plural = "s"

            extragalactic_sources_string = ""
            for source in extragalactic_sources[0]:
                extragalactic_sources_string += source + ", "

            extragalactic_sources_id = extragalactic_sources[0]
            status_daily_text_2 = "On " + extended_yesterday_date + " (MJD " + str(mjd) + "), BlackGEM observed " + data_length + " transient" + data_length_plural + ", which ha" + data_length_plural_2 + " " + num_in_gaia + " crossmatches in Gaia (radius 1 arcsec)."
            status_daily_text_3 = "BlackGEM recorded pictures of " + extragalactic_sources_length + " possible extragalactic transient" + extragalactic_sources_plural + "."
            status_daily_text_4 = extragalactic_sources_string
            # images_daily_text_1 = zip(images_urls_sorted, extragalactic_sources[0])
            images_daily_text_1 = zip(images_urls_sorted, extragalactic_sources[0], extragalactic_sources[1], extragalactic_sources[2], extragalactic_sources[3], extragalactic_sources[4])
            # images_daily_text_1 = zip(images_urls_sorted, extragalactic_sources[0], extragalactic_sources[1], extragalactic_sources[2], extragalactic_sources[3])


    else:
        extragalactic_sources_id    = ""
        transients_filename         = ""
        gaia_filename               = ""
        extragalactic_filename      = ""
        status_daily_text_1         = "BlackGEM did not observe last night (" + extended_yesterday_date + ")"
        status_daily_text_2         = ""
        status_daily_text_3         = ""
        status_daily_text_4         = ""
        images_daily_text_1         = zip([], ["BlackGEM did not observe last night."])


    return status_daily_text_1, status_daily_text_2, status_daily_text_3, status_daily_text_4, images_daily_text_1, extragalactic_sources_id, transients_filename, gaia_filename, extragalactic_filename


def NightView(request, obs_date):
    '''
    Finds and displays data from a certain date.
    '''

    response = "You're looking at BlackGEM date %s."

    obs_date = str(obs_date)
    extended_date = obs_date[:4] + "-" + obs_date[4:6] + "-" + obs_date[6:]

    data_length, num_in_gaia, extragalactic_sources_length, extragalactic_sources, images_urls_sorted, \
        transients_filename, gaia_filename, extragalactic_filename = get_blackgem_stats(obs_date)

    if data_length == "1": data_length_plural = ""; data_length_plural_2 = "s"
    else: data_length_plural = "s"; data_length_plural_2 = "ve"
    if num_in_gaia == "1": gaia_plural = ""
    else: gaia_plural = "es"
    if extragalactic_sources_length == "1": extragalactic_sources_plural = ""
    else: extragalactic_sources_plural = "s"

    context = {
        "response"                      : response % obs_date,
        "obs_date"                      : obs_date,
        "extended_date"                 : extended_date,
        "mjd"                           : int(Time(extended_date + "T00:00:00.00", scale='utc').mjd),
        "data_length"                   : data_length,
        "num_in_gaia"                   : num_in_gaia,
        "extragalactic_sources_length"  : extragalactic_sources_length,
        "extragalactic_sources_id"      : extragalactic_sources[0],
        "extragalactic_sources_name"    : extragalactic_sources[1],
        "extragalactic_sources_ra"      : extragalactic_sources[2],
        "extragalactic_sources_dec"     : extragalactic_sources[3],
        "extragalactic_sources_check"   : extragalactic_sources[4],
        # "extragalactic_sources_jpg"     : extragalactic_sources[4],
        "data_length_plural"            : data_length_plural,
        "data_length_plural_2"          : data_length_plural_2,
        "gaia_plural"                   : gaia_plural,
        "extragalactic_sources_plural"  : extragalactic_sources_plural,
        "images_urls_sorted"            : images_urls_sorted,
    }

    if (data_length == "0") and (num_in_gaia == "0") and (extragalactic_sources_length == "0") and (extragalactic_sources[0] == "") and (images_urls_sorted == ""):
        observed_string = "BlackGEM did not observe that night (" + extended_date + ")"
        images_daily_text_1 = zip([], ["BlackGEM did not observe that night."])
    else:
        observed_string = "Yes!"
        images_daily_text_1 = zip(images_urls_sorted, extragalactic_sources[0], extragalactic_sources[1], extragalactic_sources[2], extragalactic_sources[3], extragalactic_sources[4])
        # images_daily_text_1 = zip(images_urls_sorted, extragalactic_sources[0], extragalactic_sources[1], extragalactic_sources[2], extragalactic_sources[3])

    context['observed']                 = observed_string
    context['images_daily_text_1']      = images_daily_text_1
    context['transients_filename']      = transients_filename
    context['gaia_filename']            = gaia_filename
    context['extragalactic_filename']   = extragalactic_filename

    return render(request, "status/index.html", context)


def status_to_GEMTOM(request):
    '''
    Imports a target from the Status tab
    '''

    id = request.POST.get('id')
    name = request.POST.get('name')
    ra = request.POST.get('ra')
    dec = request.POST.get('dec')

    add_to_GEMTOM(id, name, ra, dec)

    return redirect(reverse('tom_targets:list'))



## =============================================================================
## ------------------- Codes for the Recent Transients page --------------------


def blackgem_recent_transients():
    '''
    Fetches BlackGEM's recent transients and returns as a pandas dataframe
    '''

    recent_transients = pd.read_csv("./data/BlackGEM_Transients_Last30Days.csv")

    return recent_transients

# def update_recent_transients(days_since_last_update):
#     '''
#     Fetches BlackGEM's recent transients and updates
#     '''
#
#     get_recent_blackgem_transients(days_since_last_update)



def get_recent_blackgem_transients(days_since_last_update):

    obs_date = date.today() - timedelta(1)
    extended_obs_date = obs_date.strftime("%Y-%m-%d")
    obs_date = obs_date.strftime("%Y%m%d")
    mjd = int(Time(extended_obs_date + "T00:00:00.00", scale='utc').mjd)

    ## Get previous history:
    previous_history = blackgem_recent_transients()

    if days_since_last_update > 30:
        days_since_last_update = 30

    base_url = 'http://xmm-ssc.irap.omp.eu/claxson/BG_images/'

    datestamp = datetime.strptime(obs_date, "%Y%m%d")

    dates = []
    for days_back in range(days_since_last_update):
        this_datestamp = datestamp - timedelta(days_back)
        this_date = this_datestamp.strftime("%Y%m%d")
        dates.append(this_date)


    update_data = False

    data_list = []

    num_sources = 0
    for obs_date in dates:
        extended_date = obs_date[:4] + "-" + obs_date[4:6] + "-" + obs_date[6:]

        ## Get the list of files from Hugo's server
        files = list_files(base_url + obs_date)

        ## Check to find the transients file. It could be under one of two names...
        if extended_date+"_gw_BlackGEM_transients.csv" in files:
            transients_filename = base_url + obs_date + "/" + extended_date + "_gw_BlackGEM_transients.csv"
            data = pd.read_csv(transients_filename)
        elif extended_date+"_BlackGEM_transients.csv" in files:
            transients_filename = base_url + obs_date + "/" + extended_date + "_BlackGEM_transients.csv"
            data = pd.read_csv(transients_filename)
        else:
            ## If it doesn't exist, assume BlackGEM didn't observe any transients that night.
            print("No transients on", obs_date)
            continue


        d           = pd.Series([extended_date])
        date_column = d.repeat(len(data))
        date_column = date_column.set_axis(range(len(data)))
        data["last_obs"] = date_column

        # data_list.append(data.iloc[:20])
        data_list.append(data)
        num_sources += len(data)
        print(obs_date, "--", len(data), "\t Total:", num_sources)

    ## If there's any new data, combine it together
    if data_list:
        update_data = True
        df_new = pd.concat(data_list).reset_index(drop=True)

    ## --- Remove data older than 30 days ---
    ## First, is there any data older than 30 days?
    oldest_date = previous_history['last_obs'].iloc[-1]
    oldest_datestamp = datetime.strptime(oldest_date, "%Y-%m-%d")
    age_of_oldest_data = (datestamp - oldest_datestamp).days
    print("Oldest data is " + str(age_of_oldest_data) + " days old (" + oldest_date + ").")

    ## If there is...
    if age_of_oldest_data > 30:
        update_data = True
        print("Removing data older than 30 days...")

        ## We need to find the index at which data is older than 30 days
        ## First, what day was it 31 days ago?
        old_datestamp = datestamp - timedelta(31)
        old_date = old_datestamp.strftime("%Y-%m-%d")
        print("Looking for data from 31 days ago (" + old_date + ")...")

        ## Find the first occurance of this value in the list.
        old_date_index = (previous_history['last_obs'].values == old_date).argmax()

        ## Sometimes that date doesn't show up.
        ## In those circumstances, we iterate back through time until we find the next date.
        n = 0
        while old_date_index == 0:
            print("No data from " + old_date + "; continuing to the next date...")
            n += 1

            ## Each time, we find the new date, and the index of its first occurance.
            old_datestamp = datestamp - timedelta(31+n)
            old_date = old_datestamp.strftime("%Y-%m-%d")
            old_date_index = (previous_history['last_obs'].values == old_date).argmax()

            ## To prevent an infinite loop, which shouldn't happen, we do this.
            if n == age_of_oldest_data:
                print("This shouldn't print! Please check your data, something's gone wrong...")
                break

        print("Data found from " + old_date + ".")
        ## When we save data, save only up to this index.

        ## If there's new data...
        if data_list:
            ## ...combine it with the old.
            df = pd.concat([df_new, previous_history.iloc[:old_date_index]]).reset_index(drop=True)
        else:
            ## Otherwise, just use the old data.
            df = previous_history.iloc[:old_date_index].reset_index(drop=True)

        oldest_date = df['last_obs'].iloc[-1]
        oldest_datestamp = datetime.strptime(oldest_date, "%Y-%m-%d")
        age_of_oldest_data = (datestamp - oldest_datestamp).days
        print("Oldest data is now " + str(age_of_oldest_data) + " days old.")
    else:
        ## If there's no data older than 30 days, just use it all!
        if data_list:
            df = pd.concat([df_new, previous_history]).reset_index(drop=True)
        else:
            df = previous_history

    if update_data:
        print("Updating Recent History...")
        ## Make new columns and create new, truncated old columns
        ## Remove bugged values
        df['q_max'] = df['q_max'].replace(99,np.nan)
        df['u_max'] = df['u_max'].replace(99,np.nan)
        df['i_max'] = df['i_max'].replace(99,np.nan)

        ## Round values for displaying
        df['ra_sml']        = round(df['ra'],4)
        df['dec_sml']       = round(df['dec'],4)
        df['snr_zogy_sml']  = round(df['snr_zogy'],1)
        df['iauname_short'] = df['iauname'].str[5:]
        df['q_min_sml']     = round(df['q_min'],1)
        df['u_min_sml']     = round(df['u_min'],1)
        df['i_min_sml']     = round(df['i_min'],1)
        df['q_max_sml']     = round(df['q_max'],1)
        df['u_max_sml']     = round(df['u_max'],1)
        df['i_max_sml']     = round(df['i_max'],1)
        df['q_dif']         = round(df['q_max']-df['q_min'],2)
        df['u_dif']         = round(df['u_max']-df['u_min'],2)
        df['i_dif']         = round(df['i_max']-df['i_min'],2)


        df.to_csv("./data/BlackGEM_Transients_Last30Days.csv", index=False)



class TransientsView(TemplateView):
    template_name = 'recent_transients.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        ## --- Update Recent Transients ---
        recent_transients = blackgem_recent_transients()
        dates = list(recent_transients.last_obs)
        dates = [this_date[0:4] + this_date[5:7] + this_date[8:] for this_date in dates]

        ## --- Check to see if the recent history is up to date. If not, update.
        ## Get yesterday's date...
        yesterday_date = datetime.today() - timedelta(1)
        yesterday_date = yesterday_date.replace(hour=0, minute=0, second=0, microsecond=0)

        ## Get the most recent date...
        most_recent_date = datetime.strptime(dates[0], "%Y%m%d")

        ## Find the difference...
        difference = yesterday_date - most_recent_date
        days_since_last_update = difference.days
        print("Days since last update:", days_since_last_update)

        yesterday_date_string = yesterday_date.strftime("%Y%m%d")

        if days_since_last_update > 0:
            get_recent_blackgem_transients(days_since_last_update)
            recent_transients = blackgem_recent_transients()
            dates = list(recent_transients.last_obs)
            dates = [this_date[0:4] + this_date[5:7] + this_date[8:] for this_date in dates]

        return context


    ## ===== Plot the transients from the past 30 days =====

    ## --- Step 1: The 'Recent Transients' Table ---
    ## Uses a Dash AG Grid

    # Initialize the Dash app
    app = DjangoDash('RecentTransients')

    # Read CSV data
    df = pd.read_csv('./data/BlackGEM_Transients_Last30Days.csv')

    ## Define the layout of the Dash app
    app.layout = html.Div([
        dag.AgGrid(
            id='csv-grid',
            rowData=df.to_dict('records'),
            columnDefs=[
                {'headerName': 'BGEM ID', 'field': 'runcat_id'},
                {'headerName': 'IAU Name', 'field': 'iauname_short'},
                {'headerName': 'RA', 'field': 'ra_sml'},
                {'headerName': 'Dec', 'field': 'dec_sml'},
                {'headerName': '#Datapoints', 'field': 'datapoints'},
                {'headerName': 'S/N', 'field': 'snr_zogy_sml'},
                {'headerName': 'q min', 'field': 'q_min_sml',   'minWidth': 75, 'maxWidth': 75},
                {'headerName': 'q dif', 'field': 'q_dif',       'minWidth': 75, 'maxWidth': 75},
                {'headerName': 'u min', 'field': 'u_min_sml',   'minWidth': 75, 'maxWidth': 75},
                {'headerName': 'u dif', 'field': 'u_dif',       'minWidth': 75, 'maxWidth': 75},
                {'headerName': 'i min', 'field': 'i_min_sml',   'minWidth': 75, 'maxWidth': 75},
                {'headerName': 'i dif', 'field': 'i_dif',       'minWidth': 75, 'maxWidth': 75},
                {'headerName': 'Last Obs', 'field': 'last_obs', 'maxWidth': 110},
            ],
            defaultColDef={
                'sortable': True,
                'filter': True,
                'resizable': True,
                'editable': True,
            },
            columnSize="autoSize",
            dashGridOptions = {"skipHeaderOnAutoSize": True, "rowSelection": "single"},
            style={'height': '400px', 'width': '100%'},  # Set explicit height for the table
            className='ag-theme-balham'  # Add a theme for better appearance
        ),
        dcc.Store(id='selected-row-data'),  # Store to hold the selected row data, for when a row is clicked

        ## The following are sections that show information based on the row clicked
        html.Div(id='information-div'),     ## For Step 2: Displays the Object ID, IAU Name, RA, and Dec
        dcc.Graph(id='lightcurve-graph'),   ## For Step 3: Displays the Lightcurve
        html.Div(id='output-div'),          ## For Step 4: Displays the link to Transients, the 'Add to GEMTOM' button, and the full data.

    ], style={'height': '1700px', 'width': '100%'} # Set explicit height for the full app, includine the extra information.
    )

    ## --- Step 1 Cont': Handle selecting rows ---
    @app.callback(
        Output('selected-row-data', 'data'),
        Input('csv-grid', 'selectedRows')
    )
    def update_selected_row(selectedRows):
        if selectedRows:
            return selectedRows[0]  # Assuming single row selection
        return {}

    ## --- Step 2: Display the Object ID, IAU Name, RA, and Dec ---
    @app.callback(
        Output('information-div', 'children'),
        Input('selected-row-data', 'data')
    )
    def display_information(row_data):
        if row_data:
            return html.Div([
                html.P("Object " + str(row_data["runcat_id"]), style={'font-size':'20px'}),
                html.P(str(row_data["iauname"]), style={'font-size':'17px'}),
                html.P("RA: " + str(row_data["ra"]) + ", Dec: " + str(row_data["dec"]), style={'font-size':'17px'}),
                ], style={'font-family': 'Arial', 'text-align': 'center'}
            )

        return html.Div(
            html.P("Select a row"),
            style={'font-family': 'Arial', 'text-align': 'center'}
        )


    ## --- Step 3: Make the Lightcurve of a given source ---
    @app.callback(
        Output('lightcurve-graph', 'figure'),
        Input('selected-row-data', 'data'),
        prevent_initial_call=True  # Prevent the callback from being called when the app loads
    )
    def create_lightcurve(row_data):

        if row_data:
            bgem_id = row_data['runcat_id']

            ## --- Lightcurve ---
            df_bgem_lightcurve, df_limiting_mag = get_lightcurve_from_BGEM_ID(bgem_id)
            fig = plot_BGEM_lightcurve(df_bgem_lightcurve, df_limiting_mag)

            return fig

        return go.Figure()

    ## --- Step 4: Display the link to Transients, the 'Add to GEMTOM' button, and the full data ---

    ## First, create the 'Add to GEMTOM' button
    ## Callback to handle button click:
    @app.callback(
        Output('button-click-message', 'children'),  # Allow multiple outputs to the same component
        Input('call-function-button', 'n_clicks'),
        State('selected-row-data', 'data'),
        prevent_initial_call=True  # Prevent the callback from being called when the app loads
    )
    ## Function to add the transient to GEMTOM:
    def transient_to_GEMTOM(n_clicks, row_data):
        if n_clicks > 0 and row_data:

            id      = str(row_data['runcat_id'])
            name    = str(row_data['iauname'])
            ra      = str(row_data['ra'])
            dec     = str(row_data['dec'])

            add_to_GEMTOM(id, name, ra, dec)

            return [html.P(f"Transient added to GEMTOM as " + name, style={'display': 'inline-block'}), html.A(". Please see the Targets page", href="/targets/", target="_blank", style={'text-decoration':'None', 'display': 'inline-block'}), html.P(".", style={'display': 'inline-block'})]
            # return [html.P(f"Target added to GEMTOM as " + name + ". Please see the ", style={'display': 'inline-block'}), html.P("Targets page.", style={'display': 'inline-block'})]

    ## Then, assumble all the rest of the information
    @app.callback(
        Output('output-div', 'children'),
        Input('selected-row-data', 'data')
    )
    def display_selected_row_data(row_data):

        ## When a row is selected, we either need to show the lightcurve or a link to request one:
        if row_data:

            # Format columns
            formatted_columns = [
                {'name': k, 'id': k, 'type': 'numeric', 'format': dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.fixed).group(True), 'presentation': 'input'} if isinstance(row_data[k], (int, float)) else {'name': k, 'id': k}
                for k in row_data.keys()
            ]

            ## Main data (Name, RA/Dec, Datapoints, etc.)
            table_1 = dash_table.DataTable(
                data=[row_data],
                columns=[[{'name': k, 'id': k} for k in row_data.keys()][i] for i in [1,3,4,5,6,7,8]],
                style_table={'margin': 'auto'}, style_cell={'textAlign': 'center', 'padding': '5px'}, style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
            ## q mag
            table_2 = dash_table.DataTable(
                data=[row_data],
                columns=[[{'name': k, 'id': k} for k in row_data.keys() if k in ['q_min', 'q_max', 'q_xtrsrc', 'q_rb', 'q_fwhm', 'q_dif']][i] for i in [1,2,0,3,4,5]],
                style_table={'margin': 'auto'}, style_cell={'textAlign': 'center', 'padding': '5px'}, style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
            ## u mag
            table_3 = dash_table.DataTable(
                data=[row_data],
                # columns=[[{'name': k, 'id': k} for k in row_data.keys()][i] for i in [13,14,15,16,17,44]],
                columns=[{'name': k, 'id': k} for k in row_data.keys() if k in ['u_min', 'u_max', 'u_xtrsrc', 'u_rb', 'u_fwhm', 'u_dif']],
                style_table={'margin': 'auto'}, style_cell={'textAlign': 'center', 'padding': '5px'}, style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
            ## i mag
            table_4 = dash_table.DataTable(
                data=[row_data],
                columns=[{'name': k, 'id': k} for k in row_data.keys() if k in ['i_min', 'i_max', 'i_xtrsrc', 'i_rb', 'i_fwhm', 'i_dif']],
                style_table={'margin': 'auto'}, style_cell={'textAlign': 'center', 'padding': '5px'}, style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
            ## Extra 1
            table_5 = dash_table.DataTable(
                data=[row_data],
                columns=[{'name': k, 'id': k} for k in row_data.keys()][23:29],
                style_table={'margin': 'auto'}, style_cell={'textAlign': 'center', 'padding': '5px'}, style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
            ## Extra 2
            table_6 = dash_table.DataTable(
                data=[row_data],
                columns=[{'name': k, 'id': k} for k in row_data.keys()][30:36],
                style_table={'margin': 'auto'}, style_cell={'textAlign': 'center', 'padding': '5px'}, style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )

            return html.Div(
                    [
                    html.A("BlackView page for this transient", href="https://staging.apps.blackgem.org/transients/blackview/show_runcat?runcatid=" + str(row_data['runcat_id']), target="_blank", style={'text-decoration':'None', "font-style": "italic"}),
                    html.Br(), html.Br(),
                    html.A("GEMTOM page for this transient", href='/transient/'+str(row_data['runcat_id']), target="_blank", style={'text-decoration':'None', "font-style": "italic"}),
                    html.Br(), html.Br(),
                    html.Div(html.Button('Add to GEMTOM', id='call-function-button', n_clicks=0, style={
                        'font-family': 'Arial',
                        'font-size': '16px',
                        'color': 'white',
                        'background-color': '#007bff',
                        'border': 'none',
                        'padding': '10px 20px',
                        'text-align': 'center',
                        'text-decoration': 'none',
                        'display': 'inline-block',
                        'margin': '4px 2px',
                        'cursor': 'pointer',
                        'border-radius': '12px'
                    })),
                    html.P(id='button-click-message')  # Div to display the message when button is clicked
                    ] +
                [table_1] + [table_2] + [table_3] + [table_4] + [table_5] + [table_6],
                style={'font-family': 'Arial', 'text-align': 'center'})

        return



## =============================================================================
## ------------------- Codes for the Live Feed page ---------------------


class LiveFeed(TemplateView):
    template_name = 'transient_search.html'


    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context

    def plot_graph_view(request):
        # Example DataFrame
        data = {
            'x': [1, 2, 3, 4, 5],
            'y': [10, 15, 13, 17, 21]
        }
        df = pd.DataFrame(data)

        # Create a Plotly graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='lines+markers', name='Line and Marker'))

        # Convert the Plotly graph to HTML
        plot_div = plot(fig, output_type='div')

        # Pass the plot_div to the template
        return render(request, 'transient/index.html', {'plot_div': plot_div})


## =============================================================================
## ------------------- Codes for the Transient Search page ---------------------

class TransientSearchView(TemplateView):
    template_name = 'transient_search.html'


    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context

    def plot_graph_view(request):
        # Example DataFrame
        data = {
            'x': [1, 2, 3, 4, 5],
            'y': [10, 15, 13, 17, 21]
        }
        df = pd.DataFrame(data)

        # Create a Plotly graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='lines+markers', name='Line and Marker'))

        # Convert the Plotly graph to HTML
        plot_div = plot(fig, output_type='div')

        # Pass the plot_div to the template
        return render(request, 'transient/index.html', {'plot_div': plot_div})

def get_limiting_magnitudes_from_BGEM_ID(blackgem_id):

    creds_user_file = str(Path.home()) + "/.bg_follow_user_john_creds"
    creds_db_file = str(Path.home()) + "/.bg_follow_transientsdb_creds"

    # Instantiate the BlackGEM object
    bg = BlackGEM(creds_user_file=creds_user_file, creds_db_file=creds_db_file)

    qu = """\
    SELECT i1.id
          ,i1."date-obs"
          ,i1."mjd-obs"
          ,i1."t-lmag"
          ,i1.filter
      FROM image i1
          ,(SELECt i0.id AS imageid
             FROM image i0
                 ,(SELECT x.object
                     FROM assoc a
                         ,extractedsource x
                    WHERE a.runcat = '%(blackgem_id)s'
                      AND a.xtrsrc = x.id
                   GROUP BY x.object
                  ) t0
            WHERE i0.object = t0.object
           EXCEPT
           SELECT x.image AS imageid
             FROM assoc a
                 ,extractedsource x
            WHERE a.runcat = 31744960
              AND a.xtrsrc = x.id
            ) t1
     WHERE i1.id = t1.imageid
       AND i1."tqc-flag" <> 'red'
    ORDER BY i1."date-obs"
    """

    params = {'blackgem_id': blackgem_id}
    query = qu % (params)
    l_results = bg.run_query(query)

    df_limiting_mag = pd.DataFrame(l_results, columns=['id','date_obs','mjd','limiting_mag','filter'])
    print(df_limiting_mag)

    return(df_limiting_mag)

def get_lightcurve_from_BGEM_ID(transient_id):

    print("Getting lightcurve for transient ID " + str(transient_id) + "...")

    creds_user_file = str(Path.home()) + "/.bg_follow_user_john_creds"
    creds_db_file = str(Path.home()) + "/.bg_follow_transientsdb_creds"

    # Instantiate the BlackGEM object
    bg = BlackGEM(creds_user_file=creds_user_file, creds_db_file=creds_db_file)

    # Create an instance of the Transients Catalog
    tc = TransientsCatalog(bg)
    df_limiting_mag = get_limiting_magnitudes_from_BGEM_ID(transient_id)

    # Get all the associated extracted sources for this transient
    # Note that you can specify the columns yourself, but here we use the defaults
    bg_columns, bg_results = tc.get_associations(transient_id)
    df_bgem_lightcurve = pd.DataFrame(bg_results, columns=bg_columns)

    return df_bgem_lightcurve, df_limiting_mag

def BGEM_to_GEMTOM_photometry(df_bg_assocs):

    gemtom_photometry = pd.DataFrame({
        'mjd' : df_bg_assocs["i.\"mjd-obs\""],
        'mag' : df_bg_assocs["x.mag_zogy"],
        'magerr' : df_bg_assocs["x.magerr_zogy"],
        'filter' : df_bg_assocs["i.filter"],
    })

    return gemtom_photometry

def BGEM_ID_View(request, bgem_id):
    '''
    Finds and displays data from a certain date.
    '''

    df_bgem_lightcurve, df_limiting_mag = get_lightcurve_from_BGEM_ID(bgem_id)
    print(df_bgem_lightcurve)
    print(df_bgem_lightcurve.columns)

    response = "You're looking at BlackGEM transient %s."

    ## --- Location on Sky ---
    fig = plot_BGEM_location_on_sky(df_bgem_lightcurve)
    location_on_sky = plot(fig, output_type='div')

    ## --- Lightcurve ---
    fig = plot_BGEM_lightcurve(df_bgem_lightcurve, df_limiting_mag)
    lightcurve = plot(fig, output_type='div')

    # Pass the plot_div to the template
    # return render(request, 'transient/index.html')

    ## Get the name, ra, and dec:

    user_home = str(Path.home())
    creds_user_file = user_home + "/.bg_follow_user_john_creds"
    creds_db_file = user_home + "/.bg_follow_transientsdb_creds"

    # Instantialte the BlackGEM object, with a connection to the database
    bg = BlackGEM(creds_user_file=creds_user_file, creds_db_file=creds_db_file)

    qu = """\
    SELECT id
          ,iau_name
          ,ra_deg
          ,dec_deg
      FROM runcat
     WHERE id = '%(bgem_id)s'
    """

    params = {'bgem_id': bgem_id}
    query = qu % (params)

    l_results = bg.run_query(query)
    source_data = pd.DataFrame(l_results, columns=['id','iau_name','ra_deg','dec_deg'])
    iau_name    = source_data['iau_name'][0]
    ra          = source_data['ra_deg'][0]
    dec         = source_data['dec_deg'][0]
    # print(source_data)
    # print(l_results)

    ## Detail each observation:

    app = DjangoDash('EachObservation')

    df_new = df_bgem_lightcurve.rename(columns={
        'a.xtrsrc'          : "xtrsrc",
        'x.ra_psf_d'        : "ra_psf_d",
        'x.dec_psf_d'       : "dec_psf_d",
        'x.flux_zogy'       : "flux_zogy",
        'x.fluxerr_zogy'    : "fluxerr_zogy",
        'x.mag_zogy'        : "mag_zogy",
        'x.magerr_zogy'     : "magerr_zogy",
        'i."mjd-obs"'       : "mjd_obs",
        'i."date-obs"'      : "date_obs",
        'i.filter'          : "filter",
    })
    # df_new.style.format({
    #     # 'runcat_id' : make_runcat_clickable,
    #     'xtrsrc' : make_xtrsrc_clickable
    # })

    df_new['xtrsrc'] = df_new['xtrsrc'].apply(lambda x: f'[{x}](https://staging.apps.blackgem.org/transients/blackview/show_xtrsrc/{x})')

    # print(df_new)

    ## Define the layout of the Dash app
    app.layout = html.Div([
        dag.AgGrid(
            id='observation-grid',
            rowData=df_new.to_dict('records'),
            # rowData=rowData_new,
            # columnDefs=[{'headerName': col, 'field': col} for col in df_new.columns],
            # columnDefs=[
            #             {'headerName': '1', 'field': 'x.ra_psf_d'},
            #             {'headerName': '2', 'field': 'x.dec_psf_d'},
            # ],
            columnDefs=[
                        {'headerName': 'a.xtrsrc', 'field':  'xtrsrc', 'cellRenderer': 'markdown'},
                        {'headerName': 'i.mjd_obs', 'field':  'mjd_obs'},
                        {'headerName': 'i.date_obs', 'field':  'date_obs'},
                        {'headerName': 'x.ra_psf_d', 'field':  'ra_psf_d'},
                        {'headerName': 'x.dec_psf_d', 'field':  'dec_psf_d'},
                        {'headerName': 'x.flux_zogy', 'field':  'flux_zogy'},
                        {'headerName': 'x.fluxerr_zogy', 'field':  'fluxerr_zogy'},
                        {'headerName': 'x.mag_zogy', 'field':  'mag_zogy'},
                        {'headerName': 'x.magerr_zogy', 'field':  'magerr_zogy'},
                        {'headerName': 'i.filter', 'field': 'filter'},
            ],
            defaultColDef={
                'sortable': True,
                'filter': True,
                'resizable': True,
                'editable': True,
            },
            columnSize="autoSize",
            dashGridOptions={
                "skipHeaderOnAutoSize": True,
                "rowSelection": "single",
            },
            style={'height': '400px', 'width': '100%'},  # Set explicit height for the grid
            className='ag-theme-balham'  # Add a theme for better appearance
        ),
        dcc.Store(id='selected-row-data'),  # Store to hold the selected row data
        html.Div(id='output-div'),  # Div to display the information
    ], style={'height': '400px', 'width': '100%'}
    )

    # print(df_new)

    # ## Render the app
    # def obs_dash_view(request):
    #     return render(request, 'transient/index.html')

    context = {
        "bgem_id"           : bgem_id,
        "iau_name"          : iau_name,
        "ra"                : ra,
        "dec"               : dec,
        "dataframe"         : df_bgem_lightcurve,
        "columns"           : df_bgem_lightcurve.columns,
        "location_on_sky"   : location_on_sky,
        "lightcurve"        : lightcurve,
    }

    return render(request, "transient/index.html", context)



## =============================================================================
## --------------------- Codes for updating various data -----------------------


class UpdateZTFView(LoginRequiredMixin, RedirectView):
    """
    View that handles the updating of ZTF data. Requires authentication.
    """

    def get(self, request, *args, **kwargs):
        """
        Method that handles the GET requests for this view. Calls the management command to update the reduced data and
        adds a hint using the messages framework about automation.
        """


        print("\n\n ===== Running UpdateZTFView... ===== \n")

        # QueryDict is immutable, and we want to append the remaining params to the redirect URL
        query_params = request.GET.copy()
        print("Fetching ZTF data:")
        print(query_params)

        target_name = query_params.pop('target', None)
        target_id   = query_params.pop('target_id', None)
        target_ra   = query_params.pop('target_ra', None)
        target_dec  = query_params.pop('target_dec', None)
        # print(target_id[0])
        # print(target_ra[0])
        # print(target_dec[0])

        out = StringIO()

        # if target_id:
        if isinstance(target_name, list):   target_name    = target_name[-1]
        if isinstance(target_id, list):     target_id      = target_id[-1]
        if isinstance(target_ra, list):     target_ra      = target_ra[-1]
        if isinstance(target_dec, list):    target_dec     = target_dec[-1]

        form = { \
            'observation_record': None, \
            'target': target_name, \
            'files': "./data/GEMTOM_ZTF_Test.csv", \
            'data_product_type': 'ztf_data', \
            'referrer': '/targets/' + target_id + '/?tab=ztf'}

        # print(form)

        print("-- ZTF: Looking for target...", end="\r")
        try:
            lcq = lightcurve.LCQuery.from_position(target_ra, target_dec, 5)

            ZTF_data_full = pd.DataFrame(lcq.data)
            ZTF_data = pd.DataFrame({'JD' : lcq.data.mjd+2400000.5, 'Magnitude' : lcq.data.mag, 'Magnitude_Error' : lcq.data.magerr})

            if len(ZTF_data) == 0:
                raise Exception("No ZTF data found within 5 arcseconds of given RA/Dec.")

            print("-- ZTF: Looking for target... target found.")
            print(lcq.__dict__)

            ## Save ZTF Data
            df = ZTF_data
            # print(df)
            if not os.path.exists("./data/" + target_name + "/none/"):
                os.makedirs("./data/" + target_name + "/none/")
            filepath = "./data/" + target_name + "/none/" + target_name + "_ZTF_Data.csv"
            df.to_csv(filepath)

            target_instance = Target.objects.get(pk=target_id)

            dp = DataProduct(
                target=target_instance,
                observation_record=None,
                data=target_name + "/none/" + target_name + "_ZTF_Data.csv",
                product_id=None,
                data_product_type='ztf_data'
            )
            # print(dp)
            dp.save()

            ## Ingest the data
            successful_uploads = []

            try:
                run_hook('data_product_post_upload', dp)
                reduced_data = run_data_processor(dp)

                if not settings.TARGET_PERMISSIONS_ONLY:
                    for group in form.cleaned_data['groups']:
                        assign_perm('tom_dataproducts.view_dataproduct', group, dp)
                        assign_perm('tom_dataproducts.delete_dataproduct', group, dp)
                        assign_perm('tom_dataproducts.view_reduceddatum', group, reduced_data)
                successful_uploads.append(str(dp))

            except InvalidFileFormatException as iffe:
                print("Invalid File Format Exception!")
                print(iffe)
                ReducedDatum.objects.filter(data_product=dp).delete()
                dp.delete()
                messages.error(
                    self.request,
                    'File format invalid for file {0} -- Error: {1}'.format(str(dp), iffe)
                )
            except Exception as iffe:
                print("Exception!")
                print(iffe)
                ReducedDatum.objects.filter(data_product=dp).delete()
                dp.delete()
                messages.error(self.request, 'There was a problem processing your file: {0}'.format(str(dp)))

            if successful_uploads:
                print("Successful upload!")
                messages.success(
                    self.request,
                    'Successfully uploaded: {0}'.format('\n'.join([p for p in successful_uploads]))
                )
            else:
                print("Upload unsuccessful!")


        except Exception as e:
            messages.error(
                self.request,
                'Error while fetching ZTF data; ' + str(e)
            )



        return redirect(form.get('referrer', '/'))

def get_blackgem_id_from_iauname(iauname):

    creds_user_file = str(Path.home()) + "/.bg_follow_user_john_creds"
    creds_db_file = str(Path.home()) + "/.bg_follow_transientsdb_creds"

    # Instantiate the BlackGEM object
    bg = BlackGEM(creds_user_file=creds_user_file, creds_db_file=creds_db_file)

    qu= """\
    SELECT id
      FROM runcat
     WHERE iau_name = '%(iau_name)s'
    """
    params = {'iau_name': iauname}
    query = qu % (params)
    l_results = bg.run_query(query)

    blackgem_id = l_results[0][0]

    return blackgem_id

def add_bgem_lightcurve_to_GEMTOM(target_name, target_id, target_blackgemid):

    form = { \
        'observation_record': None, \
        'target': target_name, \
        'files': "./data/GEMTOM_BlackGEM_Test.csv", \
        'data_product_type': 'blackgem_data', \
        'referrer': '/targets/' + target_id + '/'}
    # print(form)

    print("-- BlackGEM: Getting Data...", end="\r")

    successful_uploads = []
    iffe = ""
    iffe2 = ""
    iffe3 = ""

    try:
        df_bgem_lightcurve, df_limiting_mag = get_lightcurve_from_BGEM_ID(target_blackgemid)
        photometry = BGEM_to_GEMTOM_photometry(df_bgem_lightcurve)
        print(df_bgem_lightcurve)
        print(df_bgem_lightcurve.columns)

        print("-- BlackGEM: Getting Data... Done.")

        ## Save ZTF Data
        df = photometry
        # print(df)
        if not os.path.exists("./data/" + target_name + "/none/"):
            os.makedirs("./data/" + target_name + "/none/")
        filepath = "./data/" + target_name + "/none/" + target_name + "_BGEM_Data.csv"
        df.to_csv(filepath, index=False)

        target_instance = Target.objects.get(pk=target_id)

        dp = DataProduct(
            target=target_instance,
            observation_record=None,
            data=target_name + "/none/" + target_name + "_BGEM_Data.csv",
            product_id=None,
            data_product_type='blackgem_data'
        )
        # print(dp)
        dp.save()

        ## Ingest the data
        try:
            run_hook('data_product_post_upload', dp)
            reduced_data = run_data_processor(dp)

            if not settings.TARGET_PERMISSIONS_ONLY:
                for group in form.cleaned_data['groups']:
                    assign_perm('tom_dataproducts.view_dataproduct', group, dp)
                    assign_perm('tom_dataproducts.delete_dataproduct', group, dp)
                    assign_perm('tom_dataproducts.view_reduceddatum', group, reduced_data)
            successful_uploads.append(str(dp))

        except InvalidFileFormatException as iffe:
            print("Invalid File Format Exception!")
            print(iffe)
            ReducedDatum.objects.filter(data_product=dp).delete()
            dp.delete()

        except Exception as iffe2:
            print("Exception!")
            print(iffe2)
            ReducedDatum.objects.filter(data_product=dp).delete()
            dp.delete()

    except Exception as iffe3:
        print(iffe3)

    return successful_uploads, dp, iffe, iffe2, iffe3, form



class UpdateBlackGEMView(LoginRequiredMixin, RedirectView):
    """
    View that handles the updating of BlackGEM data. Requires authentication.
    """

    def get(self, request, *args, **kwargs):
        """
        Method that handles the GET requests for this view. Calls the management command to update the reduced data and
        adds a hint using the messages framework about automation.
        """


        print("\n\n ===== Running UpdateBlackGEMView... ===== \n")

        # QueryDict is immutable, and we want to append the remaining params to the redirect URL
        query_params = request.GET.copy()
        print("Fetching BlackGEM data:")
        print(query_params)

        target_name         = query_params.pop('target', None)
        target_id           = query_params.pop('target_id', None)
        target_blackgemid   = query_params.pop('blackgem_id', None)

        out = StringIO()

        # if target_id:
        if isinstance(target_name, list):       target_name         = target_name[-1]
        if isinstance(target_id, list):         target_id           = target_id[-1]
        if isinstance(target_blackgemid, list): target_blackgemid   = target_blackgemid[-1]

        ## Upload to GEMTOM
        successful_uploads, dp, iffe, iffe2, iffe3, form = add_bgem_lightcurve_to_GEMTOM(target_name, target_id, target_blackgemid)

        ## Return messages for sucess or failute
        if successful_uploads:
            print("Successful upload!")
            messages.success(
                self.request,
                'Successfully uploaded: {0}'.format('\n'.join([p for p in successful_uploads]))
            )
        else:
            print("Upload unsuccessful!")
            if iffe:
                messages.error(
                    self.request,
                    'File format invalid for file {0} -- Error: {1}'.format(str(dp), iffe)
                )
            if iffe2:
                messages.error(self.request, 'There was a problem processing your file: {0} -- Error: {1}'.format(str(dp), iffe2))
            if iffe3:
                messages.error(
                    self.request,
                    'Error while fetching BlackGEM data; ' + str(iffe2)
                )

        return redirect(form.get('referrer', '/'))

#
# class UpdateBlackGEMView(LoginRequiredMixin, RedirectView):
#     """
#     View that handles the updating of BlackGEM data. Requires authentication.
#     """
#
#     def get(self, request, *args, **kwargs):
#         """
#         Method that handles the GET requests for this view. Calls the management command to update the reduced data and
#         adds a hint using the messages framework about automation.
#         """
#
#
#         print("\n\n ===== Running UpdateZTFView... ===== \n")
#
#         # QueryDict is immutable, and we want to append the remaining params to the redirect URL
#         query_params = request.GET.copy()
#         print("Fetching ZTF data:")
#         print(query_params)
#
#         target_name = query_params.pop('target', None)
#         target_id   = query_params.pop('target_id', None)
#         target_ra   = query_params.pop('target_ra', None)
#         target_dec  = query_params.pop('target_dec', None)
#         # print(target_id[0])
#         # print(target_ra[0])
#         # print(target_dec[0])
#
#         out = StringIO()
#
#         # if target_id:
#         if isinstance(target_name, list):   target_name    = target_name[-1]
#         if isinstance(target_id, list):     target_id      = target_id[-1]
#         if isinstance(target_ra, list):     target_ra      = target_ra[-1]
#         if isinstance(target_dec, list):    target_dec     = target_dec[-1]
#
#         form = { \
#             'observation_record': None, \
#             'target': target_name, \
#             'files': "./data/GEMTOM_ZTF_Test.csv", \
#             'data_product_type': 'ztf_data', \
#             'referrer': '/targets/' + target_id + '/?tab=ztf'}
#
#         # print(form)
#
#         print("-- ZTF: Looking for target...", end="\r")
#         try:
#             lcq = lightcurve.LCQuery.from_position(target_ra, target_dec, 5)
#
#             ZTF_data_full = pd.DataFrame(lcq.data)
#             ZTF_data = pd.DataFrame({'JD' : lcq.data.mjd+2400000.5, 'Magnitude' : lcq.data.mag, 'Magnitude_Error' : lcq.data.magerr})
#
#             if len(ZTF_data) == 0:
#                 raise Exception("No ZTF data found within 5 arcseconds of given RA/Dec.")
#
#             print("-- ZTF: Looking for target... target found.")
#             print(lcq.__dict__)
#
#             ## Save ZTF Data
#             df = ZTF_data
#             # print(df)
#             if not os.path.exists("./data/" + target_name + "/none/"):
#                 os.makedirs("./data/" + target_name + "/none/")
#             filepath = "./data/" + target_name + "/none/" + target_name + "_ZTF_Data.csv"
#             df.to_csv(filepath)
#
#             target_instance = Target.objects.get(pk=target_id)
#
#             dp = DataProduct(
#                 target=target_instance,
#                 observation_record=None,
#                 data=target_name + "/none/" + target_name + "_ZTF_Data.csv",
#                 product_id=None,
#                 data_product_type='ztf_data'
#             )
#             # print(dp)
#             dp.save()
#
#             ## Ingest the data
#             successful_uploads = []
#
#             try:
#                 run_hook('data_product_post_upload', dp)
#                 reduced_data = run_data_processor(dp)
#
#                 if not settings.TARGET_PERMISSIONS_ONLY:
#                     for group in form.cleaned_data['groups']:
#                         assign_perm('tom_dataproducts.view_dataproduct', group, dp)
#                         assign_perm('tom_dataproducts.delete_dataproduct', group, dp)
#                         assign_perm('tom_dataproducts.view_reduceddatum', group, reduced_data)
#                 successful_uploads.append(str(dp))
#
#             except InvalidFileFormatException as iffe:
#                 print("Invalid File Format Exception!")
#                 print(iffe)
#                 ReducedDatum.objects.filter(data_product=dp).delete()
#                 dp.delete()
#                 messages.error(
#                     self.request,
#                     'File format invalid for file {0} -- Error: {1}'.format(str(dp), iffe)
#                 )
#             except Exception as iffe:
#                 print("Exception!")
#                 print(iffe)
#                 ReducedDatum.objects.filter(data_product=dp).delete()
#                 dp.delete()
#                 messages.error(self.request, 'There was a problem processing your file: {0}'.format(str(dp)))
#
#             if successful_uploads:
#                 print("Successful upload!")
#                 messages.success(
#                     self.request,
#                     'Successfully uploaded: {0}'.format('\n'.join([p for p in successful_uploads]))
#                 )
#             else:
#                 print("Upload unsuccessful!")
#
#
#         except Exception as e:
#             messages.error(
#                 self.request,
#                 'Error while fetching ZTF data; ' + str(e)
#             )
#
#
#
#         return redirect(form.get('referrer', '/'))

#
# class AboutView(TemplateView):
#     template_name = 'about.html'
#
#
#     def get_context_data(self, **kwargs):
#         return {'targets': Target.objects.all()}
#
#
#     def post(self, request, **kwargs):
#         source_ra  = float(request.POST['num1'])
#         source_dec = float(request.POST['num2'])
#
#         print("-- ZTF: Looking for target...", end="\r")
#         lcq = lightcurve.LCQuery.from_position(source_ra, source_dec, 5)
#         ZTF_data_full = pd.DataFrame(lcq.data)
#         ZTF_data = pd.DataFrame({'JD' : lcq.data.mjd, 'Magnitude' : lcq.data.mag, 'Magnitude_Error' : lcq.data.magerr})
#
#         if len(ZTF_data) == 0:
#             raise Exception("-- ZTF: Target not found. Try AAVSO instead?")
#
#         print("-- ZTF: Looking for target... target found.")
#         print(lcq.__dict__)
#
#         df = ZTF_data # replace with your own data source
#
#         fig = px.scatter(df, x='JD', y='Magnitude')
#         fig.update_layout(
#             yaxis = dict(autorange="reversed")
#         )
#         return HttpResponse("Closest target within 5 arcseconds:" + fig.to_html() + ZTF_data_full.to_html())
