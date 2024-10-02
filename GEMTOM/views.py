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
import dash_bootstrap_components as dbc

from django.views.generic import TemplateView, FormView
from django.http import HttpResponse, HttpResponseRedirect, FileResponse, JsonResponse
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

## For the History View
import requests
from django.template import loader
from astropy.time import Time
from bs4 import BeautifulSoup
from datetime import datetime, date, timedelta, timezone
import numpy as np
import astropy.coordinates as coord
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for rendering plots
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import urllib, base64

## For the Transients View
from django_plotly_dash import DjangoDash
import dash_ag_grid as dag
import json
from pathlib import Path
import plotly.graph_objs as go
from plotly.offline import plot
from mocpy import MOC
from PIL import Image, ImageDraw
import shutil # save img locally
from urllib.request import urlretrieve

## For the ToO Forms
from .forms import ToOForm
from django.core.exceptions import ValidationError

## For the Live Feed
from dash.exceptions import PreventUpdate
import math
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astroplan import Observer

## BlackGEM Stuff
from blackpy import BlackGEM
from blackpy.catalogs.blackgem import TransientsCatalog

from tom_common.hooks import run_hook
# from tom_observations.models import Target
from tom_targets.models import Target
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
            processed_file.to_csv("./data/processed_file.csv", index=False)

            ## Output them into a StringIO format for the import_targets function
            csv_stream = StringIO(open(os.getcwd()+"/data/processed_file.csv", "rb").read().decode('utf-8'), newline=None)

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


def add_to_GEMTOM(id, name, ra, dec, tns_prefix=False, tns_name=False):

    # get_lightcurve(id)
    if tns_prefix and tns_name:
        name = tns_prefix + " " + tns_name

    gemtom_dataframe = pd.DataFrame({
        'name' : [name],
        'ra' : [ra],
        'dec' : [dec],
        'BlackGEM ID' : [int(id)],
        'type' : ['SIDEREAL'],
        'groups' : ['Public']
    })


    gemtom_dataframe = gemtom_dataframe.reindex(gemtom_dataframe.index)

    gemtom_dataframe.to_csv("./data/processed_file.csv", index=False)
    csv_stream = StringIO(open(os.getcwd()+"/data/processed_file.csv", "rb").read().decode('utf-8'), newline=None)

    ## And finally, read them in!
    result = import_targets(csv_stream)

    if tns_prefix and tns_name:
        print(tns_prefix + " " + tns_name)
        target = Target.objects.get(name=name)
        target.save(extras={'TNS Name': tns_prefix + " " + tns_name})

    return redirect(reverse('tom_targets:list'))

def test_print():
    print("Hello World!")

def plot_BGEM_lightcurve(df_bgem_lightcurve, df_limiting_mag):

    # print("\n\nBark!")
    # print(df_bgem_lightcurve.columns)
    # print(df_limiting_mag.columns)
    # print(df_limiting_mag[['mjd', 'magnitude', 'limiting_mag', 'filter', 'error']])

    time_now = datetime.now(timezone.utc)
    mjd_now = datetime_to_mjd(time_now)

    ## Lightcurve
    filters = ['u', 'g', 'q', 'r', 'i', 'z']
    colors = ['darkviolet', 'forestgreen', 'darkorange', 'orangered', 'crimson', 'dimgrey']
    symbols = ['triangle-up', 'diamond-wide', 'circle', 'diamond-tall', 'pentagon', 'star']

    fig = go.Figure()

    for f in filters:
        df_2 = df_bgem_lightcurve.loc[df_bgem_lightcurve['i.filter'] == f]
        df_limiting_mag_2 = df_limiting_mag.loc[df_limiting_mag['filter'] == f]

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
                    opacity         = 0.3,
                    name            = filters[filters.index(f)],
                    hovertemplate   =
                        '<i>MJD: %{x:.3f}</i><br>' +
                        '<i>Limit: %{y:.3f}</i>',
                    hoverlabel      = dict(bgcolor="white")

        ))
        fig.add_vline(x=mjd_now, line_width=1, line_dash="dash", line_color="grey",
                annotation_text="Now ",
                annotation_position="bottom left",
                annotation_font_color = "grey",
                annotation_textangle = 90,)

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
    fig.add_trace(go.Scatter(x=df_bgem_lightcurve['x.ra_psf_d'], y=df_bgem_lightcurve['x.dec_psf_d'], mode='markers', name='Line and Marker', marker_color="red"))
    fig.update_layout(width=350, height=350)

    return fig

class ComingSoonView(TemplateView):
    template_name = 'comingsoon.html'

    def get_context_data(self, **kwargs):
        return {'targets': Target.objects.all()}

class BlackGEMView(TemplateView):
    template_name = 'blackGEM.html'

    def get_context_data(self, **kwargs):
        return {'targets': Target.objects.all()}

def datetime_to_mjd(date):
    '''
    From https://gist.github.com/jiffyclub/1294443
    '''

    year    = date.year
    month   = date.month
    day     = date.day
    hour    = date.hour
    min     = date.minute
    sec     = date.second
    micro   = date.microsecond

    days = sec + (micro / 1.e6)
    days = min + (days / 60.)
    days = hour + (days / 60.)
    days = days / 24.

    day = day + days

    if month == 1 or month == 2:
        yearp = year - 1
        monthp = month + 12
    else:
        yearp = year
        monthp = month

    # this checks where we are in relation to October 15, 1582, the beginning
    # of the Gregorian calendar.
    if ((year < 1582) or
        (year == 1582 and month < 10) or
        (year == 1582 and month == 10 and day < 15)):
        # before start of Gregorian calendar
        B = 0
    else:
        # after start of Gregorian calendar
        A = math.trunc(yearp / 100.)
        B = 2 - A + math.trunc(A / 4.)

    if yearp < 0:
        C = math.trunc((365.25 * yearp) - 0.75)
    else:
        C = math.trunc(365.25 * yearp)

    D = math.trunc(30.6001 * (monthp + 1))

    jd = B + C + D + day + 1720994.5

    mjd = jd - 2400000.5

    return mjd


## =============================================================================
## -------------------------- Codes for the ToO page ---------------------------


def get_ToO_data():
    ToO_filename = "./data/too_data.csv"
    ToO_data = pd.read_csv(ToO_filename)
    return ToO_data

def plot_ToO_timeline():

    ToO_data = get_ToO_data()
    print(ToO_data)
    # ToO_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')
    # ToO_data = ToO_data.rename(columns={
    #     'Date'          : "date_start",
    #     'AAPL.Open'          : "Telescope",
    # })

    print("\nMaking ToO Timeline...")

    time_now = datetime.now()
    mjd_now = datetime_to_mjd(time_now)

    ## Lightcurve
    fig = go.Figure()

    for i in range(len(ToO_data)):
        ToO_data.loc[i, "date_start"]    = datetime.strptime(str(ToO_data["date_start"].iloc[i]), '%Y-%m-%d')
        # ToO_data.loc[i, "date_close"]    = datetime.strptime(str(ToO_data["date_close"].iloc[i]),   '%Y-%m-%d')

    # print(ToO_data["date_start"].iloc[0])
    # print(time_now)

    fig = px.timeline(
        ToO_data,
        x_start='date_start',
        x_end='date_close',
        y = 'Telescope',
        hover_data = ['Name', 'Email', 'Notes'],
        color='Band',
        # color=['red', 'green', 'blue'],
        opacity=0.5,
    )
    # fig = fig.add_trace(
    #     px_timeline.data[0]
    # )
    # fig.layout = px_timeline.layout

    fig.add_vline(x=time_now, line_width=1, line_dash="dash", line_color="grey",)
    fig.add_annotation(x=time_now,text="Now", textangle=90, xshift=-7, showarrow=False, font=dict(color="grey"))
    fig.update_layout(
        # barmode='group',
        height=600,
        # hovermode="x",
        title="Timeline",
        xaxis_title="Date",
        yaxis_title="Telescope",)

    print("ToO Timeline made.")

    return fig



class ToOView(TemplateView):
    template_name = 'too.html'


    def get_context_data(self, **kwargs):

        if not os.path.isfile("./data/too_data.csv"):
            ToO_data = pd.DataFrame(columns = ['Name', 'Email', 'date_start', 'date_close', 'Telescope', 'Band', 'Notes'])
            ToO_data.to_csv("./data/too_data.csv", index=False)

        too_lightcurve = plot(plot_ToO_timeline(), output_type='div')

        context = super().get_context_data(**kwargs)
        context['form'] = ToOForm()  # Add the form to the context
        context['lightcurve'] = too_lightcurve
        # context['csv_data'] = json.dumps(self.get_csv_data())  # Pass the CSV data as JSON
        # context['csv_data'] = get_csv_data().to_json()  # Pass the CSV data as JSON
        self.load_ToO_data()
        return context

    def post(self, request, *args, **kwargs):
        form = ToOForm(request.POST)
        if form.is_valid():
            # Get the data from the form
            name =          form.cleaned_data['name']
            email =         form.cleaned_data['email']
            date_start =    form.cleaned_data['date_start']
            date_close =    form.cleaned_data['date_close']
            telescope =     form.cleaned_data['telescope']
            band =          form.cleaned_data['band']
            notes =         form.cleaned_data['notes']

            ## Standardise the band data:
            band = band.title()
            if band == "Opt": band = "Optical"
            elif band == "X": band = "X-Ray"
            elif band == "Infrared": band = "IR"
            elif band == "Ultraviolet": band = "UV"
            elif band == "Milli": band = "Millimetre"
            elif band == "Micro": band = "Microwave"
            elif band == "Rad": band = "Radio"

            print(name)
            print(email)
            print(date_start)
            print(date_close)
            print(telescope)
            print(band)
            print(notes)

            fileOut = "./data/too_data.csv"

            new_output = pd.DataFrame({
                'Name' : [name],
                'Email' : [email],
                'date_start' : [date_start],
                'date_close' : [date_close],
                'Telescope' : [telescope],
                'Band' : [band],
                'Notes' : [notes],
            })

            if os.path.exists(fileOut):
                output = pd.read_csv(fileOut)
                full_output = pd.concat([output,new_output]).reset_index(drop=True)
                full_output.to_csv(fileOut, index=False)

            else:
                output.to_csv(fileOut, index=False)

            # Redirect after POST to avoid resubmitting form on page refresh
            return HttpResponseRedirect('/ToOs/')  # Redirect to a success page (to be created)
        else:
            # Re-render the form with errors if invalid
            # print(form)
            print("Form not valid!")
            # raise ValidationError("Form not valid!")
            return self.render_to_response(self.get_context_data(form=form))

    def load_ToO_data(self):

        ToO_data = get_ToO_data()

        app = DjangoDash('ToO_Database')

        ToO_data = ToO_data.rename(columns={
            'date_start'          : "Date (Start)",
            'date_close'          : "Date (End)",
        })

        # for i in range(len(ToO_data)):
        #     ToO_data.loc[i, "Date (Start)"]    = datetime.strptime(str(ToO_data["Date (Start)"].iloc[i]), '%Y%m%d')
        #     ToO_data.loc[i, "Date (End)"]      = datetime.strptime(str(ToO_data["Date (End)"].iloc[i]),   '%Y%m%d')

        app.layout = html.Div([
            dag.AgGrid(
                id='ToO_Database',
                rowData=ToO_data.to_dict('records'),
                columnDefs=[{'headerName': col, 'field': col} for col in ToO_data.columns],
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
                style={'height': '300px', 'width': '100%'},  # Set explicit height for the grid
                className='ag-theme-balham'  # Add a theme for better appearance
            ),
            dcc.Store(id='selected-row-data'),  # Store to hold the selected row data
            html.Div(id='output-div'),  # Div to display the information
        ], style={'height': '300px', 'width': '100%'}
        )

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
## ----------------------- Codes for the History pages -------------------------



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

    fileOut = "./data/Recent_BlackGEM_History.csv"
    new_history = pd.DataFrame({'Date' : dates, 'MJD' : mjds, 'Observed' : observed, 'Number_Of_Transients' : transients, 'Number_of_Gaia_Crossmatches' : gaia, 'Number_Of_Extragalactic' : extagalactic})

    output = pd.concat([new_history,previous_history.iloc[:(10-days_since_last_update)]]).reset_index(drop=True)
    output.to_csv(fileOut, index=False)


def blackgem_history():
    '''
    Fetches BlackGEM's history and returns as a pandas dataframe
    '''

    if os.path.isfile("./data/Recent_BlackGEM_History.csv"):
        history = pd.read_csv("./data/Recent_BlackGEM_History.csv")
    else:
        history = pd.DataFrame({
            'Date' : ['2010-01-01'],
            'MJD' : [0],
            'Observed' : ['No'],
            'Number_Of_Transients' : [0],
            'Number_of_Gaia_Crossmatches' : [0],
            'Number_Of_Extragalactic' : [0],
        })
    # print(history)

    return history

def manually_update_history(request):
    # Call your function with the hidden value
    update_history(10)
    # Redirect to the desired page (e.g., 'home' view)
    return redirect('history')

def update_history(days_since_last_update):
    '''
    Fetches BlackGEM's history and returns as several lists, in order to make a table
    '''

    get_recent_blackgem_history(days_since_last_update)

    history = pd.read_csv("./data/Recent_BlackGEM_History.csv")
    # print(history)

    return redirect('history')  # Redirect to the original view if no input

def get_any_nights_sky_plot(night):

    bg = authenticate_blackgem()
    tc = TransientsCatalog(bg)

    time_now = datetime.now(timezone.utc)
    mjd_now = datetime_to_mjd(time_now)

    qu = """\
    select i.id
          , i.filter
          ,i."date-obs"
          ,"tqc-flag"
          ,i.object as tile
          ,i.dec_cntr_deg
          ,i.dec_deg
          ,s.ra_c
          ,s.dec_c
      from image i
          ,skytile s
     where i.object = s.field_id
       AND i."date-obs" BETWEEN '%(time0)s'
                            AND '%(time1)s'
    order by "date-obs" desc
    """

    time_night_start = datetime.strptime(night, '%Y%m%d')
    time_night_end = datetime.strptime(night, '%Y%m%d')+timedelta(days=1)
    time_night_start = time_night_start.strftime("%Y-%m-%d")+" 12:00:00"
    time_night_end = time_night_end.strftime("%Y-%m-%d")+" 12:00:00"

    print("Time start:", time_night_start)
    print("Time close:", time_night_end)

    params = {'time0': time_night_start,
              'time1': time_night_end,
             }
    query = qu % (params)


    l_results = bg.run_query(query)
    df_images = pd.DataFrame(l_results, columns=['id', 'filter', 'date-obs',
                                                 'tqc-flag', 'field',
                                                 'dec_cntr_deg','dec_deg',
                                                 'ra_c', 'dec_c'
                                             ])

    num_fields  = len(df_images)
    num_green   = len(df_images['tqc-flag'][df_images['tqc-flag'] == 'green'])
    num_yellow  = len(df_images['tqc-flag'][df_images['tqc-flag'] == 'yellow'])
    num_orange  = len(df_images['tqc-flag'][df_images['tqc-flag'] == 'orange'])
    num_red     = len(df_images['tqc-flag'][df_images['tqc-flag'] == 'red'])
    print("Number of Fields:", num_fields)
    print("Green fields:    ", num_green)
    print("Yellow fields:   ", num_yellow)
    print("Orange fields:   ", num_orange)
    print("Red fields:      ", num_red)
    field_stats = [num_fields, num_green, num_yellow, num_orange, num_red]

    ## ===== Plotting =====

    ## Create list for RA, Dec, and times
    ra_list = df_images['ra_c']
    dec_list = df_images['dec_c']
    if len(df_images) > 0:
        time_list = np.array([datetime_to_mjd(x) for x in df_images['date-obs']])
        time_list_2 = (time_list-time_list[-1])
        time_list_2 /= time_list_2[0]

        ## Adjust RA/Dec to the right coordinates
        ra = coord.Angle(ra_list, unit=u.degree)
        ra = ra.wrap_at(180*u.degree)
        dec = coord.Angle(dec_list, unit=u.degree)

        ## Plot!
        fig = plt.figure(figsize=(8,5), dpi=110)
        ax = fig.add_subplot(111, projection="mollweide")

        ## Plot each point as an area reoughly the size of the telescope's FOV
        fov_radius = 0.82
        cmap = cm.cool
        this_z = len(ra)+10
        for this_ra, this_dec, this_alpha in zip(ra, dec, time_list_2):
            this_z -= 1
            color = cmap(this_alpha)
            lo_ra   = np.radians(this_ra -fov_radius*u.degree).value
            hi_ra   = np.radians(this_ra +fov_radius*u.degree).value
            lo_dec  = np.radians(this_dec-fov_radius*u.degree).value
            hi_dec  = np.radians(this_dec+fov_radius*u.degree).value
            ax.fill([lo_ra,lo_ra,hi_ra,hi_ra], [lo_dec,hi_dec,hi_dec,lo_dec], color=color, zorder=this_z)  # Adjust color and transparency with 'alpha'
        ax.grid(True)
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")
        ax.xaxis.set_label_position('top')

        # Add a color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])  # Only needed for creating a colorbar
        cb = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05)
        cb.set_label('Time', labelpad=-20)
        tick_positions = [0.08, 0.92]
        tick_labels = [str(df_images['date-obs'].iloc[-1])[:-7] + "\n(First Obs)", str(df_images['date-obs'].iloc[0])[:-7] + "\n(Last Obs)",]
        cb.set_ticks(tick_positions)
        cb.ax.tick_params(bottom = False)
        cb.set_ticklabels(tick_labels)

    else:
        ## Plot!
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111, projection="mollweide")
        ax.grid(True)

    # plt.show()
    # fileOut = "./data/BlackGEM_LastNightsSkymap.png"
    # plt.savefig(fileOut, bbox_inches='tight')
    # plt.close("all")

    # Save the plot to a BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # Encode the image in base64
    image_base64 = base64.b64encode(image_png)
    image_base64 = image_base64.decode('utf-8')

    return field_stats, image_base64


class HistoryView(TemplateView):
    template_name = 'history.html'

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
                print("No transients were recorded by BlackGEM on " + extended_date + " (MJD " + str(mjd) + ").")
                return HttpResponse("No transients were recorded by BlackGEM on " + extended_date + " (MJD " + str(mjd) + ").")

            else:
                return HttpResponse(e)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context['history_daily_text_1'], \
            context['history_daily_text_2'], \
            context['history_daily_text_3'], \
            context['history_daily_text_4'], \
            context['images_daily_text_1'], \
            context['extragalactic_sources_id'], \
            context['transients_filename'], \
            context['gaia_filename'], \
            context['extragalactic_filename'] = history_daily()
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

        field_stats, image_base64 = get_any_nights_sky_plot(yesterday_date_string)
        context['num_fields']       = field_stats[0]
        context['green_fields']     = field_stats[1]
        context['yellow_fields']    = field_stats[2]
        context['orange_fields']    = field_stats[3]
        context['red_fields']       = field_stats[4]
        context['plot_image']       = image_base64

        return context


def handle_input(request):
    '''
    Redirects to a history page about a certain date.
    '''
    user_input = request.GET.get('user_input')
    print(user_input)
    if user_input:
        return redirect(f'/history/{user_input}')
    return redirect('history')  # Redirect to the original view if no input


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


## Function for checking last night's BlackGEM history.
def history_daily():
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
        history_daily_text_1 = "Yes!"

        data_length, num_in_gaia, extragalactic_sources_length, extragalactic_sources, images_urls_sorted, \
            transients_filename, gaia_filename, extragalactic_filename = get_blackgem_stats(yesterday_date)

        ## If there was no data, assume BlackGEM didn't observe.
        if (data_length == "0") and (num_in_gaia == "0") and (extragalactic_sources_length == "0"):
            extragalactic_sources_id    = ""
            history_daily_text_1         = "No transients were recorded by BlackGEM last night (" + extended_yesterday_date + ")"
            history_daily_text_2         = ""
            history_daily_text_3         = ""
            history_daily_text_4         = ""
            images_daily_text_1         = zip([], ["No transients were recorded by BlackGEM last night."])

        else:
            if data_length == "1": data_length_plural = ""; data_length_plural_2 = "s"
            else: data_length_plural = "s"; data_length_plural_2 = "ve"
            if extragalactic_sources_length == "1": extragalactic_sources_plural = ""
            else: extragalactic_sources_plural = "s"

            extragalactic_sources_string = ""
            for source in extragalactic_sources[0]:
                extragalactic_sources_string += source + ", "

            ## Update the most recent row of the recent history
            current_history = blackgem_history()
            if current_history["Date"][0] == extended_yesterday_date:
                current_history.loc[0,"Date"]                          = extended_yesterday_date
                current_history.loc[0,"MJD"]                           = mjd
                current_history.loc[0,"Observed"]                      = "Yes"
                current_history.loc[0,"Number_Of_Transients"]          = int(data_length)
                current_history.loc[0,"Number_of_Gaia_Crossmatches"]   = int(num_in_gaia)
                current_history.loc[0,"Number_Of_Extragalactic"]       = int(extragalactic_sources_length)
                fileOut = "./data/Recent_BlackGEM_History.csv"
                current_history.to_csv(fileOut, index=False)

            print("Running!")

            extragalactic_sources_id = extragalactic_sources[0]
            history_daily_text_2 = "On " + extended_yesterday_date + " (MJD " + str(mjd) + "), BlackGEM observed " + data_length + " transient" + data_length_plural + ", which ha" + data_length_plural_2 + " " + num_in_gaia + " crossmatches in Gaia (radius 1 arcsec)."
            history_daily_text_3 = "BlackGEM recorded pictures of " + extragalactic_sources_length + " possible extragalactic transient" + extragalactic_sources_plural + "."
            history_daily_text_4 = extragalactic_sources_string
            # images_daily_text_1 = zip(images_urls_sorted, extragalactic_sources[0])
            images_daily_text_1 = zip(images_urls_sorted, extragalactic_sources[0], extragalactic_sources[1], extragalactic_sources[2], extragalactic_sources[3], extragalactic_sources[4])
            # images_daily_text_1 = zip(images_urls_sorted, extragalactic_sources[0], extragalactic_sources[1], extragalactic_sources[2], extragalactic_sources[3])


    else:
        extragalactic_sources_id    = ""
        transients_filename         = ""
        gaia_filename               = ""
        extragalactic_filename      = ""
        history_daily_text_1         = "No transients were recorded by BlackGEM last night (" + extended_yesterday_date + ")"
        history_daily_text_2         = ""
        history_daily_text_3         = ""
        history_daily_text_4         = ""
        images_daily_text_1         = zip([], ["No transients were recorded by BlackGEM last night."])


    return history_daily_text_1, history_daily_text_2, history_daily_text_3, history_daily_text_4, images_daily_text_1, extragalactic_sources_id, transients_filename, gaia_filename, extragalactic_filename


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
        observed_string = "No transients were recorded by BlackGEM that night (" + extended_date + ")"
        images_daily_text_1 = zip([], ["No transients were recorded by BlackGEM that night."])
    else:
        observed_string = "Yes!"
        images_daily_text_1 = zip(images_urls_sorted, extragalactic_sources[0], extragalactic_sources[1], extragalactic_sources[2], extragalactic_sources[3], extragalactic_sources[4])
        # images_daily_text_1 = zip(images_urls_sorted, extragalactic_sources[0], extragalactic_sources[1], extragalactic_sources[2], extragalactic_sources[3])

    context['observed']                 = observed_string
    context['images_daily_text_1']      = images_daily_text_1
    context['transients_filename']      = transients_filename
    context['gaia_filename']            = gaia_filename
    context['extragalactic_filename']   = extragalactic_filename

    field_stats, image_base64 = get_any_nights_sky_plot(obs_date)
    context['num_fields']       = field_stats[0]
    context['green_fields']     = field_stats[1]
    context['yellow_fields']    = field_stats[2]
    context['orange_fields']    = field_stats[3]
    context['red_fields']       = field_stats[4]
    context['plot_image'] = image_base64

    return render(request, "history/index.html", context)


def history_to_GEMTOM(request):
    '''
    Imports a target from the History tab
    '''

    id = request.POST.get('id')
    name = request.POST.get('name')
    ra = request.POST.get('ra')
    dec = request.POST.get('dec')

    add_to_GEMTOM(id, name, ra, dec)

    return redirect(reverse('tom_targets:list'))

def TNS_to_GEMTOM(request):
    '''
    Imports a target from the History tab with a TNS ID
    '''

    id          = request.POST.get('id')
    name        = request.POST.get('name')
    ra          = request.POST.get('ra')
    dec         = request.POST.get('dec')
    tns_prefix  = request.POST.get('tns_prefix')
    tns_name    = request.POST.get('tns_name')

    print(tns_prefix + " " + tns_name)

    add_to_GEMTOM(id, name, ra, dec, tns_prefix, tns_name)

    return redirect(reverse('tom_targets:list'))



## =============================================================================
## ----------------------- Codes for the Transients page -----------------------


def blackgem_recent_transients():
    '''
    Fetches BlackGEM's recent transients and returns as a pandas dataframe
    '''

    if os.path.isfile("./data/BlackGEM_Transients_Last30Days.csv"):
        recent_transients = pd.read_csv("./data/BlackGEM_Transients_Last30Days.csv")
    else:
        recent_transients = pd.DataFrame({
            'index_1'       : [0],
            'runcat_id'     : [0],
            'q_xtrsrc'      : [0],
            'iauname'       : [' '],
            'ra'            : [0],
            'dec'           : [0],
            'datapoints'    : [0],
            'within_10min'  : [0],
            'snr_zogy'      : [0],
            'q_min'         : [0],
            'q_max'         : [0],
            'q_rb'          : [0],
            'q_fwhm'        : [0],
            'u_min'         : [0],
            'u_max'         : [0],
            'u_xtrsrc'      : [0],
            'u_rb'          : [0],
            'u_fwhm'        : [0],
            'i_min'         : [0],
            'i_max'         : [0],
            'i_xtrsrc'      : [0],
            'i_rb'          : [0],
            'i_fwhm'        : [0],
            'xtrsrc'        : [0],
            'qui_min'       : [0],
            'fwhm'          : [0],
            'desi_cutout'   : [0],
            'Gmag'          : [0],
            'last_obs'      : ['2010-01-01'],
            'index_2'       : [0],
            'tns'           : [0],
            'ra_gal'        : [0],
            'dec_gal'       : [0],
            'D_gal'         : [0],
            'angDist'       : [0],
            'metric'        : [0],
            'ra_sml'        : [0],
            'dec_sml'       : [0],
            'snr_zogy_sml'  : [0],
            'iauname_short' : [0],
            'q_min_sml'     : [0],
            'u_min_sml'     : [0],
            'i_min_sml'     : [0],
            'q_max_sml'     : [0],
            'u_max_sml'     : [0],
            'i_max_sml'     : [0],
            'q_dif'         : [0],
            'u_dif'         : [0],
            'i_dif'         : [0],
        })

        recent_transients.to_csv("./data/BlackGEM_Transients_Last30Days.csv", index=False)

    return recent_transients


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

        if oldest_date == '2010-01-01':
            update_data = True
            print("Finding data for the first time.")
            old_date = "a long time ago, and will not be saved.."
        else:
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
                    old_date_index = len(previous_history)
                    old_date = "a long time ago, and will not be saved.."
                    break

            print("Data found from " + old_date + ".")
        ## When we save data, save only up to this index.

        ## If there's new data...
        if data_list:
            ## ...combine it with the old.
            if old_date == "a long time ago, and will not be saved..":
                df = df_new
            else:
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


def search_BGEM_ID(request):
    '''
    Redirects to a history page about a certain date.
    '''
    user_input = request.GET.get('user_input')
    print(user_input)
    if user_input:
    #     return redirect(f'/GEMTOM/transients/{user_input}')
    # return redirect('/GEMTOM/transients')  # Redirect to the original view if no input
        return redirect(f'/transients/{user_input}')
    return redirect('/transients')  # Redirect to the original view if no input


def transient_cone_search(ra, dec, radius=60):
    bg = authenticate_blackgem()

    tc = TransientsCatalog(bg)

    # ra  = 73.69518912800072
    # dec = -18.203891208244954

    ## Cone Search
    bg_columns, bg_results = tc.conesearch(ra, dec, radius_arcsec=radius)
    df_bg_cone = pd.DataFrame(bg_results, columns=bg_columns)

    return df_bg_cone


def get_limiting_magnitudes_from_BGEM_ID(blackgem_id):

    bg = authenticate_blackgem()


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

    # print("\n\nLimitingBark!")
    # print(l_results)
    df_limiting_mag = pd.DataFrame(l_results, columns=['id','date_obs','mjd','limiting_mag','filter'])
    # print(df_limiting_mag)

    return(df_limiting_mag)


def get_lightcurve_from_BGEM_ID(transient_id):

    print("Getting lightcurve for transient ID " + str(transient_id) + "...")

    bg = authenticate_blackgem()


    # Create an instance of the Transients Catalog
    tc = TransientsCatalog(bg)
    df_limiting_mag = get_limiting_magnitudes_from_BGEM_ID(transient_id)

    # Get all the associated extracted sources for this transient
    # Note that you can specify the columns yourself, but here we use the defaults
    bg_columns, bg_results = tc.get_associations(transient_id)
    df_bgem_lightcurve = pd.DataFrame(bg_results, columns=bg_columns)

    print(df_bgem_lightcurve.iloc[0])
    print(df_limiting_mag.iloc[0])

    ## Remove all points in the limiting_mag lightcurve that have detections.
    # num_removed = 0
    for i in range(len(df_limiting_mag["date_obs"])):
        if df_limiting_mag["date_obs"][i] in df_bgem_lightcurve['i."date-obs"'].unique():
            df_limiting_mag = df_limiting_mag.drop([i])
            # num_removed += 1
    # print(num_removed, "points removed.")
    # print(len(df_bgem_lightcurve), "points in lightcurve.")


    return df_bgem_lightcurve, df_limiting_mag


def BGEM_to_GEMTOM_photometry(df_bg_assocs):

    gemtom_photometry = pd.DataFrame({
        'mjd' : df_bg_assocs["i.\"mjd-obs\""],
        'mag' : df_bg_assocs["x.mag_zogy"],
        'magerr' : df_bg_assocs["x.magerr_zogy"],
        'filter' : df_bg_assocs["i.filter"],
    })

    return gemtom_photometry

def BGEM_to_GEMTOM_photometry_2(df_bgem_lightcurve, df_limiting_mag=[]):

    print("df_bgem_lightcurve:")
    print(df_bgem_lightcurve)
    print("df_limiting_mag:")
    print(df_limiting_mag)

    gemtom_photometry = pd.DataFrame({
        'mjd' : df_bgem_lightcurve["i.\"mjd-obs\""],
        'mag' : df_bgem_lightcurve["x.mag_zogy"],
        'magerr' : df_bgem_lightcurve["x.magerr_zogy"],
        'limit' : [''] * len(df_bgem_lightcurve),
        'filter' : df_bgem_lightcurve["i.filter"],
    })

    print("GEMTOM Photometry:")
    print(gemtom_photometry)

    if len(df_limiting_mag) > 0:
        gemtom_limiting_photometry = pd.DataFrame({
            'mjd' : df_limiting_mag["mjd"],
            'mag' : [''] * len(df_limiting_mag),
            'magerr' : [''] * len(df_limiting_mag),
            'limit' : df_limiting_mag["limiting_mag"],
            'filter' : df_limiting_mag["filter"],
        })


        print("GEMTOM Limiting Photometry:")
        print(gemtom_limiting_photometry)

        gemtom_photometry = pd.concat([gemtom_photometry,gemtom_limiting_photometry]).reset_index(drop=True)

    return gemtom_photometry

## =========================
## ----- TNS Functions -----

from collections import OrderedDict

## Get TNS token
print("Loading dotenv...")
from dotenv import load_dotenv, dotenv_values
load_dotenv()
print(dotenv_values())
print("Dotenv loaded.")

TNS                 = "www.wis-tns.org"
url_tns_api         = "https://" + TNS + "/api/get"

TNS_BOT_ID          = "187806"
TNS_BOT_NAME        = "BotGEM"
TNS_API_KEY         = os.getenv('TNS_API_TOKEN', 'TNS_API_TOKEN not set')

def set_bot_tns_marker():
    tns_marker = 'tns_marker{"tns_id": "' + str(TNS_BOT_ID) + '", "type": "bot", "name": "' + TNS_BOT_NAME + '"}'
    return tns_marker

def search(search_obj):
    search_url = url_tns_api + "/search"
    tns_marker = set_bot_tns_marker()
    headers = {'User-Agent': tns_marker}
    json_file = OrderedDict(search_obj)
    search_data = {'api_key': TNS_API_KEY, 'data': json.dumps(json_file)}
    response = requests.post(search_url, headers = headers, data = search_data)
    print("TNS API Key:", TNS_API_KEY)

    return response

def get(get_obj):
    get_url = url_tns_api + "/object"
    tns_marker = set_bot_tns_marker()
    headers = {'User-Agent': tns_marker}
    json_file = OrderedDict(get_obj)
    get_data = {'api_key': TNS_API_KEY, 'data': json.dumps(json_file)}
    response = requests.post(get_url, headers = headers, data = get_data)
    return response

def format_to_json(source):
    parsed = json.loads(source, object_pairs_hook = OrderedDict)
    result = json.dumps(parsed, indent = 4)
    return result

##  --- TNS functions for GEMTOM ---
def get_tns_from_ra_dec(ra, dec, radius):

    search_obj          = [("ra", str(ra)), ("dec", str(dec)), ("radius", str(radius)), ("units", "arcsec"), ("objname", ""),
                       ("objname_exact_match", 0), ("internal_name", ""),
                       ("internal_name_exact_match", 0), ("objid", ""), ("public_timestamp", "")]

    response = search(search_obj)
    json_data = format_to_json(response.text)
    # print(json_data)
    json_data = json.loads(json_data)
    if json_data["id_code"] == 429:
        return "Too many requests!"
    elif json_data["id_code"] == 401:
        return "Unauthorised!"
    else:
        print("ID Code:", json_data["id_code"])
        print("ID Code:", json_data["id_code"])
        print(json_data.keys())
        print(json_data["data"]["reply"])
        print(len(json_data["data"]["reply"]))
        return json_data

def get_ra_dec_from_tns(tns_object_name):
    get_obj             = [("objname", tns_object_name), ("objid", ""), ("photometry", ""), ("spectra", "")]
    response = get(get_obj)
    json_data = format_to_json(response.text)
    json_data = json.loads(json_data)
    # print(json_data)
    if json_data["id_code"] == 429:
        return "Too many requests!"
    else:
        return json_data

## ----- TNS Functions -----
## =========================


def BGEM_ID_View(request, bgem_id):
    '''
    Displays data of a certain transient
    '''

    df_bgem_lightcurve, df_limiting_mag = get_lightcurve_from_BGEM_ID(bgem_id)
    # print(df_bgem_lightcurve)
    # print(df_bgem_lightcurve.columns)

    response = "You're looking at BlackGEM transient %s."

    ## --- Location on Sky ---
    fig = plot_BGEM_location_on_sky(df_bgem_lightcurve)
    location_on_sky = plot(fig, output_type='div')

    # print(df_bgem_lightcurve['x.ra_psf_d'])
    # print(df_bgem_lightcurve['x.dec_psf_d'])

    ## --- Lightcurve ---
    fig = plot_BGEM_lightcurve(df_bgem_lightcurve, df_limiting_mag)
    lightcurve = plot(fig, output_type='div')

    # Pass the plot_div to the template
    # return render(request, 'transient/index.html')

    ## Get the name, ra, and dec:
    bg = authenticate_blackgem()

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

    ## 6385260

    l_results = bg.run_query(query)
    source_data = pd.DataFrame(l_results, columns=['id','iau_name','ra_deg','dec_deg'])
    if len(source_data) == 0:

        context = {
            "bgem_id"           : bgem_id,
        }

        return render(request, "transient/index_2.html", context)
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
            style={'height': '550px', 'width': '100%'},  # Set explicit height for the grid
            className='ag-theme-balham'  # Add a theme for better appearance
        ),
        dcc.Store(id='selected-row-data'),  # Store to hold the selected row data
        html.Div(id='output-div'),  # Div to display the information
    ], style={'height': '550px', 'width': '100%'}
    )

    ## TNS:
    search_radius = 1000
    tns_data = get_tns_from_ra_dec(ra, dec, search_radius)
    if tns_data == "Too many requests!":
        tns_text = "Too many TNS requests. Please check later."
        tns_list = []
    elif tns_data == "Unauthorised!":
        tns_text = "Note: TNS Unauthorised. Please check."
        tns_list = []
    else:
        tns_reply = tns_data["data"]["reply"]
        tns_reply_length = len(tns_data["data"]["reply"])
        if tns_reply_length == 0:
            tns_text = "No TNS object found within " + str(search_radius) + " arcseconds."
            tns_list = []
        else:
            tns_text = "TNS results within " + str(search_radius) + " arcseconds"
            tns_list = tns_reply

    tns_object_names    = []
    tns_object_ids      = []
    for tns_object in tns_list:
        tns_object_names.append(tns_object["objname"])
        tns_object_ids.append(tns_object["objid"])
    # print(tns_object_names)

    # tns_objects_data = []
    tns_object_prefix       = []
    tns_object_ra           = []
    tns_object_dec          = []
    tns_object_sep          = []
    tns_object_internalname = []
    bgem_object_radec       = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
    for tns_object_name in tns_object_names:
        # tns_object_ra, tns_object_dec = get_ra_dec_from_tns(tns_object_name)
        tns_object_data = get_ra_dec_from_tns(tns_object_name)
        if tns_data == "Too many requests!":
            tns_object_data = "Too many TNS requests. Please check later."
            break
        else:
            # print(tns_object_data)
            # print(tns_object_data["data"])
            # print(tns_object_data["data"]["reply"])

            ## Get RA and Dec, and find distance to our current target.
            this_object_ra      = tns_object_data["data"]["reply"]["radeg"]
            this_object_dec     = tns_object_data["data"]["reply"]["decdeg"]
            this_object_radec   = SkyCoord(this_object_ra*u.deg, this_object_dec*u.deg, frame='icrs')
            this_object_sep     = bgem_object_radec.separation(this_object_radec)

            # print(this_object_radec)
            # print(bgem_object_radec)
            # print(this_object_sep.arcsecond)

            ## Save the individual details
            tns_object_prefix.append(tns_object_data["data"]["reply"]["name_prefix"])
            tns_object_ra.append(this_object_ra)
            tns_object_dec.append(this_object_dec)
            tns_object_sep.append(this_object_sep.arcsecond)
            tns_object_internalname.append(tns_object_data["data"]["reply"]["internal_names"])
    # print(tns_objects_data)
    # print(tns_object_ra)
    # print(tns_object_dec)
    print(tns_object_sep)
    # tns_objects_data = zip(tns_object_names, tns_object_ra, tns_object_dec)

    # ## If object is close enough...
    # close_enough_sep = 40
    # if tns_object_sep:
    #     if np.min(tns_object_sep) < close_enough_sep:
    #         tns_nearby = "TNS Object within " + str(close_enough_sep) + " arcseconds!"
    #     else:
    #         tns_nearby = ""
    # else:
    #     tns_nearby = ""

    tns_objects_data = pd.DataFrame({
        'ObjID': tns_object_ids,
        'Prefix': tns_object_prefix,
        'Name': tns_object_names,
        'RA': tns_object_ra,
        'Dec': tns_object_dec,
        'Internal_Name': tns_object_internalname,
        'Separation': tns_object_sep,
    })

    tns_objects_potential = tns_objects_data.loc[tns_objects_data['Separation'] < 10]
    if len(tns_objects_potential) > 0:
        tns_flag = True
        tns_flag_prefix = tns_objects_potential["Prefix"].iloc[0]
        tns_flag_name = tns_objects_potential["Name"].iloc[0]
        tns_flag_sep = round(tns_objects_potential["Separation"].iloc[0], 2)
    else:
        tns_flag = False
        tns_flag_prefix = ""
        tns_flag_name = ""
        tns_flag_sep = ""

    ## --- Find the image ---
    print("Getting image...")
    print(os.getcwd())
    if tns_flag:
        file_name = "../" + get_transient_image(bgem_id, ra, dec, df_bgem_lightcurve,
            tns_objects_potential["RA"].iloc[0], tns_objects_potential["Dec"].iloc[0]
            )
    else:
        file_name = "../" + get_transient_image(bgem_id, ra, dec, df_bgem_lightcurve)

    print("Image name:", file_name)




    tns_objects_data['Name'] = tns_objects_data['Name'].apply(lambda x: f'[{x}](https://www.wis-tns.org/object/{x})')

    # TNS_object_url = "https://sandbox.wis-tns.org/object/2024ot"

    tns_objects_data = tns_objects_data.sort_values(by=['Separation'])

    ## TNS Dash App
    app = DjangoDash('TNS_Sources')
    app.layout = html.Div([
        dag.AgGrid(
            id='tns-grid',
            rowData=tns_objects_data.to_dict('records'),
            columnDefs=[
                        {'headerName': 'ObjID',         'field':  'ObjID'},
                        {'headerName': 'Prefix',        'field':  'Prefix'},
                        {'headerName': 'Name', 'field':  'Name', 'cellRenderer': 'markdown'},
                        # {'headerName': 'Name',          'field':  'Name'},
                        {'headerName': 'RA',            'field':  'RA'  },
                        {'headerName': 'Dec',           'field':  'Dec' },
                        {'headerName': 'Internal Name', 'field':  'Internal_Name' },
                        {'headerName': 'Separation (arcsec)',    'field':  'Separation' },
            ],
            defaultColDef={
                'sortable': True,
                'filter': True,
                'resizable': True,
                'editable': True,
            },
            columnSize="autoSize",
            dashGridOptions={
                "skipHeaderOnAutoSize": False,
                "rowSelection": "single",
            },
            style={'height': '200px', 'width': '100%'},  # Set explicit height for the grid
            className='ag-theme-balham'  # Add a theme for better appearance
        ),
        dcc.Store(id='selected-row-data'),  # Store to hold the selected row data
        html.Div(id='output-div'),  # Div to display the information
    ], style={'height': '200px', 'width': '100%'}
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
        "tns_flag"          : tns_flag,
        "tns_flag_prefix"   : tns_flag_prefix,
        "tns_flag_name"     : tns_flag_name,
        "tns_flag_sep"      : tns_flag_sep,
        "tns_data"          : tns_data,
        "tns_text"          : tns_text,
        "tns_list"          : tns_list,
        "image_name"        : file_name
        # "tns_nearby"        : tns_nearby,
        # "tns_objects_data"  : tns_objects_data,
    }

    print(context["image_name"])

    return render(request, "transient/index.html", context)



## =============================================================================
## ------------------ Codes for the Unified Transients page --------------------

def get_transient_image(bgem_id, ra, dec, df_bgem_lightcurve = False, tns_ra=False, tns_dec=False):

    # Instantialte the BlackGEM object, with a connection to the database
    bg = authenticate_blackgem()

    desimoc = MOC.from_fits('./data/MOC_DESI-Legacy-Surveys_DR10.fits')
    coords = SkyCoord(ra,dec,unit='deg',frame='icrs')
    indesi = desimoc.contains_lonlat(coords.ra,coords.dec)

    ra  = coords.ra.value
    dec = coords.dec.value
    file_name = "./data/" + str(bgem_id) + "/ra%s"%str(ra)+"dec%s_cutout.png"%str(dec)

    ## Individual Detections
    ra_2    = df_bgem_lightcurve["x.ra_psf_d"]
    dec_2   = df_bgem_lightcurve["x.dec_psf_d"]
    coords_2 = SkyCoord(ra_2,dec_2,unit='deg',frame='icrs')
    # print(coords.separation(coords_2).arcsecond)

    pa = coords.position_angle(coords_2)
    sep = coords.separation(coords_2)

    ra_arcsecond_sep    = sep.arcsecond*np.sin(pa)
    dec_arcsecond_sep   = sep.arcsecond*np.cos(pa)


    if not os.path.exists("./data/" + str(bgem_id) + "/"):
        os.makedirs("./data/" + str(bgem_id) + "/")

    # if os.path.isfile(file_name):
    #     return file_name

    if indesi:
        pixel_scale = 0.262/2
        print("Field in DESI...")
        url = 'https://www.legacysurvey.org/viewer/cutout.jpg?ra=%s&dec=%s&layer=ls-dr10-grz&zoom=15'%(ra,dec)
    elif float(dec)>-30:
        pixel_scale = 0.257
        print("Field in PS1...")
        image_index_url_red = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra=%s&dec=%s&filters=i'%(ra, dec)
        image_index_url_green = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra=%s&dec=%s&filters=r'%(ra, dec)
        image_index_url_blue = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra=%s&dec=%s&filters=g'%(ra, dec)

        urlretrieve(image_index_url_red, '/tmp/image_index_red.txt')
        urlretrieve(image_index_url_green, '/tmp/image_index_green.txt')
        urlretrieve(image_index_url_blue, '/tmp/image_index_blue.txt')

        ix_red = np.genfromtxt('/tmp/image_index_red.txt', names=True, dtype=None, encoding='utf-8')
        ix_green = np.genfromtxt('/tmp/image_index_green.txt', names=True, dtype=None, encoding='utf-8')
        ix_blue = np.genfromtxt('/tmp/image_index_blue.txt', names=True, dtype=None, encoding='utf-8')

        url = "http://ps1images.stsci.edu/cgi-bin/fitscut.cgi?red=%s&green=%s&blue=%s&filetypes=stack&auxiliary=data&size=%d&ra=%s&dec=%s&output_size=256"%\
        (ix_red["filename"], ix_green["filename"], ix_blue["filename"], 240, ra, dec)
    else:
        print("Field in DSS2...")
        pixel_scale = 1
        url = 'https://archive.stsci.edu/cgi-bin/dss_search?v=poss2ukstu_red&r=%s&d=%s&e=J2000&h=4&w=4&f=gif'%(ra,dec)


    ra_pixel_sep = -ra_arcsecond_sep/pixel_scale
    dec_pixel_sep = dec_arcsecond_sep/pixel_scale

    if tns_ra and tns_dec:
        coords_3 = SkyCoord(tns_ra,tns_dec,unit='deg',frame='icrs')
        # print(coords.separation(coords_2).arcsecond)

        tns_pa = coords.position_angle(coords_3)
        tns_sep = coords.separation(coords_3)

        tns_ra_arcsecond_sep    = tns_sep.arcsecond*np.sin(tns_pa)
        tns_dec_arcsecond_sep   = tns_sep.arcsecond*np.cos(tns_pa)

        tns_ra_pixel_sep = -tns_ra_arcsecond_sep/pixel_scale
        tns_dec_pixel_sep = tns_dec_arcsecond_sep/pixel_scale


    cutout = requests.get(url, stream = True)

    if cutout.status_code == 200:
        with open(file_name,'wb') as f:
            shutil.copyfileobj(cutout.raw, f)
        print('Image successfully Downloaded: ',file_name)

        ## Draw crosshairs
        with Image.open(file_name) as im:
            draw = ImageDraw.Draw(im)

            ## Crosshairs - All units in percentages
            gap = 8
            size = 25
            # draw.line((im.size[0]/2, im.size[1]*(100+gap)/200, im.size[0]/2, im.size[1]*(100+size)/200), fill="red", width=2)
            # draw.line((im.size[0]*(100+gap)/200, im.size[1]/2, im.size[1]*(100+size)/200, im.size[1]/2), fill="red", width=2)

            ## Scale
            draw.line((im.size[0]/2, im.size[1]/2+10/pixel_scale+5, im.size[0]/2+10/pixel_scale, im.size[1]/2+10/pixel_scale+5), fill="white", width=2)
            draw.text((im.size[0]/2, im.size[1]/2+10/pixel_scale+10), "10 arcseconds", fill="white")

            ## Circle
            draw.arc([(im.size[0]/2-10/pixel_scale, im.size[0]/2-10/pixel_scale), (im.size[0]/2+10/pixel_scale, im.size[0]/2+10/pixel_scale)], start=0, end=359, fill="white")

            ## TNS Source?
            if tns_ra and tns_dec:

                tns_ra_pixel    = im.size[0]/2-tns_ra_pixel_sep
                tns_dec_pixel   = im.size[0]/2-tns_dec_pixel_sep

                draw.rectangle([(tns_ra_pixel-1, tns_dec_pixel-1), (tns_ra_pixel+1, tns_dec_pixel+1)], fill="lightgreen")

                draw.line((tns_dec_pixel, tns_ra_pixel-10, tns_dec_pixel, tns_ra_pixel+10), fill="lightgreen", width=2)
                draw.line((tns_dec_pixel-10, tns_ra_pixel, tns_dec_pixel+10, tns_ra_pixel), fill="lightgreen", width=2)
                draw.text((im.size[0]/2, im.size[1]/2+10/pixel_scale+34), "TNS Source", fill="lightgreen")

            ## All BlackGEM Detections
            draw.text((im.size[0]/2, im.size[1]/2+10/pixel_scale+22), "BlackGEM Detections", fill="red")
            for this_ra, this_dec in zip(ra_pixel_sep, dec_pixel_sep):
                # print("Detection!")
                detections_x = im.size[0]/2-this_ra
                detections_y = im.size[1]/2-this_dec

                # print(detections_x, detections_y)

                draw.point([(detections_x, detections_y)], fill="red")


            # im.save(file_name[:-4] + "_cross.png")
            im.save(file_name)

    else:
        print('Image Couldn\'t be retrieved')
        print(file_name)

    return file_name

def check_blackgem_recent_transients(recent_transients):
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
        return blackgem_recent_transients()
    else:
        return recent_transients


class UnifiedTransientsView(TemplateView):
    template_name = 'unified_transients.html'

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


    app = DjangoDash("resizable_container")

    app.layout = html.Div([
        html.Div(id='resizable-container', children=[
            dcc.Graph(
                id='example-graph',
                figure={
                    'data': [{'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'Sample'}],
                    'layout': {'title': 'Sample Graph'}
                }
            )
        ], style={'width': '50%', 'height': '500px', 'border': '1px solid black', 'transition': 'all 0.5s'}),

        html.Button('Resize Container', id='resize-button')
    ])

    @app.callback(
        Output('resizable-container', 'style'),
        Input('resize-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def resize_container(n_clicks):
        if n_clicks % 2 == 1:
            return {'width': '80%', 'height': '700px', 'border': '1px solid black', 'transition': 'all 0.5s'}
        else:
            return {'width': '50%', 'height': '500px', 'border': '1px solid black', 'transition': 'all 0.5s'}




    app = DjangoDash('ConeSearchTable')

    style_dict = {
      "font-size": "16px",
      "margin-right": "10px",
      'width': '200px',
      'height': '30px',
      # 'scale':'2'
    }

    button_style_dict = {
        "font-size": "16px",
        "margin-right": "10px",
        "padding": "7px 24px",
        # 'border-radius': '5px',
        'color':'#027bff',
        'background-color': '#ffffff',
        'border': '2px solid #027bff',
        'border-radius': '5px',
        'cursor': 'pointer',
    }

    app.layout = html.Div([
        html.Div([
            # html.A("Link to external site", href='https://plot.ly', target="_blank"),
            dcc.Input(id='ra-input',        type='number', min=0, max=360,  placeholder=' RA (deg)',        style=style_dict),
            dcc.Input(id='dec-input',       type='number', min=-90, max=90,   placeholder=' Dec (deg)',       style=style_dict),
            dcc.Input(id='radius-input',    type='number', min=0, max=600,  placeholder=' Radius (arcseconds)',    style=style_dict),
            html.Button('Search', id='submit-button', n_clicks=0, style=button_style_dict),
            # html.Button('Search', id='submit-button', n_clicks=0, style={"font-size": "16px","margin-right": "10px",})
        ], style={'margin-bottom': '20px', "text-align":"center"}),
        html.Div(id='results-container', children=[]),
        html.Div(id='redirect-trigger', style={'display': 'none'})
    ])

    # Define the callback to update the table based on input coordinates
    @app.callback(
        Output('results-container', 'children'),
        Input('submit-button', 'n_clicks'),
        State('ra-input', 'value'),
        State('dec-input', 'value'),
        State('radius-input', 'value'),
        prevent_initial_call=False
    )
    def update_results(n_clicks, ra, dec, radius):
        print(ra, dec, radius)
        if ra is not None and dec is not None:
            df = transient_cone_search(ra, dec, radius)
            if len(df) == 0:
                return html.Div([
                            html.P("RA: " +str(ra)),
                            html.P("Dec: " +str(dec)),
                            html.P("Radius: " +str(radius)),
                            html.Em("No targets found"),
                        ], style={"text-align":"center", "font-size": "18px", "font-family":"sans-serif"}
                    )
            else:
                df['id'] = df['id'].apply(lambda x: f'[{x}](/transients/{x})')
                # df['id'] = df['id'].apply(lambda x: f'<a href="/transients/{x}" target="_blank">{x}</a>)')
                message = html.Div([html.P(html.Em("Ctrl/Cmd-click on links to open the transient in a new tab"))], style={"text-align":"center", "font-size": "18px", "font-family":"sans-serif"})
                table = dag.AgGrid(
                    id='results-table',
                    columnDefs=[
                        {'headerName': 'ID', 'field': 'id', 'cellRenderer': 'markdown'},
                        # {'headerName': 'ID', 'field': 'id', 'cellRenderer': 'htmlCellRenderer'},
                        # {'headerName': 'ID', 'field': 'id'},
                        {'headerName': 'Datapoints', 'field': 'datapoints'},
                        {'headerName': 'RA', 'field': 'ra_deg'},
                        {'headerName': 'Dec', 'field': 'dec_deg'},
                        {'headerName': 'Dist (")', 'field': 'distance_arcsec'},
                    ],
                    rowData=df.to_dict('records'),
                    dashGridOptions={"rowSelection": "single"},
                    style={'height': '200px', 'width': '100%'},
                    className='ag-theme-balham'  # Add a theme for better appearance
                )
                return message, table
        else:
            message = html.Div([html.P(html.Em("Enter co-ordinates to search."))], style={"text-align":"center", "font-size": "18px", "font-family":"sans-serif", "color":"grey"})

            return message

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        ## --- Update Recent Transients ---
        recent_transients = blackgem_recent_transients()
        recent_transients = check_blackgem_recent_transients(recent_transients)
        # dates = list(recent_transients.last_obs)
        # dates = [this_date[0:4] + this_date[5:7] + this_date[8:] for this_date in dates]
        #
        # ## --- Check to see if the recent history is up to date. If not, update.
        # ## Get yesterday's date...
        # yesterday_date = datetime.today() - timedelta(1)
        # yesterday_date = yesterday_date.replace(hour=0, minute=0, second=0, microsecond=0)
        #
        # ## Get the most recent date...
        # most_recent_date = datetime.strptime(dates[0], "%Y%m%d")
        #
        # ## Find the difference...
        # difference = yesterday_date - most_recent_date
        # days_since_last_update = difference.days
        # print("Days since last update:", days_since_last_update)
        #
        # yesterday_date_string = yesterday_date.strftime("%Y%m%d")
        #
        # if days_since_last_update > 0:
        #     get_recent_blackgem_transients(days_since_last_update)
        #     recent_transients = blackgem_recent_transients()
        #     dates = list(recent_transients.last_obs)
        #     dates = [this_date[0:4] + this_date[5:7] + this_date[8:] for this_date in dates]

        return context





    ## ===== Plot the transients from the past 30 days =====

    ## --- Step 1: The 'Recent Transients' Table ---
    ## Uses a Dash AG Grid

    # Initialize the Dash app
    app = DjangoDash('RecentTransients')

    # Read CSV data
    df = blackgem_recent_transients()
    df = check_blackgem_recent_transients(df)

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
            html.Em(html.P("Select a row")), style={"text-align":"center", "font-size": "18px", "font-family":"sans-serif", "color":"grey"}
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
                    html.A("GEMTOM page for this transient", href='/transients/'+str(row_data['runcat_id']), target="_blank", style={'text-decoration':'None', "font-style": "italic"}),
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

def search_BGEM_ID_for_live_feed(request):
    '''
    Redirects to a live feed page for a certain transient
    '''
    user_input = request.GET.get('user_input')
    print(user_input)
    if user_input:
        return redirect(f'/live_feed/{user_input}')
    return redirect('live_feed')  # Redirect to the original view if no input

def fetch_random_data(n):
    print("Fetching New Data!")
    df = pd.DataFrame({
        'Time': range(n),
        'Mag': range(n, n*2)
    })
    return df

def datetime_to_mjd(date):
    '''
    From https://gist.github.com/jiffyclub/1294443
    '''

    year    = date.year
    month   = date.month
    day     = date.day
    hour    = date.hour
    min     = date.minute
    sec     = date.second
    micro   = date.microsecond

    days = sec + (micro / 1.e6)
    days = min + (days / 60.)
    days = hour + (days / 60.)
    days = days / 24.

    day = day + days

    if month == 1 or month == 2:
        yearp = year - 1
        monthp = month + 12
    else:
        yearp = year
        monthp = month

    # this checks where we are in relation to October 15, 1582, the beginning
    # of the Gregorian calendar.
    if ((year < 1582) or
        (year == 1582 and month < 10) or
        (year == 1582 and month == 10 and day < 15)):
        # before start of Gregorian calendar
        B = 0
    else:
        # after start of Gregorian calendar
        A = math.trunc(yearp / 100.)
        B = 2 - A + math.trunc(A / 4.)

    if yearp < 0:
        C = math.trunc((365.25 * yearp) - 0.75)
    else:
        C = math.trunc(365.25 * yearp)

    D = math.trunc(30.6001 * (monthp + 1))

    jd = B + C + D + day + 1720994.5

    mjd = jd - 2400000.5

    return mjd

def find_latest_BlackGEM_field():
    user_home = str(Path.home())
    bg = authenticate_blackgem()


    ## Get most recent pointing.
    query = """\
    select i.id
          , i.filter
          ,i."date-obs"
          ,"tqc-flag"
          ,i.object as tile
          ,i.dec_cntr_deg
          ,i.dec_deg
          ,s.ra_c
          ,s.dec_c
      from image i
          ,skytile s
     where i.object = s.field_id
    order by "date-obs" desc
    limit 20;
    """
    l_results = bg.run_query(query)
    df_images = pd.DataFrame(l_results, columns=['id', 'filter', 'date-obs',
                                                 'tqc-flag', 'field',
                                                 'dec_cntr_deg','dec_deg',
                                                 'ra_c', 'dec_c'
                                             ])
    # df_images.round({'s-seeing': 2})
    # print(df_images)
    # print(list(df_images['dec_cntr_deg']))
    # list_min = 0
    # list_max = 10
    # print(df_images['dec_cntr_deg'].iloc[   list_min:list_max])
    # print(df_images['dec_ref_dms'].iloc[    list_min:list_max])
    # print(df_images['dec_deg'].iloc[        list_min:list_max])
    print(df_images["date-obs"].iloc[0])
    most_recent_image_time = df_images["date-obs"].iloc[0]
    most_recent_image_mjd = datetime_to_mjd(most_recent_image_time)
    print(most_recent_image_mjd)
    time_now = datetime.now(timezone.utc)
    mjd_now = datetime_to_mjd(time_now)
    hours_since_last_field      = ((mjd_now - most_recent_image_mjd)*24)
    minutes_since_last_field    = ((mjd_now - most_recent_image_mjd)*24*60)#-(hours_since_last_field*60)
    seconds_since_last_minute   = np.floor(((minutes_since_last_field-np.floor(minutes_since_last_field))*60))
    print("Seconds", seconds_since_last_minute)

    print("")
    print("======")
    print("Time since last field: %.2f" % minutes_since_last_field, "minutes.")
    print("Last field observed:", df_images["field"].iloc[0])
    print("RA: %.3f" % df_images["ra_c"].iloc[0], "; Dec: %.3f" % df_images["dec_c"].iloc[0])

    context = {}

    context['BlackGEM_hours']   = "%.0f" % np.floor(hours_since_last_field)
    context['BlackGEM_hrplur']  = "s"
    context['BlackGEM_minutes'] = "%.0f" % np.floor(minutes_since_last_field)
    context['BlackGEM_minplur'] = "s"
    context['BlackGEM_seconds'] = "%.0f" % seconds_since_last_minute
    context['BlackGEM_secplur'] = "s"
    if hours_since_last_field == 1:     context['BlackGEM_hrplur']  = ""
    if minutes_since_last_field == 1:   context['BlackGEM_minplur'] = ""
    if seconds_since_last_minute == 1:  context['BlackGEM_secplur'] = ""
    context['BlackGEM_fieldid'] = df_images["field"].iloc[0]
    context['BlackGEM_RA']      = df_images["ra_c"].iloc[0]
    context['BlackGEM_Dec']     = df_images["dec_c"].iloc[0]
    if minutes_since_last_field >= 60:
        context['BlackGEM_message'] = "BlackGEM is not observing"
        context['BlackGEM_colour']  = "Black"
    if minutes_since_last_field >= 30:
        context['BlackGEM_message'] = "BlackGEM is probably not observing."
        context['BlackGEM_colour']  = "Black"
    else:
        context['BlackGEM_message'] = "BlackGEM is observing!"
        context['BlackGEM_colour']  = "MediumSeaGreen"

    BlackGEM_minutes = context['BlackGEM_minutes']
    BlackGEM_minplur = context['BlackGEM_minplur']
    BlackGEM_seconds = context['BlackGEM_seconds']
    BlackGEM_secplur = context['BlackGEM_secplur']
    BlackGEM_fieldid = context['BlackGEM_fieldid']
    BlackGEM_RA         = context['BlackGEM_RA']
    BlackGEM_Dec        = context['BlackGEM_Dec']

    return BlackGEM_minutes, BlackGEM_minplur, BlackGEM_seconds, BlackGEM_secplur, BlackGEM_fieldid, BlackGEM_RA, BlackGEM_Dec

def update_latest_BlackGEM_Field(request):

    BlackGEM_minutes, BlackGEM_minplur, BlackGEM_seconds, BlackGEM_secplur, BlackGEM_fieldid, BlackGEM_RA, BlackGEM_Dec = find_latest_BlackGEM_field()

    if int(BlackGEM_minutes) >= 60:
        BlackGEM_status = "BlackGEM is not observing"
        BlackGEM_colour = "Black"
    elif int(BlackGEM_minutes) >= 30:
        BlackGEM_status = "BlackGEM is probably not observing"
        BlackGEM_colour = "Black"
    else:
        BlackGEM_status = "BlackGEM is observing!"
        BlackGEM_colour  = "MediumSeaGreen"

    BlackGEM_message = '<h4 style="color: ' + BlackGEM_colour + ';">' + BlackGEM_status + '</h4>'
    message_1 = "The most recent field was observed " + BlackGEM_minutes + " minute" + BlackGEM_minplur + ", " + BlackGEM_seconds + " second" + BlackGEM_secplur + " ago<br>"
    message_2 = "&nbsp&nbsp ID: " + str(BlackGEM_fieldid) + " &nbsp&nbsp RA: " + str(BlackGEM_RA) + " &nbsp&nbsp Dec: " + str(BlackGEM_Dec)
    message_2 = f"<span style='color: grey; font-style: italic;'>{message_2}</span>"

    print(message_1)
    print(message_2)

    # message = message_1+message_2
    message = BlackGEM_message+message_1+message_2

    return JsonResponse({'time': message})

def update_latest_BlackGEM_Field_small(request):
    BlackGEM_minutes, BlackGEM_minplur, BlackGEM_seconds, BlackGEM_secplur, BlackGEM_fieldid, BlackGEM_RA, BlackGEM_Dec = find_latest_BlackGEM_field()

    if int(BlackGEM_minutes) >= 60:
        BlackGEM_status = "BlackGEM is not observing"
        BlackGEM_colour = "Black"
    elif int(BlackGEM_minutes) >= 30:
        BlackGEM_status = "BlackGEM is probably not observing"
        BlackGEM_colour = "Black"
    else:
        BlackGEM_status = "BlackGEM is observing!"
        BlackGEM_colour  = "MediumSeaGreen"

    BlackGEM_message = '<h4 style="color: ' + BlackGEM_colour + ';">' + BlackGEM_status + '</h4>'

    return JsonResponse({'time': BlackGEM_message})


def get_time_in_la_silla():

    ## Define the observation time
    obstime = Time.now()  # Replace with your desired date
    # obstime = Time('2024-08-29 10:00:21')

    ## Define the location
    location = EarthLocation(lat=-29.25738889*u.deg, lon=-70.73791667*u.deg, height=2200*u.m)
    observer = Observer(location=location, timezone='UTC')

    # print("\n\nBark!\n\n")
    sun = coord.get_sun(obstime)
    altaz = coord.AltAz(location=location, obstime=obstime)
    current_altitude = get_sun(obstime).transform_to(altaz).alt.degree
    # print(current_altitude)
    # print("\n\nBark!\n\n")

    ## --- Calculate sunrise, sunset, and twilight times ---
    sunrise = observer.sun_rise_time(obstime, which='next')
    sunset = observer.sun_set_time(obstime, which='next')

    # Morning twilight times
    civil_twilight_morning = observer.twilight_morning_civil(obstime, which='next')
    nautical_twilight_morning = observer.twilight_morning_nautical(obstime, which='next')
    astronomical_twilight_morning = observer.twilight_morning_astronomical(obstime, which='next')

    # Evening twilight times
    civil_twilight_evening = observer.twilight_evening_civil(obstime, which='next')
    nautical_twilight_evening = observer.twilight_evening_nautical(obstime, which='next')
    astronomical_twilight_evening = observer.twilight_evening_astronomical(obstime, which='next')

    ## Calculate next event
    time_of_next_event = min([
        astronomical_twilight_morning,
        nautical_twilight_morning,
        civil_twilight_morning,
        sunrise,
        sunset,
        civil_twilight_evening,
        nautical_twilight_evening,
        astronomical_twilight_evening,
    ])

    time_until_next_event = (time_of_next_event - obstime).value
    hour_until_next_event = int(np.floor(time_until_next_event*24))
    mins_until_next_event = int(np.floor(time_until_next_event*60*24)) - hour_until_next_event*60
    secs_until_next_event = int(np.floor(time_until_next_event*60*60*24)) - mins_until_next_event*60 - hour_until_next_event*3600

    hour_plur = "s"
    mins_plur = "s"
    secs_plur = "s"
    if hour_until_next_event == 1: hour_plur = ""
    if mins_until_next_event == 1: mins_plur = ""
    if secs_until_next_event == 1: secs_plur = ""

    time_until_string = \
        str(hour_until_next_event) + " hour"   + hour_plur + ", " +    \
        str(mins_until_next_event) + " minute" + mins_plur + ", " +    \
        str(secs_until_next_event) + " second" + secs_plur + "<br> until "

    hour_until_next_event_string = str(hour_until_next_event)
    mins_until_next_event_string = str(mins_until_next_event)
    secs_until_next_event_string = str(secs_until_next_event)
    if len(hour_until_next_event_string) == 1: hour_until_next_event_string = '0'+hour_until_next_event_string
    if len(mins_until_next_event_string) == 1: mins_until_next_event_string = '0'+mins_until_next_event_string
    if len(secs_until_next_event_string) == 1: secs_until_next_event_string = '0'+secs_until_next_event_string

    time_until_string = \
        hour_until_next_event_string + ":" +    \
        mins_until_next_event_string + ":" +    \
        secs_until_next_event_string + "<br>"

    if   time_of_next_event == astronomical_twilight_morning:  current_event = "night";                           next_event = "Astronomical twilight (dawn)"
    elif time_of_next_event == nautical_twilight_morning:      current_event = "astronomical twilight (dawn)";    next_event = "Nautical twilight (dawn)"
    elif time_of_next_event == civil_twilight_morning:         current_event = "nautical twilight (dawn)";        next_event = "Civil twilight (dawn)"
    elif time_of_next_event == sunrise:                        current_event = "civil twilight (dawn)";           next_event = "Sunrise"
    elif time_of_next_event == sunset:                         current_event = "daytime";                         next_event = "Sunset"
    elif time_of_next_event == civil_twilight_evening:         current_event = "civil twilight (dusk)";           next_event = "Civil twilight (dawn)"
    elif time_of_next_event == nautical_twilight_evening:      current_event = "nautical twilight (dusk)";        next_event = "Nautical twilight (dawn)"
    elif time_of_next_event == astronomical_twilight_evening:  current_event = "astronomical twilight (dusk)";    next_event = "Astronomical twilight (dawn)"
    message_1 = "It is <b>" + current_event + "</b> in La Silla."
    message_2 = "Sun Altitude: %.2f" % current_altitude + '°'
    message_3 = next_event + " in " + time_until_string

    return message_1, message_2, message_3


def update_time_in_la_silla(request):

    message_1, message_2, message_3 = get_time_in_la_silla()

    message_1 = message_1 + "<br>"
    message_2 = f"<span style='color: grey; font-style: italic;'>{message_2}</span><br>"
    message_3 = f"<span style='color: grey; font-style: italic;'>{message_3}</span>"

    print(message_1)
    print(message_2)
    print(message_3)

    message = message_1+message_2+message_3

    return JsonResponse({'time': message})

class LiveFeed(TemplateView):
    template_name = 'live_feed.html'

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
    # return render(request, 'live_feed.html', {'plot_div': plot_div})



    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        bg = authenticate_blackgem()


        ## Get most recent pointing.
        query = """\
        select i.id
              , i.filter
              ,i."date-obs"
              ,"tqc-flag"
              ,i.object as tile
              ,i.dec_cntr_deg
              ,i.dec_deg
              ,s.ra_c
              ,s.dec_c
          from image i
              ,skytile s
         where i.object = s.field_id
        order by "date-obs" desc
        limit 20;
        """
        l_results = bg.run_query(query)
        df_images = pd.DataFrame(l_results, columns=['id', 'filter', 'date-obs',
                                                     'tqc-flag', 'field',
                                                     'dec_cntr_deg','dec_deg',
                                                     'ra_c', 'dec_c'
                                                 ])
        # df_images.round({'s-seeing': 2})
        # print(df_images)
        # print(list(df_images['dec_cntr_deg']))
        # list_min = 0
        # list_max = 10
        # print(df_images['dec_cntr_deg'].iloc[   list_min:list_max])
        # print(df_images['dec_ref_dms'].iloc[    list_min:list_max])
        # print(df_images['dec_deg'].iloc[        list_min:list_max])
        print(df_images["date-obs"].iloc[0])
        most_recent_image_time = df_images["date-obs"].iloc[0]
        most_recent_image_mjd = datetime_to_mjd(most_recent_image_time)
        print(most_recent_image_mjd)
        time_now = datetime.now(timezone.utc)
        mjd_now = datetime_to_mjd(time_now)
        hours_since_last_field      = ((mjd_now - most_recent_image_mjd)*24)
        minutes_since_last_field    = ((mjd_now - most_recent_image_mjd)*24*60)#-(hours_since_last_field*60)
        seconds_since_last_minute   = np.floor(((minutes_since_last_field-np.floor(minutes_since_last_field))*60))
        print("Seconds", seconds_since_last_minute)

        print("")
        print("======")
        print("Time since last field: %.2f" % minutes_since_last_field, "minutes.")
        print("Last field observed:", df_images["field"].iloc[0])
        print("RA: %.3f" % df_images["ra_c"].iloc[0], "; Dec: %.3f" % df_images["dec_c"].iloc[0])

        context['BlackGEM_hours']   = "%.0f" % np.floor(hours_since_last_field)
        context['BlackGEM_hrplur']  = "s"
        context['BlackGEM_minutes'] = "%.0f" % np.floor(minutes_since_last_field)
        context['BlackGEM_minplur'] = "s"
        context['BlackGEM_seconds'] = "%.0f" % seconds_since_last_minute
        context['BlackGEM_secplur'] = "s"
        if hours_since_last_field == 1:     context['BlackGEM_hrplur']  = ""
        if minutes_since_last_field == 1:   context['BlackGEM_minplur'] = ""
        if seconds_since_last_minute == 1:  context['BlackGEM_secplur'] = ""
        context['BlackGEM_fieldid'] = df_images["field"].iloc[0]
        context['BlackGEM_RA']      = df_images["ra_c"].iloc[0]
        context['BlackGEM_Dec']     = df_images["dec_c"].iloc[0]
        if minutes_since_last_field >= 60:
            context['BlackGEM_message'] = "BlackGEM is not observing"
            context['BlackGEM_colour']  = "Black"
        elif minutes_since_last_field >= 30:
            context['BlackGEM_message'] = "BlackGEM is probably not observing"
            context['BlackGEM_colour']  = "Black"
        else:
            context['BlackGEM_message'] = "BlackGEM is observing!"
            context['BlackGEM_colour']  = "MediumSeaGreen"

        return context

    style_dict = {
      "font-size": "16px",
      "margin-right": "10px",
      'width': '150px',
      'height': '30px',
      # 'scale':'2'
    }

    button_style_dict = {
        "font-size": "16px",
        "margin-right": "10px",
        "padding": "7px 24px",
        # 'border-radius': '5px',
        'color':'#027bff',
        'background-color': '#ffffff',
        'border': '2px solid #027bff',
        'border-radius': '5px',
        'cursor': 'pointer',
    }

    ## Live Feeds


    # x = []
    # y = []
    df = pd.DataFrame({
        'Time': range(5),
        'Mag': range(5, 10)
    })

    wait_interval = 5       ## Waiting interval in seconds

    app_names = [
        'Live_Observation_1',
        'Live_Observation_2',
        'Live_Observation_3',
        'Live_Observation_4',]



    for app_name in app_names:

        app_1 = DjangoDash(app_name)

        app_1.layout = html.Div([
            html.Div([
                dcc.Input(id='id-input',        type='text', min=0, max=360,  placeholder=' BlackGEM ID',        style=style_dict),
                html.Button('Search', id='submit-button', n_clicks=0, style=button_style_dict),
                dcc.Graph(id='live-update-graph_1', figure=go.Figure(layout={'margin': dict(l=20, r=20, t=40, b=30), 'height': 300})),
                dcc.Interval(
                    id='interval-component',
                    interval=wait_interval*1000,  # 5 seconds in milliseconds
                    n_intervals=0
                )
                # html.Button('Search', id='submit-button', n_clicks=0, style={"font-size": "16px","margin-right": "10px",})
            ], style={"text-align":"center"}),
            html.Div(id='results-container', children=[]),
        ], style={'height': '200px'})

        # Define the callback to update the table based on input coordinates
        @app_1.callback(
            Output('live-update-graph_1', 'figure'),
            [Input('submit-button', 'n_clicks'),
             Input('interval-component', 'n_intervals')],
            [State('id-input', 'value')],
            prevent_initial_call=True
        )
        def update_results(n_clicks, n_intervals, bgem_id):
            if bgem_id is None:
                raise PreventUpdate  # Prevents the callback from updating the output

            if bgem_id is not None:
                if bgem_id == "test":
                    print("BlackGEM ID: " + bgem_id)
                    print(os.getcwd())
                    df = pd.read_csv("./data/blackgem_test_data.csv")
                    df.loc[9+n_intervals] = [df['mjd'].iloc[-1]+n_intervals] + [df['mag'].iloc[-1]+(np.random.poisson(lam=100)/100-1)] + [0] + ["q"]
                    df_x = df['mjd']
                    df_y = df['mag']

                    min_x = np.min(df_x)
                    max_x = np.max(df_x)
                    max_y = np.max(df_y)

                    fig = px.line(df, x='mjd', y='mag')
                    time_last_update = time_now = Time.now()

                else:
                    df_bgem_lightcurve, df_limiting_mag = get_lightcurve_from_BGEM_ID(bgem_id)
                    df = df_bgem_lightcurve
                    df_x = df_bgem_lightcurve['i."mjd-obs"']
                    df_y = df_bgem_lightcurve['x.mag_zogy']

                    print(np.min(df_x))
                    print(np.min(df_limiting_mag['mjd']))

                    min_x = np.min([np.min(df_x), np.min(df_limiting_mag['mjd'])])
                    max_x = np.max([np.max(df_x), np.max(df_limiting_mag['mjd'])])
                    # min_y = np.min([df_y, df_limiting_mag['x.mag_zogy']])
                    max_y = np.max([np.max(df_y), np.max(df_limiting_mag['limiting_mag'])])


                    fig = plot_BGEM_lightcurve(df_bgem_lightcurve, df_limiting_mag)
                    time_last_update = Time(df_bgem_lightcurve['i."mjd-obs"'].iloc[-1], format='mjd')

                ## Get time since last update
                time_now = Time.now()
                print(time_now.mjd)
                print(time_last_update.mjd)
                days_since_last_update = time_now.mjd-time_last_update.mjd
                message_start = "(Latest: "
                if days_since_last_update > 1:
                    message = message_start + "%.2f" % days_since_last_update + " days ago)"
                elif days_since_last_update > 1/24:
                    message = message_start + "%.2f" % (days_since_last_update*24) + " hours ago)"
                else:
                    message = message_start + str(int(days_since_last_update*86400)) + " seconds ago)"
                print(message)

                annotation_x = min_x + ((max_x-min_x)/2)
                annotation_y = max_y + 0.5

                fig.add_annotation(
                            x               = annotation_x,
                            y               = annotation_y,
                            text            = 'Last check at ' + str(time_now)[:19] + ' UT',
                            showarrow       = False

                )


                # Adjust layout to reduce whitespace
                fig.update_layout(
                    title="BGEM ID " + bgem_id +"\n" + message,
                    margin=dict(l=20, r=0, t=40, b=30),  # Set margins to reduce whitespace
                    title_x=0.5,  # Center the title
                    title_y=0.95,  # Adjust title position
                    height=300
                )

                return fig
            return []


## Lightcurve


    # # Define the callback to update the grid
    # @app_1.callback(
    #     Output('live-update-graph',    'figure'),
    #     Input('interval-component',    'n_intervals'),
    #     State('id-input', 'value'),
    # )
    # def update_grid_table(n_intervals):
    #     # fetch_data()
    #     # df = fetch_data(n_intervals)
    #     print("Fetching New Data!")
    #     x.append(n_intervals)
    #     y.append(16+np.random.poisson(lam=100)/100)
    #     df = pd.DataFrame({
    #         'Time': x,
    #         'Mag': y
    #     })
    #     figure = px.line(df, x='Time', y='Mag')
    #     return figure, df.to_dict('records')




    # ## Define the layout of the Dash app
    # app_1.layout = html.Div([
    # ]
    # )
    # # Define the callback to update the grid
    # @app_1.callback(
    #     [Output('live-update-graph',    'figure'),
    #      Output('live-update-grid',     'rowData')],
    #     [Input('interval-component',    'n_intervals')]
    # )
    # def update_grid_table(n_intervals):
    #     # fetch_data()
    #     # df = fetch_data(n_intervals)
    #     print("Fetching New Data!")
    #     x.append(n_intervals)
    #     y.append(16+np.random.poisson(lam=100)/100)
    #     df = pd.DataFrame({
    #         'Time': x,
    #         'Mag': y
    #     })
    #     figure = px.line(df, x='Time', y='Mag')
    #     return figure, df.to_dict('records')



def LiveFeed_BGEM_ID_View(request, bgem_id):
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

    bg = authenticate_blackgem()

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


    ## --- Image ---
    print("Getting image...")
    # if tns_flag:
        # file_name = "../" + get_transient_image(bgem_id, ra, dec, df_bgem_lightcurve,
            # tns_objects_potential["RA"].iloc[0], tns_objects_potential["Dec"].iloc[0]
            # )
    # else:
    file_name = "../" + get_transient_image(bgem_id, ra, dec, df_bgem_lightcurve)
    print("Image name:", file_name)


    ## Detail each observation:

    #
    # df_new = df_bgem_lightcurve.rename(columns={
    #     'a.xtrsrc'          : "xtrsrc",
    #     'x.ra_psf_d'        : "ra_psf_d",
    #     'x.dec_psf_d'       : "dec_psf_d",
    #     'x.flux_zogy'       : "flux_zogy",
    #     'x.fluxerr_zogy'    : "fluxerr_zogy",
    #     'x.mag_zogy'        : "mag_zogy",
    #     'x.magerr_zogy'     : "magerr_zogy",
    #     'i."mjd-obs"'       : "mjd_obs",
    #     'i."date-obs"'      : "date_obs",
    #     'i.filter'          : "filter",
    # })
    # df_new.style.format({
    #     # 'runcat_id' : make_runcat_clickable,
    #     'xtrsrc' : make_xtrsrc_clickable
    # })

    # df_new['xtrsrc'] = df_new['xtrsrc'].apply(lambda x: f'[{x}](https://staging.apps.blackgem.org/transients/blackview/show_xtrsrc/{x})')

    # print(df_new)

    ## --- Update Data! ---
    def fetch_data(n):
        print("Fetching New Data!")
        df = pd.DataFrame({
            'Time': range(n),
            'Mag': range(n, n*2)
        })
        return df

    x = []
    y = []
    df = pd.DataFrame({
        'Time': range(5),
        'Mag': range(5, 10)
    })

    wait_interval = 5       ## Waiting interval in seconds

    ## --- Update All ---
    app_all = DjangoDash('Live_Observation_All')

    ## Define the layout of the Dash app
    app_all.layout = html.Div([
        html.Div([
            dcc.Graph(id='live-update-graph',
                style={'height': '450px', 'width': '100%'}),
            html.Br(),
            # html.H2(
            #     "BlackGEM ID ......",
            #     style={"border-right" : "thin solid #dddddd"},
            # ),
            # html.Div(
            #     "Is BlackGEM Observing? (Heck should I know?)",
            # ),
            dag.AgGrid(
                id='live-update-grid',
                rowData=[],
                defaultColDef={
                    'sortable': True,
                    'filter': True,
                    'resizable': True,
                    'editable': False,
                },
                columnSize="autoSize",
                dashGridOptions = {"skipHeaderOnAutoSize": True, "rowSelection": "single"},
                style={'height': '400px', 'width': '100%'},  # Set explicit height for the table
                className='ag-theme-balham'  # Add a theme for better appearance
            ),
            dcc.Interval(
                id='interval-component',
                interval=wait_interval*1000,  # 5 minutes in milliseconds
                n_intervals=0
            ),
        ], style={'margin-bottom': '20px', "text-align":"center"}),
        html.Div(id='results-container', children=[]),
    ], style={'height': '800px', 'width': '100%'})

    # <div class="row">
    #     <div class="col-md-6" style="border-right:thin solid #dddddd;">
    #         <h4>BlackGEM ID {{ bgem_id }}</h4>
    #     </div>
    #     <div class="col-md-6">
    #         <h4>Is BlackGEM Observing? <b>Unknown</b><br></h4>
    #     </div>
    # </div>


    # Define the callback to update the table based on input coordinates
    @app_all.callback(
        Output('live-update-graph', 'figure'),
        Output('live-update-grid', 'rowData'),
        Output('live-update-grid', 'columnDefs'),
        Input('interval-component', 'n_intervals'),
        # [State('bgem_id', 'value')],
        # prevent_initial_call=True
    )
    def update_results(n_intervals):
        if bgem_id is None:
            raise PreventUpdate  # Prevents the callback from updating the output

        if bgem_id is not None:
            if bgem_id == "test":
                print("BlackGEM ID: " + str(bgem_id))
                print(os.getcwd())
                df = pd.read_csv("./data/blackgem_test_data.csv")
                df.loc[9+n_intervals] = [df['mjd'].iloc[-1]+n_intervals] + [df['mag'].iloc[-1]+(np.random.poisson(lam=100)/100-1)] + [0] + ["q"]
                df_x = df['mjd']
                df_y = df['mag']

                min_x = np.min(df_x)
                max_x = np.max(df_x)
                max_y = np.max(df_y)

                fig = px.line(df, x='mjd', y='mag')
                time_last_update = time_now = Time.now()

            else:
                df_bgem_lightcurve, df_limiting_mag = get_lightcurve_from_BGEM_ID(bgem_id)
                df = df_bgem_lightcurve
                df_x = df_bgem_lightcurve['i."mjd-obs"']
                df_y = df_bgem_lightcurve['x.mag_zogy']

                min_x = np.min([np.min(df_x), np.min(df_limiting_mag['mjd'])])
                max_x = np.max([np.max(df_x), np.max(df_limiting_mag['mjd'])])
                # min_y = np.min([df_y, df_limiting_mag['x.mag_zogy']])
                max_y = np.max([np.max(df_y), np.max(df_limiting_mag['limiting_mag'])])


                fig = plot_BGEM_lightcurve(df_bgem_lightcurve, df_limiting_mag)
                time_last_update = Time(df_bgem_lightcurve['i."mjd-obs"'].iloc[-1], format='mjd')

            ## Get time since last update
            time_now = Time.now()
            print(time_now.mjd)
            print(time_last_update.mjd)
            days_since_last_update = time_now.mjd-time_last_update.mjd
            message_start = "(Latest datum "
            if days_since_last_update > 1:
                message = message_start + "%.2f" % days_since_last_update + " days ago)"
            elif days_since_last_update > 1/24:
                message = message_start + "%.2f" % (days_since_last_update*24) + " hours ago)"
            else:
                message = message_start + str(int(days_since_last_update*86400)) + " seconds ago)"
            print(message)

            annotation_x = min_x + ((max_x-min_x)/2)
            annotation_y = max_y + 0.5

            fig.add_annotation(
                        x               = annotation_x,
                        y               = annotation_y,
                        text            = 'Last check at ' + str(time_now)[:19] + ' UT',
                        showarrow       = False

            )


            # Adjust layout to reduce whitespace
            fig.update_layout(
                title="BlackGEM ID: " + str(bgem_id) +"\n" + message,
                margin=dict(l=20, r=20, t=20, b=20),  # Set margins to reduce whitespace
                title_x=0.5,  # Center the title
                title_y=0.95,  # Adjust title position
                height=400
            )

            ## Rename to play nice with the Dash AgGrid
            df_new = df.rename(columns={
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

            columnDefs = [{'headerName': col, 'field': col, 'type': 'leftAligned'} for col in df_new.columns]  # Column definitions


            return fig, df_new.to_dict('records'), columnDefs
        return []




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
        "image_name"        : file_name
    }

    return render(request, "live_feed/index.html", context)






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

def authenticate_blackgem():
    creds_user_file = str(Path.home()) + "/.bg_follow_user_john_creds"
    creds_db_file = str(Path.home()) + "/.bg_follow_transientsdb_creds"
    creds_user_file = "../../.bg_follow_user_john_creds"
    creds_db_file = "../../.bg_follow_transientsdb_creds"
    print(creds_user_file)
    print(creds_db_file)

    # Instantiate the BlackGEM object
    bg = BlackGEM(creds_user_file=creds_user_file, creds_db_file=creds_db_file)

    return bg

def get_blackgem_id_from_iauname(iauname):
    # Instantiate the BlackGEM object
    bg = authenticate_blackgem()

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
        # photometry = BGEM_to_GEMTOM_photometry(df_bgem_lightcurve)
        photometry = BGEM_to_GEMTOM_photometry_2(df_bgem_lightcurve, df_limiting_mag)
        # print(df_bgem_lightcurve)
        # print(df_bgem_lightcurve.columns)

        print("-- BlackGEM: Getting Data... Done.")

        # ## If the magnitude shows an upper limit, remove.
        # datum_magnitude = str(datum['magnitude'])
        # print(datum_magnitude)
        # if ('>' in datum_magnitude) or ('<' in datum_magnitude):
        #     datum['limit'] = float(datum_magnitude[1:])
        #     datum['magnitude'] = 0
        # elif datum_magnitude == '--':
        #     datum_magnitude = ''
        # else:
        #     datum['magnitude'] = float(datum_magnitude)

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

def UpdateBlackGEMFunc(target_name, target_id, target_blackgemid):

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
