# from .forms import UploadFileForm
# from django.core.files.uploadedfile import SimpleUploadedFile

import os
import sys
import pandas as pd
from astropy.coordinates import SkyCoord, Galactocentric, EarthLocation, AltAz, get_sun, get_moon
from astropy import units as u
import plotly.express as px
import dash
from dash import Dash, dcc, html, Input, Output, State, callback, dash_table, ctx
from io import StringIO
import dash_bootstrap_components as dbc
import time

from django.views.generic import TemplateView, FormView
from django.http import HttpResponse, HttpResponseRedirect, FileResponse, JsonResponse
from django.urls import reverse
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from django.utils.safestring import mark_safe
from django.core.files.storage import FileSystemStorage
from guardian.shortcuts import assign_perm, get_objects_for_user
from django.contrib.auth.decorators import login_required

## For Register and Login
from django.contrib.auth import authenticate, login

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
import dash_ag_grid as dag
import json
import plotly.graph_objs as go
import re
import shutil # save img locally
from django_plotly_dash import DjangoDash
from mocpy import MOC
from pathlib import Path
from PIL import Image, ImageDraw
from plotly.offline import plot
from urllib.request import urlretrieve
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier

## For the ToO Forms
from .forms import *
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
from tom_targets.models import Target, TargetList, TargetName, TargetExtra
from django.db.models import Q
from tom_common.hints import add_hint
from tom_dataproducts.exceptions import InvalidFileFormatException

## Data Products
from processors.ztf_processor import ZTFProcessor
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic.base import RedirectView
from data_processor import run_data_processor
from tom_dataproducts.models import DataProduct, DataProductGroup, ReducedDatum
from tom_dataproducts.forms import DataProductUploadForm

## Sending Email
from django.core.mail import send_mail
import smtplib
from email.mime.multipart import MIMEMultipart
from email.message import Message
from email.mime.text import MIMEText

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


def get_lightcurve(bgem_id):
    '''
    Fetches the lightcurve from Hugo's server. Returns blank if no lightcurve exists.
    '''
    url = "http://xmm-ssc.irap.omp.eu/claxson/BG_images/lcrequests/" + bgem_id + "_lc.jpg"
    r = requests.get(url)
    if r.status_code != 404:
        return url
    else:
        return ""


def add_to_GEMTOM(id, name, ra, dec, tns_prefix=False, tns_name=False):

    # get_lightcurve(id)


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

    print(tns_prefix)
    print(tns_name)
    if tns_prefix and tns_name:
        print("Saving TNS Name...")
        # print(tns_prefix + " " + tns_name)
        target = Target.objects.get(name=name)
        target.save(extras={'TNS Name': tns_prefix + " " + tns_name})

    ## --- Did this target already exist? ---
    existing_target_id = None
    all_targets = Target.objects.all()
    num_targets = len(list(all_targets))

    ## Find the id of this target
    for i in range(num_targets):
        if str(list(all_targets)[i]) == name:
            break
    existing_target_id = all_targets[i].id

    ## If the most recent target is this target...
    if all_targets[len(all_targets)-1].name == name:

        ## If it was made more than 10 seconds ago, it already existed.
        time_created = all_targets[len(all_targets)-1].created
        time_now = datetime.now(timezone.utc)
        if (time_now-time_created).seconds < 2:
            print("Target Created")
            existing_target_id = all_targets[len(all_targets)-1].id
            created = True
        else:
            print("Target Already Existed")
            created = False
    else:
        print("Target Already Existed")
        created = False

    return created, existing_target_id

def test_print():
    print("Hello World!")

def download_lightcurve(request):

    bgem_id = request.POST.get('bgem_id')

    df_bgem_lightcurve, df_limiting_mag = get_lightcurve_from_BGEM_ID(bgem_id)

    # Convert DataFrame to CSV string
    csv_string = df_bgem_lightcurve.to_csv(index=False)

    # Create the HttpResponse object with appropriate headers for CSV.
    response = HttpResponse(csv_string, content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="BGEM_' + bgem_id + '_lightcurve.csv"'
    # response['Content-Disposition'] = 'attachment; filename="BlackGEM_Potential_CVs_' + bgem_id + '.csv"'

    # Return the response which prompts the download
    return response


def plot_BGEM_lightcurve(df_bgem_lightcurve, df_limiting_mag):

    print("Starting BGEM lightcurve plotting...")

    time_list = []
    time_list.append(time.time())
    # print("\n\nBark!")
    # print(df_bgem_lightcurve.columns)
    # print(df_limiting_mag.columns)
    # print(df_limiting_mag[['mjd', 'magnitude', 'limiting_mag', 'filter', 'error']])

    time_now = datetime.now(timezone.utc)
    mjd_now = datetime_to_mjd(time_now)

    time_list.append(time.time())

    ## Lightcurve
    filters = ['u', 'g', 'q', 'r', 'i', 'z']
    colors = ['darkviolet', 'forestgreen', 'darkorange', 'orangered', 'crimson', 'dimgrey']
    symbols = ['triangle-up', 'diamond-wide', 'circle', 'diamond-tall', 'pentagon', 'star']

    fig = go.Figure()

    for f in filters:
        time_list.append(time.time())
        df_2 = df_bgem_lightcurve.loc[df_bgem_lightcurve['i.filter'] == f]
        # print(len(df_2))
        df_2 = df_2[df_2['x.mag_zogy'] != 99]
        # print(len(df_2))
        df_limiting_mag_2 = df_limiting_mag.loc[df_limiting_mag['filter'] == f]

        if len(df_2) > 0 or len(df_limiting_mag_2) >0:

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

    time_list.append(time.time())
    fig.update_layout(
        height=400,
        hovermode="x",
        xaxis=dict(tickformat ='digits'),
        # xaxis=dict(tickformat ='d'),
    # fig.update_layout(xaxis=dict(tickformat ='d'),
        # margin=dict(l=2, r=2),  # Set margins to reduce whitespace
        margin=dict(t=0, b=50, l=2, r=2),  # Set margins to reduce whitespace
        # title="Lightcurves",
        xaxis_title="MJD",
        yaxis_title="Magnitude",)
    fig.update_yaxes(autorange="reversed")


    print("Plot BGEM Lightcurve Times:")
    for i in range(len(time_list)-1):
        print("t"+str(i)+"->t"+str(i+1)+": "+str(time_list[i+1]-time_list[i]))

    return fig


def plot_BGEM_location_on_sky(df_bgem_lightcurve, ra, dec):
    # print("\n\n\n")
    # print(df_bgem_lightcurve)
    # print(df_bgem_lightcurve.index)
    # print("\n\n\n")
    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df_bgem_lightcurve['x.ra_psf_d'], y=df_bgem_lightcurve['x.dec_psf_d'], mode='markers', name='Line and Marker', marker_color="LightSeaGreen"))
    fig.add_trace(go.Scatter(x=df_bgem_lightcurve['x.ra_psf_d'], y=df_bgem_lightcurve['x.dec_psf_d'], mode='markers', name='Line and Marker', marker_color=df_bgem_lightcurve.index, marker_colorscale=["cyan", "magenta"]))
    fig.update_xaxes(autorange="reversed")
    mid_x = ra
    mid_y = dec
    c = SkyCoord(mid_x, mid_y, frame='icrs', unit='deg')
    min_circle = c.directional_offset_by(45  * u.deg, 1 / np.cos(abs(np.deg2rad(dec))) * u.arcsecond)
    max_circle = c.directional_offset_by(225 * u.deg, 1 / np.cos(abs(np.deg2rad(dec))) * u.arcsecond)

    ## Make blank circle to ensure aspect ratio
    lightcurve_ras = np.array(df_bgem_lightcurve['x.ra_psf_d']) * u.deg
    lightcurve_decs = np.array(df_bgem_lightcurve['x.dec_psf_d']) * u.deg
    lightcurve_radecs   = SkyCoord(lightcurve_ras, lightcurve_decs, frame='icrs')
    lightcurve_seps     = c.separation(lightcurve_radecs)
    max_separation = np.max(lightcurve_seps*1.2).to(u.arcsecond)
    min_blank_circle = c.directional_offset_by(45  * u.deg, max_separation / np.cos(abs(np.deg2rad(dec))))
    max_blank_circle = c.directional_offset_by(225 * u.deg, max_separation / np.cos(abs(np.deg2rad(dec))))


    # print("Circles")
    # print(min_circle)
    # print(max_circle)

    # print(min_circle.separation(max_circle))

    circle_x0 = min_circle.ra.value
    circle_y0 = min_circle.dec.value
    circle_x1 = max_circle.ra.value
    circle_y1 = max_circle.dec.value

    blank_circle_x0 = min_blank_circle.ra.value
    blank_circle_y0 = min_blank_circle.dec.value
    blank_circle_x1 = max_blank_circle.ra.value
    blank_circle_y1 = max_blank_circle.dec.value

    if circle_x0 < 1 and circle_x1 > 359:
        circle_x1 -= 360

    if blank_circle_x0 < 1 and blank_circle_x1 > 359:
        blank_circle_x1 -= 360

    fig.add_shape(type="circle",
        xref="x", yref="y",
        x0=circle_x0, y0=circle_y0, x1=circle_x1, y1=circle_y1,
        line_color="red",
    )
    fig.add_shape(type="circle",
        xref="x", yref="y",
        x0=blank_circle_x0, y0=blank_circle_y0, x1=blank_circle_x1, y1=blank_circle_y1,
        line_color="white", layer="below", opacity=0
    )
    fig.update_layout(width=350, height=295,
        minreducedwidth=250,
        minreducedheight=245,
        margin=dict(t=10, b=40, l=10, r=30),  # Set margins to reduce whitespace
    )

    return fig



def ra_dec_to_galactic(ra, dec):
    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

    gal = coord.galactic

    l = gal.l.deg
    b = gal.b.deg

    return l, b


## =============================================================================
## -------------------- Code for the Target Watchlist Page ---------------------

# class WatchlistView(TemplateView):
@login_required
def WatchlistView(request):

    context = {
        "testing" : 'Test successful!',
        # "targets" : targets,
        # "target_names" : [target.name for target in targets],
    }

    template_name = 'target_watchlist.html'

    df_targets = pd.DataFrame(list(Target.objects.all().values()))
    # target_2 = Target.targetextra_set.all()

    # --- Now handle extras ---
    # Suppose you want keywords like "redshift", "priority", "magnitude"
    desired_extras = ['BlackGEM ID']

    # Query extras for those keywords
    extras = TargetExtra.objects.filter(key__in=desired_extras).values(
        'target_id', 'key', 'value'
    )

    extras_df = pd.DataFrame(list(extras))

    # Pivot so each key becomes a column
    extras_pivot = extras_df.pivot(index='target_id', columns='key', values='value').reset_index()

    # Merge with main df
    df_targets = df_targets.merge(extras_pivot, left_on='id', right_on='target_id', how='left')

    # Drop the duplicate id if you like
    df_targets = df_targets.drop(columns=['target_id'])

    df_targets.to_csv("./data/target_watchlist_ids.csv", index=False)

    app = DjangoDash("watchlist_table")

    print(df_targets.columns)

    getRowStyle = {
        "styleConditions": [
            {
                "condition": "params.data.limiting_mag > 0",
                "style": {"color": "lightgrey"},
            },
        ],
        "defaultStyle": {"color": "black"},
    }

    bg = authenticate_blackgem()


    targets_check = False
    if os.path.exists("./data/target_watchlist_latestmags.csv"):
        df_targets_all = pd.read_csv("./data/target_watchlist_latestmags.csv")
        targets_check = True
    else:
        print("Waiting for mags! Please run BG_Update_Watchlist.py!")

    if targets_check:

        context["num_targets"] = len(df_targets_all)

        ## Define the layout of the Dash app
        app.layout = html.Div([
            dag.AgGrid(
                id='observation-grid',
                rowData=df_targets_all.to_dict('records'),
                columnDefs=[
                            {'headerName': 'GEMTOM', 'field': 'GEMTOM_Link', 'cellRenderer': 'markdown', "linkTarget":"_blank"},
                            {'headerName': 'BGEM ID', 'field': 'BGEM_ID_Link', 'cellRenderer': 'markdown', "linkTarget":"_blank", 'minWidth' : 110},
                            {'headerName': 'Name', 'field': 'name'},
                            {'headerName': 'RA', 'field': 'ra',
                                "valueFormatter": {"function": "d3.format('.2f')(params.value)"}},
                            {'headerName': 'Dec', 'field': 'dec',
                                "valueFormatter": {"function": "d3.format('.2f')(params.value)"}},
                            {'headerName': 'Mag', 'field': 'latest_mag',
                                "valueFormatter": {"function": "d3.format('.1f')(params.value)"}},
                            {'headerName': 'Filter', 'field': 'filter', 'minWidth' : 90},
                            {'headerName': 'Latest Obs', 'field': 'last_obs'},
                ],
                getRowStyle=getRowStyle,
                defaultColDef={
                    'sortable': True,
                    'filter': True,
                    'resizable': True,
                    'editable': False,
                },
                # columnSize="sizeToFit",
                columnSize="autoSize",
                dashGridOptions={
                    "skipHeaderOnAutoSize": True,
                    "rowSelection": "single",
                    "enableCellTextSelection": True,
                },
                style={'height': '500px', 'width': '100%'},  # Set explicit height for the grid
                # className='ag-theme-balham'  # Add a theme for better appearance
                className='ag-theme-alpine'  # Add a theme for better appearance
            ),
            dcc.Store(id='selected-row-data'),  # Store to hold the selected row data
            html.Div(id='output-div'),  # Div to display the information
        ], style={'height': '500px', 'width': '100%'}
        )

    return render(request, "target_watchlist.html", context)



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
## --------------------------- Code for Registering ----------------------------

def sign_up(request):

    if request.method == 'POST':
        form = RegistrationForm(request.POST)

        if form.is_valid():
            user = form.save()
            user.is_active = False
            user.save()
            messages.info(request, "Thanks for registering. Please send an email to johnapaice@gmail.com for authentication.")
            # new_user = authenticate(username=form.cleaned_data['username'],
                                    # password=form.cleaned_data['password1'],
                                    # )
            # login(request, new_user)
            return redirect('/authentication/')
            # return redirect('/accounts/login/')
        else:
            print("Not valid!")
            print(form.errors)
            # return redirect('/accounts/signup/')
            return render(request, 'register.html', {'form':form})

        # return redirect('/')

    else:
        form = RegistrationForm()
        return render(request, 'register.html', {'form':form})

def authentication(request):
    return render(request, 'authentication.html')

## =============================================================================
## -------------------------- Codes for the ToO page ---------------------------

@login_required
def delete_telescopetime(request):

    '''
    Deletes telescope time entry
    '''

    num = request.POST.get('Num')
    if len(num) > 0: num = np.int64(num)

    ToO_filename = "./data/too_data.csv"
    ToO_data = pd.read_csv(ToO_filename)

    index = ToO_data.index[ToO_data['num'] == num]

    ToO_data = ToO_data.drop(index)

    ToO_data.to_csv(ToO_filename, index=False)

    return HttpResponseRedirect('/telescope_time/')  # Redirect to a success page (to be created)

def get_ToO_data():
    ToO_filename = "./data/too_data.csv"
    ToO_data = pd.read_csv(ToO_filename)
    return ToO_data

def plot_ToO_timeline():

    ToO_data = get_ToO_data()

    ## Split up all multi-band observations
    first = True
    for i in range(len(ToO_data)):
        band = ToO_data.Band[i]
        if "/" in band:
            band = band.split("/")
            sub_df = [[ToO_data.loc[i,j]]*len(band) for j in ToO_data.columns]
            # sub_df = zip(sub_df)
            sub_df = pd.DataFrame.from_records(sub_df)
            sub_df = sub_df.transpose()
            sub_df.columns = ToO_data.columns
            for k in range(len(band)):
                sub_df.loc[k,"Band"] = band[k]

            ToO_data = ToO_data.drop(i)
            if first:
                sub_df_array = sub_df
                first = False
            else:
                sub_df_array = pd.concat([sub_df_array,sub_df]).reset_index(drop=True)

    if not first:
        ToO_data = pd.concat([ToO_data,sub_df_array]).reset_index(drop=True)

    custom_dict = {
        "Radio"         : 0,
        "Millimetre"    : 1,
        "Microwave"     : 2,
        "Infrared"      : 3,
        "Optical"       : 4,
        "Ultraviolet"   : 5,
        "X-Ray"         : 6,
        "Gamma"         : 7,
        "Other"         : 8,
    }

    ToO_data = ToO_data.sort_values(by="Band", key=lambda x: x.map(custom_dict))

    print("\nMaking ToO Timeline...")

    time_now = datetime.now()
    mjd_now = datetime_to_mjd(time_now)

    ## Lightcurve
    fig = go.Figure()

    for i in range(len(ToO_data)):
        ToO_data.loc[i, "date_start"]    = datetime.strptime(ToO_data.loc[i, "date_start"], '%Y-%m-%d')
        ToO_data.loc[i, "date_close"]    = datetime.strptime(ToO_data.loc[i, "date_close"], '%Y-%m-%d')+timedelta(days=1)

    ToO_data["Notes"] = [" " if i != i else i for i in ToO_data["Notes"]]

    fig = px.timeline(
        ToO_data,
        x_start='date_start',
        x_end='date_close',
        # y = 'Telescope',
        y = 'Band',
        hover_data = ['Name', 'Notes'],
        color='Telescope',
        # color=['red', 'green', 'blue'],
        opacity=0.5,
    )

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
            ToO_data = pd.DataFrame(columns = ['num','Name','date_start','date_close','Telescope','Location','Band','Notes','Submitter'])
            ToO_data.to_csv("./data/too_data.csv", index=False)

        too_lightcurve = plot(plot_ToO_timeline(), output_type='div')

        ToO_data = get_ToO_data()
        ToO_data = ToO_data.sort_values(by=["date_start"])

        context = super().get_context_data(**kwargs)
        context['form'] = ToOForm()  # Add the form to the context
        context['lightcurve'] = too_lightcurve
        ToO_data["Notes"]       = [" " if i != i else i for i in ToO_data["Notes"]]
        ToO_data["Submitter"]   = [" " if i != i else i for i in ToO_data["Submitter"]]

        in_past = [(datetime.strptime(i, '%Y-%m-%d') < (datetime.now()-timedelta(1))) for i in ToO_data["date_close"]]
        # in_past = [True if i.seconds < 0 else False for i in in_past]
        # print(in_past)
        ToO_data["in_past"] = in_past

        context["ToO_data"] = zip(*[ToO_data[col] for col in ToO_data])

        return context

    def post(self, request, *args, **kwargs):
        form = ToOForm(request.POST)
        if form.is_valid():
            # Get the data from the form
            name        = form.cleaned_data['PI']
            date_start  = form.cleaned_data['date_start']
            date_close  = form.cleaned_data['date_close']
            telescope   = form.cleaned_data['telescope']
            band        = form.cleaned_data['band']
            notes       = form.cleaned_data['notes']
            location    = form.cleaned_data['location']
            submitter   = request.POST.get('submitter')

            try:
                telescope_lon = coords.EarthLocation.of_site(location).geodetic.lon
                telescope_lat = coords.EarthLocation.of_site(location).geodetic.lat

                print("Telescope:", telescope)
                print("Location:", location)
                print("Lon:", telescope_lon, " Lat:", telescope_lat)
            except:
                print("Telescope location not recognised.")


            band = "/".join(band)

            fileOut = "./data/too_data.csv"

            if os.path.exists(fileOut):
                old_ToO_data = pd.read_csv(fileOut)
                if len(old_ToO_data) > 0:
                    new_num = max(old_ToO_data.num)+1
                else:
                    new_num = 1
            else:
                new_num = 1

            new_ToO_data = pd.DataFrame({
                'num'           : [new_num],
                'Name'          : [name],
                'date_start'    : [date_start],
                'date_close'    : [date_close],
                'Telescope'     : [telescope],
                'Location'      : [location],
                'Band'          : [band],
                'Notes'         : [notes],
                'Submitter'     : [submitter],
            })




            if os.path.exists(fileOut):
                all_ToO_data = pd.concat([old_ToO_data,new_ToO_data]).reset_index(drop=True)
                all_ToO_data.to_csv(fileOut, index=False)
            else:
                new_ToO_data.to_csv(fileOut, index=False)

            # Redirect after POST to avoid resubmitting form on page refresh
            return HttpResponseRedirect('/telescope_time/')  # Redirect to a success page (to be created)
        else:
            # Re-render the form with errors if invalid
            # print(form)
            print("Form not valid!")
            # raise ValidationError("Form not valid!")
            print(str(form.errors.as_json())[26:-16])
            return JsonResponse({"error": str(form.errors.as_json())[26:-16]}, status=400)

            # return self.render_to_response(self.get_context_data(form=form))

    # def load_ToO_data(self):



        # app = DjangoDash('ToO_Database')
        #
        # ToO_data = ToO_data.rename(columns={
        #     'date_start'          : "Date (Start)",
        #     'date_close'          : "Date (End)",
        # })
        #
        # # for i in range(len(ToO_data)):
        # #     ToO_data.loc[i, "Date (Start)"]    = datetime.strptime(str(ToO_data["Date (Start)"].iloc[i]), '%Y%m%d')
        # #     ToO_data.loc[i, "Date (End)"]      = datetime.strptime(str(ToO_data["Date (End)"].iloc[i]),   '%Y%m%d')
        #
        # app.layout = html.Div([
        #     dag.AgGrid(
        #         id='ToO_Database',
        #         rowData=ToO_data.to_dict('records'),
        #         columnDefs=[{'headerName': col, 'field': col} for col in ToO_data.columns],
        #         defaultColDef={
        #             'sortable': True,
        #             'filter': True,
        #             'resizable': True,
        #             'editable': True,
        #         },
        #         columnSize="autoSize",
        #         dashGridOptions={
        #             "skipHeaderOnAutoSize": True,
        #             "rowSelection": "single",
        #         },
        #         style={'height': '300px', 'width': '100%'},  # Set explicit height for the grid
        #         className='ag-theme-balham'  # Add a theme for better appearance
        #     ),
        #     dcc.Store(id='selected-row-data'),  # Store to hold the selected row data
        #     html.Div(id='output-div'),  # Div to display the information
        # ], style={'height': '300px', 'width': '100%'}
        # )

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



## Extra code for just my secret history page:



@login_required
def SecretOrphans(request):
    '''
    Finds and displays data for my singular month:
    '''
    template_name = 'secretorphans.html'

    print("Starting SecretOrphans...")
    time_list = []
    time_list.append(time.time())

    context = {
        "obs_date"                      : "FakeDate",
    }

    time_list.append(time.time())

    df_orphans = pd.read_csv("./data/orphans_20250701_to_20250731.csv")
    ## Cut out the terrible ones
    df_orphans = df_orphans[df_orphans["real_bogus_probabilities"]>0.4]

    time_list.append(time.time())

    rated_orphans = False
    try:
        df_orphans_all = pd.read_csv("./data/history_transients/rated_orphans.csv")
        rated_orphans = True

    except:
        print("'Rated Orphans' file not present. Please create!")

    if df_orphans is not None:
        if rated_orphans:
            yes_no_list = []
            notes_list = []

            for runcat_id in df_orphans.runcat_id:
                if runcat_id in list(df_orphans_all.runcat_id):
                    index = df_orphans_all.index[df_orphans_all['runcat_id'] == int(runcat_id)]

                    yes_no_list.append(df_orphans_all['yes_no'].values[index][0])
                    notes_list.append(df_orphans_all['notes'].values[index][0])
                else:
                    yes_no_list.append("")
                    notes_list.append("")

            df_orphans['yes_no'] = yes_no_list
            df_orphans['notes'] = notes_list

        else:
            df_orphans['yes_no'] = [""]*len(df_orphans)
            df_orphans['notes']  = [""]*len(df_orphans)

        df_orphans = df_orphans.sort_values(by=['i_rb_avg'], ascending=False)
        df_orphans = df_orphans.sort_values(by=['u_rb_avg'], ascending=False)
        df_orphans = df_orphans.sort_values(by=['q_rb_avg'], ascending=False)
        df_orphans = df_orphans.fillna('')

        if "std_max" not in df_orphans.columns:
            df_orphans['std_max'] = [np.nan for x in df_orphans.det_sep]
            df_orphans['std_min'] = [np.nan for x in df_orphans.det_sep]
            df_orphans['std_frc'] = [np.nan for x in df_orphans.det_sep]
            df_orphans['std_ang'] = [np.nan for x in df_orphans.det_sep]

        if "probabilities" not in df_orphans.columns:
            df_orphans['probabilities'] = [0 for x in df_orphans.det_sep]

        nan_fix = False
        if "real_bogus_probabilities" not in df_orphans.columns:
            nan_fix = True
            df_orphans['real_bogus_probabilities']  = df_orphans['probabilities']
            df_orphans['asteroid_probabilities']    = [0 for x in df_orphans.det_sep]
            df_orphans['diff_spike_probabilities']  = [0 for x in df_orphans.det_sep]

        df_orphans = df_orphans.sort_values(by=['real_bogus_probabilities'], ascending=False)
        df_orphans = df_orphans.sort_values(by=['yes_no', 'real_bogus_probabilities'], ascending=[True, False])


        # df_orphans.std_min = df_orphans.std_min.fillna(value=np.nan)
        df_orphans.std_min = df_orphans.std_min.replace('', np.nan)
        # df_orphans.std_frc = df_orphans.std_min.fillna(value=np.nan)
        df_orphans.std_frc = df_orphans.std_frc.replace('', np.nan)
        # print(df_orphans['std_min'][2:5])
        # print(type(df_orphans['std_min'][3]))
        # aaa_orphans_std_min = df_orphans.std_min
        # print(df_orphans.std_min)

        # real_bogus_color = df_orphans["real_bogus_probabilities"]
        mediumaquamarine_rgb = tuple(int("099C6C"[i:i+2], 16) for i in (0, 2, 4))
        darkorange_rgb = tuple(int("E88410"[i:i+2], 16) for i in (0, 2, 4))
        lightgrey_rgb = tuple(int("D3D3D3"[i:i+2], 16) for i in (0, 2, 4))
        # real_bogus_blue = df_orphans["real_bogus_probabilities"]*(mediumaquamarine_rgb[2]-lightgrey_rgb[2])+lightgrey_rgb[2]
        real_bogus_red  = df_orphans["real_bogus_probabilities"]*(mediumaquamarine_rgb[0]-lightgrey_rgb[0])+lightgrey_rgb[0]
        real_bogus_grn  = df_orphans["real_bogus_probabilities"]*(mediumaquamarine_rgb[1]-lightgrey_rgb[1])+lightgrey_rgb[1]
        real_bogus_blu  = df_orphans["real_bogus_probabilities"]*(mediumaquamarine_rgb[2]-lightgrey_rgb[2])+lightgrey_rgb[2]
        asteroid_red  = df_orphans["asteroid_probabilities"]*(darkorange_rgb[0]-lightgrey_rgb[0])+lightgrey_rgb[0]
        asteroid_grn  = df_orphans["asteroid_probabilities"]*(darkorange_rgb[1]-lightgrey_rgb[1])+lightgrey_rgb[1]
        asteroid_blu  = df_orphans["asteroid_probabilities"]*(darkorange_rgb[2]-lightgrey_rgb[2])+lightgrey_rgb[2]
        diff_spike_red  = df_orphans["diff_spike_probabilities"]*(darkorange_rgb[0]-lightgrey_rgb[0])+lightgrey_rgb[0]
        diff_spike_grn  = df_orphans["diff_spike_probabilities"]*(darkorange_rgb[1]-lightgrey_rgb[1])+lightgrey_rgb[1]
        diff_spike_blu  = df_orphans["diff_spike_probabilities"]*(darkorange_rgb[2]-lightgrey_rgb[2])+lightgrey_rgb[2]
        df_orphans["real_bogus_color"] = [hex(int(x))[2:]+hex(int(y))[2:]+hex(int(z))[2:] for x,y,z in zip(real_bogus_red, real_bogus_grn, real_bogus_blu)]
        df_orphans["asteroid_color"] = [hex(int(x))[2:]+hex(int(y))[2:]+hex(int(z))[2:] for x,y,z in zip(asteroid_red, asteroid_grn, asteroid_blu)]
        df_orphans["diff_spike_color"] = [hex(int(x))[2:]+hex(int(y))[2:]+hex(int(z))[2:] for x,y,z in zip(diff_spike_red, diff_spike_grn, diff_spike_blu)]

        if nan_fix:
            df_orphans['real_bogus_probabilities']  = df_orphans['probabilities']
            df_orphans['asteroid_probabilities']    = [np.nan for x in df_orphans.det_sep]
            df_orphans['diff_spike_probabilities']  = [np.nan for x in df_orphans.det_sep]

        context['orphans'] = zip(
            list(df_orphans.runcat_id),
            ['%.3f'%x for x in df_orphans.ra_psf],
            ['%.3f'%x for x in df_orphans.dec_psf],
            # ['%.3g'%x for x in df_orphans.ra_std],
            # ['%.3g'%x for x in df_orphans.dec_std],
            ['%.5s'%x for x in df_orphans.q_min],
            ['%.4s'%x for x in df_orphans.q_rb_avg],
            ['%.5s'%x for x in df_orphans.u_min],
            ['%.4s'%x for x in df_orphans.u_rb_avg],
            ['%.5s'%x for x in df_orphans.i_min],
            ['%.4s'%x for x in df_orphans.i_rb_avg],
            ['%.3g'%(x*60*60) for x in df_orphans.std_max],
            ['%.3g'%(x*60*60) for x in df_orphans.std_min],
            ['%.3g'%x for x in df_orphans.std_frc],
            ['%.4g'%x for x in df_orphans.angle_eigs],
            ['%.4s'%x for x in df_orphans.std_ang],
            ['%.3f'%x for x in df_orphans["real_bogus_probabilities"]],
            ['%.3f'%x for x in df_orphans["asteroid_probabilities"]],
            ['%.3f'%x for x in df_orphans["diff_spike_probabilities"]],
            [x for x in df_orphans.real_bogus_color],
            [x for x in df_orphans.asteroid_color],
            [x for x in df_orphans.diff_spike_color],
            [x for x in df_orphans.yes_no],
            [x for x in df_orphans.notes],
        )

        # print(df_orphans["real_bogus_color"])
        # print("BARKBARKBARK")
        # print(["color: "+x for x in df_orphans.real_bogus_color])

        context['orphans_sources_length'] = len(df_orphans)
        if len(df_orphans) == 1:
            context['orphans_sources_plural'] = ""
        else:
            context['orphans_sources_plural'] = "s"

    else:
        context['orphans'] = ""
        context['orphans_sources_length'] = 0
        context['orphans_sources_plural'] = "s"
        # context['orphans_bool'] = False

    time_list.append(time.time())

    if 'history_daily_text_1' in context:
        if field_stats[0] == 0 and "No" not in context['history_daily_text_1']:
            context['blackhub'] = False
        else:
            context['blackhub'] = True
    else:
        context['blackhub'] = True

    time_list.append(time.time())
    print("NightView Times:")
    for i in range(len(time_list)-1):
        print("t"+str(i)+"->t"+str(i+1)+": "+str(time_list[i+1]-time_list[i]))


    return render(request, template_name, context)


def secret_rate_target(request):
    '''
    Rates a target as interesting or not
    '''

    id = request.POST.get('id')
    yes_no = request.POST.get('yes_no')
    notes = request.POST.get('notes')

    time_list = []
    time_list.append(time.time())
    # name = iau_name_from_bgem_id(id)

    # df_orphans = pd.read_csv("./data/history_transients/"+obs_date+"_orphans.csv")
    rated_orphans_exists = True
    try:
        df_orphans = pd.read_csv("./data/history_transients/rated_orphans.csv")
    except:
        rated_orphans_exists = False
        print("'Rated orphans' file not present. Please create!")

    time_list.append(time.time())
    if rated_orphans_exists:
        index_list = df_orphans.index[df_orphans['runcat_id'] == int(id)]

        if len(index_list) == 0:
            df_this_night = pd.read_csv("./data/orphans_20250701_to_20250731.csv")
            this_index = df_this_night.index[df_this_night['runcat_id'] == int(id)][0]
            print(df_this_night.loc[this_index,"runcat_id"])
            # df_orphans = pd.concat([df_orphans, df_this_night.iloc[this_index]]).reset_index(drop=True)
            index = len(df_orphans)
            df_orphans.loc[index] = df_this_night.iloc[this_index]
            print(df_orphans.loc[index,"runcat_id"])

        else:
            index = index_list[0]

        time_list.append(time.time())
        if "yes_no" not in df_orphans.columns:
            df_orphans["yes_no"] = [None]*len(df_orphans)
        if "notes" not in df_orphans.columns:
            df_orphans["notes"] = [None]*len(df_orphans)

        time_list.append(time.time())
        df_orphans.loc[index,"yes_no"] = yes_no
        if len(notes) > 0:
            df_orphans.loc[index,"notes"] = notes
        for column_name in df_orphans.columns:
            if 'Unnamed' in column_name:
                df_orphans = df_orphans.drop(column_name, axis=1)
        time_list.append(time.time())
        df_orphans.to_csv("./data/history_transients/rated_orphans.csv")
        time_list.append(time.time())
        print("\n\n")
        print(df_orphans)
        print(yes_no)
        print(id)
        print(df_orphans.runcat_id)
        print(index)
        print(df_orphans.iloc[index])
        print("\n\n")

    # add_to_GEMTOM(id, name, ra, dec)

    # return redirect(reverse('tom_targets:list'))

    time_list.append(time.time())

    print("rate_target Times:")
    for i in range(len(time_list)-1):
        print("t"+str(i)+"->t"+str(i+1)+": "+str(time_list[i+1]-time_list[i]))
    print("")

    return redirect('/SecretOrphans/')



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
    extragalactic    = []

    if days_since_last_update > 10:
        days_since_last_update = 10

    for this_mjd in np.arange(mjd,mjd-days_since_last_update,-1):

        time_mjd        = Time(this_mjd, format='mjd')
        time_isot       = time_mjd.isot

        extended_date   = time_isot[0:10]
        obs_date      = extended_date[0:4] + extended_date[5:7] + extended_date[8:10]

        dates.append(extended_date)
        mjds.append(this_mjd)

        num_new_transients, num_in_gaia, num_extragalactic, extragalactic_sources, extragalactic_urls, \
            transients_filename, gaia_filename, extragalactic_filename =  get_blackgem_stats(obs_date)

        if float(num_new_transients) > 0: observed.append("Yes")
        else: observed.append("No")

        transients.append(num_new_transients)
        gaia.append(num_in_gaia)
        extragalactic.append(num_extragalactic)
        # extragalactic_sources.append(extragalactic_sources)


    fileOut = "./data/Recent_BlackGEM_History.csv"
    new_history = pd.DataFrame({'Date' : dates, 'MJD' : mjds, 'Observed' : observed, 'Number_Of_Transients' : transients, 'Number_of_Gaia_Crossmatches' : gaia, 'Number_Of_Extragalactic' : extragalactic})

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

    ## Check if it's more than a day ago
    this_nights_datetime = obs_date_to_datetime(night)
    now_datetime = datetime.now()
    time_diff = now_datetime - this_nights_datetime
    time_diff_days = time_diff.total_seconds()/86400
    print(time_diff_days)

    stats_out = "./data/night_"+str(night)+"/" + str(night) + "_fieldstats_final.png"
    fileout = "./data/night_"+str(night)+"/" + str(night) + "_skyview_final.png"

    if time_diff_days < 0.875:
        return [0,0,0,0,0], "../../data/empty_skyview.png"

    if not os.path.exists("./data/night_"+str(night)+"/"):
        os.makedirs("./data/night_"+str(night)+"/")


    if not os.path.exists(stats_out) or time_diff_days < 1.6:
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
        field_dict = {  'num_fields'    : [num_fields],
                        'num_green'     : [num_green],
                        'num_yellow'    : [num_yellow],
                        'num_orange'    : [num_orange],
                        'num_red'       : [num_red]}
        df_field_stats = pd.DataFrame(data=field_dict)

        if time_diff_days < 1.6:
            stats_out = "./data/night_"+str(night)+"/" + str(night) + "_fieldstats_temp.png"

        df_field_stats.to_csv(stats_out)

    else:
        df = pd.read_csv(stats_out)
        field_stats = [df.num_fields[0], df.num_green[0], df.num_yellow[0], df.num_orange[0], df.num_red[0]]
        print(field_stats)

    if not os.path.exists(fileout) or time_diff_days < 1.6:
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

            plt.close("all")

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
            fig = plt.figure(figsize=(8,5), dpi=110)
            ax = fig.add_subplot(111, projection="mollweide")
            ax.grid(True)

        # plt.show()
        # fileOut = "./data/BlackGEM_LastNightsSkymap.png"
        # plt.savefig(fileOut, bbox_inches='tight')
        # plt.close("all")

        # ## -- ORIGINAL - Save the plot to a BytesIO object
        # buffer = io.BytesIO()
        # plt.savefig(buffer, format='png', bbox_inches='tight')
        # buffer.seek(0)
        # image_png = buffer.getvalue()
        # buffer.close()
        #
        # # Encode the image in base64
        # image_base64 = base64.b64encode(image_png)
        # image_base64 = image_base64.decode('utf-8')
        #
        # print("Sky view plotted.")
        # print(night)
        #
        # return field_stats, image_base64

        ## -- NEW - Save the plot to a file

        if time_diff_days < 1.6:
            fileout = "./data/night_"+str(night)+"/" + str(night) + "_skyview_temp.png"

        plt.savefig(fileout)
        print("Sky view plotted.")
        print(night)

    return field_stats, "../../"+fileout



def get_nightly_orphans(night):
    orphans_filename = "./data/history_transients/" + night + "_orphans.csv"

    try:
        df_orphans = pd.read_csv(orphans_filename)

        print("Bark!")
        print(df_orphans)
        return df_orphans
    except:
        print("No orphaned transients that night.")
        return None




class HistoryView(LoginRequiredMixin, TemplateView):
    template_name = 'history.html'

    print("Navigating to history...")


    def get_context_data(self, **kwargs):
        t0 = time.time()
        print("History view: Getting context data...")
        context = super().get_context_data(**kwargs)

        context['history_daily_text_1'], \
            context['history_daily_text_2'], \
            context['history_daily_text_3'], \
            context['images_daily_text_1'], \
            context['extragalactic_sources_id'], \
            context['transients_filename'], \
            context['gaia_filename'], \
            context['extragalactic_filename'] = history_daily()

        t1 = time.time()

        history = blackgem_history()
        dates = list(history.Date)
        dates = [this_date[0:4] + this_date[5:7] + this_date[8:] for this_date in dates]
        # print(dates[0])

        t2 = time.time()

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

        t3 = time.time()

        if days_since_last_update > 0:
            update_history(days_since_last_update)
            history = blackgem_history()
            dates = list(history.Date)
            dates = [this_date[0:4] + this_date[5:7] + this_date[8:] for this_date in dates]

        t4 = time.time()

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

        # df_orphans = get_nightly_orphans(yesterday_date_string)
        # # print(df.orphans.columns)
        # if df_orphans:
        #     context['orphans'] = zip(list(df_orphans.runcat_id), list(df_orphans.q_avg_mag))
        #     # contest['orphans_bool'] = True
        # else:
        #     context['orphans'] = ""
        #     # contest['orphans_bool'] = False
        #

        if 'history_daily_text_1' in context:
            if field_stats[0] == 0 and "No" not in context['history_daily_text_1']:
                context['blackhub'] = False
            else:
                context['blackhub'] = True
        else:
            context['blackhub'] = True

        t5 = time.time()

        print("Times:")
        print("t0->t1:", t1-t0)
        print("t1->t2:", t2-t1)
        print("t2->t3:", t3-t2)
        print("t3->t4:", t4-t3)
        print("t4->t5:", t5-t4)

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

    t0 = time.time()

    ## Get date in different formats
    extended_date = obs_date[:4] + "-" + obs_date[4:6] + "-" + obs_date[6:]
    mjd = int(Time(extended_date + "T00:00:00.00", scale='utc').mjd)

    base_url = 'http://xmm-ssc.irap.omp.eu/claxson/BG_images/'
    # new_base_url = "http://34.90.13.7/quick_selection/"
    new_base_url = "https://blackpearl.blackgem.org/quick_selection/"

    ## Get the list of files from Hugo's server
    t1 = time.time()

    # print("Finding new file list... ", end="\r")
    # new_files = list_files(new_base_url)
    # print("Finding new file list... Found.")
    t2 = time.time()

    transients_filename, gaia_filename, extragalactic_filename = get_transients_filenames(obs_date, url_selection="all")
    t3 = time.time()

    try:
        data = pd.read_csv(transients_filename)
    except Exception as e:
        ## If it doesn't exist, assume BlackGEM didn't observe any transients that night.
        print("No transients on ", obs_date, ":", sep="")
        print(e)
        return "0", "0", "0", ["","","","","",""], "", "", "", ""

    try:
        data_gaia = pd.read_csv(gaia_filename)
        num_in_gaia = str(len(data_gaia))
    except Exception as e:
        ## If it doesn't exist, assume BlackGEM didn't observe any transients that night.
        print("No gaia crossmatches on ", obs_date, ":", sep="")
        print(e)
        gaia_filename = ""
        num_in_gaia = "0"

    try:
        extragalactic_data = pd.read_csv(extragalactic_filename)
    except Exception as e:
        ## If it doesn't exist, assume BlackGEM didn't observe any transients that night.
        print("No extragalactic sources on ", obs_date, ":", sep="")
        print(e)
        extragalactic_filename = ""
    t4 = time.time()

    ## --- Find the details of each extragalactic source ---
    images_urls                 = []
    extragalactic_sources       = []
    extragalactic_sources_id    = []
    extragalactic_sources_ra    = []
    extragalactic_sources_dec   = []
    extragalactic_sources_pipe  = []
    extragalactic_sources_check = []

    old_images = False
    if extragalactic_filename:
        if 'xmm' in extragalactic_filename:
            print("Finding original file list... ", end="\r")
            files = list_files(base_url + obs_date)
            print("Finding original file list... Found.")
            # For each image file...
            old_images = True
            for file in files:
                if ".png" in file:
                    ## Save the URL...
                    images_urls.append(base_url + obs_date + "/" + file[2:])

                    ## If we haven't got the data yet...
                    if file[2:10] not in extragalactic_sources_id:

                        ## Save the ID...
                        extragalactic_sources_id.append(file[2:10])
                        runcat_id_list = list(extragalactic_data['runcat_id'])
                        print(int(file[2:10]) in runcat_id_list)

                        ## And this source is in that data...
                        if int(file[2:10]) in runcat_id_list:
                            ## Save the name, RA, dec, and look for a lightcurve.
                            row_number = runcat_id_list.index(int(file[2:10]))
                            extragalactic_sources_ra.append(  extragalactic_data['ra'][row_number])
                            extragalactic_sources_dec.append( extragalactic_data['dec'][row_number])
                            extragalactic_sources_check.append(True)
                            # extragalactic_sources_jpg.append(get_lightcurve(file[2:10]))
                        else:
                            ## If it's not, state they're all unknown.
                            extragalactic_sources_ra.append("(Unknown)")
                            extragalactic_sources_dec.append("(Unknown)")
                            extragalactic_sources_check.append(False)
                            # extragalactic_sources_jpg.append("")

        ## Files from the new server
        else:
            extragalactic_sources_id = list(extragalactic_data['runcat_id'][extragalactic_data.pipeline != "star"])
            extragalactic_sources_ra = list(extragalactic_data['ra'][extragalactic_data.pipeline != "star"])
            extragalactic_sources_dec = list(extragalactic_data['dec'][extragalactic_data.pipeline != "star"])
            extragalactic_sources_pipe = list(extragalactic_data['pipeline'][extragalactic_data.pipeline != "star"])
            extragalactic_sources_check = [True] * len(extragalactic_sources_ra)

            for runcat_id in list(extragalactic_data['runcat_id'][extragalactic_data.pipeline != "star"]):
                # if str(runcat_id) + "_cutouts_lc.png" in new_files:
                images_urls.append(new_base_url + str(runcat_id) + "_cutouts_lc.png")

    t5 = time.time()

    ## Combine these together.
    extragalactic_sources = [extragalactic_sources_id, extragalactic_sources_ra, extragalactic_sources_dec, extragalactic_sources_pipe, extragalactic_sources_check, old_images]

    ## Sort the images into a list, separated into each source
    images_urls_sorted = []
    for this_source in extragalactic_sources[0]:
        matching = [url for url in images_urls if str(this_source) in url]
        images_urls_sorted.append(matching)

    # print(extragalactic_sources_id)

    num_new_transients  = str(len(data))
    num_extragalactic   = str(len(extragalactic_sources[0]))
    extragalactic_urls  = images_urls_sorted

    t6 = time.time()

    if t0 and t1 and t2 and t3 and t4 and t5 and t6:
        print("get_blackgem_stats times:")
        print("t0->t1:", t1-t0)
        print("t1->t2:", t2-t1)
        print("t2->t3:", t3-t2)
        print("t3->t4:", t4-t3)
        print("t4->t5:", t5-t4)
        print("t5->t6:", t6-t5)

    return num_new_transients, num_in_gaia, num_extragalactic, extragalactic_sources, extragalactic_urls, \
        transients_filename, gaia_filename, extragalactic_filename


## Function for checking last night's BlackGEM history.
def history_daily():
    '''
    Specifically checks last night's observation.
    '''
    t0 = time.time()

    yesterday = date.today() - timedelta(1)
    yesterday_date = yesterday.strftime("%Y%m%d")
    extended_yesterday_date = yesterday.strftime("%Y-%m-%d")
    mjd = int(Time(extended_yesterday_date + "T00:00:00.00", scale='utc').mjd)

    t1 = time.time()

    # url = 'http://xmm-ssc.irap.omp.eu/claxson/BG_images/' + yesterday_date + "/"

    data_length, num_in_gaia, extragalactic_sources_length, extragalactic_sources, images_urls_sorted, \
        transients_filename, gaia_filename, extragalactic_filename = get_blackgem_stats(yesterday_date)

    t2 = time.time()

    # print(url)
    # r = requests.get(url)
    if transients_filename != "":
        result = "BlackGEM observed last night!"
        history_daily_text_1 = "Yes!"

        ## If there was no data, assume BlackGEM didn't observe.
        if (data_length == "0") and (num_in_gaia == "0") and (extragalactic_sources_length == "0"):
            t3 = time.time()
            t4 = time.time()
            t5 = time.time()
            extragalactic_sources_id    = ""
            history_daily_text_1         = "No transients were recorded by BlackGEM last night (" + extended_yesterday_date + ")"
            history_daily_text_2         = ""
            history_daily_text_3         = ""
            # history_daily_text_4         = ""
            images_daily_text_1         = zip([], ["No transients were recorded by BlackGEM last night."])

        else:
            if data_length == "1": data_length_plural = ""; data_length_plural_2 = "s"
            else: data_length_plural = "s"; data_length_plural_2 = "ve"
            if extragalactic_sources_length == "1": extragalactic_sources_plural = ""
            else: extragalactic_sources_plural = "s"

            extragalactic_sources_string = ""
            for source in extragalactic_sources[0]:
                extragalactic_sources_string += str(source) + ", "
            print(extragalactic_sources_string)

            ## Update the most recent row of the recent history
            t3 = time.time()

            current_history = blackgem_history()
            t4 = time.time()

            if current_history["Date"][0] == extended_yesterday_date:
                current_history.loc[0,"Date"]                          = extended_yesterday_date
                current_history.loc[0,"MJD"]                           = mjd
                current_history.loc[0,"Observed"]                      = "Yes"
                current_history.loc[0,"Number_Of_Transients"]          = int(data_length)
                current_history.loc[0,"Number_of_Gaia_Crossmatches"]   = int(num_in_gaia)
                current_history.loc[0,"Number_Of_Extragalactic"]       = int(extragalactic_sources_length)
                fileOut = "./data/Recent_BlackGEM_History.csv"
                current_history.to_csv(fileOut, index=False)

            t5 = time.time()

            extragalactic_sources_id = extragalactic_sources[0]
            history_daily_text_2 = "On " + extended_yesterday_date + " (MJD " + str(mjd) + "), BlackGEM observed " + data_length + " transient" + data_length_plural + ", which ha" + data_length_plural_2 + " " + num_in_gaia + " crossmatches in Gaia (radius 1 arcsec)."
            history_daily_text_3 = "BlackGEM recorded pictures of " + extragalactic_sources_length + " possible extragalactic transient" + extragalactic_sources_plural + "."
            # history_daily_text_4 = extragalactic_sources_string
            # images_daily_text_1 = zip(images_urls_sorted, extragalactic_sources[0])
            # images_daily_text_1 = zip(images_urls_sorted, extragalactic_sources[0], extragalactic_sources[1], extragalactic_sources[2], extragalactic_sources[3])
            images_daily_text_1 = zip(images_urls_sorted, extragalactic_sources[0], extragalactic_sources[1], extragalactic_sources[2], extragalactic_sources[3], extragalactic_sources[4])


    else:
        t3 = time.time()
        t4 = time.time()
        t5 = time.time()
        extragalactic_sources_id    = ""
        transients_filename         = ""
        gaia_filename               = ""
        extragalactic_filename      = ""
        history_daily_text_1         = "No transients were recorded by BlackGEM last night (" + extended_yesterday_date + ")"
        history_daily_text_2         = ""
        history_daily_text_3         = ""
        # history_daily_text_4         = ""
        images_daily_text_1         = zip([], ["No transients were recorded by BlackGEM last night."])

    t6 = time.time()
    if t0 and t1 and t2 and t3 and t4 and t5 and t6:
        print("history_daily times:")
        print("t0->t1:", t1-t0)
        print("t1->t2:", t2-t1)
        print("t2->t3:", t3-t2)
        print("t3->t4:", t4-t3)
        print("t4->t5:", t5-t4)
        print("t5->t6:", t6-t5)

    return history_daily_text_1, history_daily_text_2, history_daily_text_3, images_daily_text_1, extragalactic_sources_id, transients_filename, gaia_filename, extragalactic_filename


def hugo_2_GEMTOM(df):

    bg = authenticate_blackgem()

    for filt in ['q','u','i']:
        params_f = {'runcatids': tuple(list(df.runcat_id)),
                    'filter':filt}


        # min and max magnitude, max rb score, last xtrsrc
        print("computing stats in band %s..."%filt)

        minmag = np.empty(len(df))
        maxmag = np.empty(len(df))
        maxsnr = np.empty(len(df))
        rb_max = np.empty(len(df))
        fwhm = np.empty(len(df))
        xtrsrc = np.zeros(len(df)).astype(int)
        minmag[:] = np.nan
        maxmag[:] = np.nan
        maxsnr[:] = 0
        rb_max[:] = np.nan
        fwhm[:] = np.nan



        qu = """\
        SELECT a.runcat runcat_id
                      ,MIN(x.mag_zogy) as minmag
                      ,MAX(x.mag_zogy) as maxmag
                      ,MAX(x.class_real) as rbmax
                      ,MAX(x.id) as last_xtrsrc
                      ,MAX(x.snr_zogy) as maxsnr
                      ,AVG(x.fwhm_gauss_d) as fwhm
                  FROM extractedsource x, assoc a
                 WHERE a.runcat IN %(runcatids)s
                   AND a.xtrsrc = x.id
                   AND x.filter = '%(filter)s'
                   AND x.mag_zogy < 99
                 GROUP BY a.runcat
        ORDER BY runcat
        """

        query = qu % (params_f)
        l_results = bg.run_query(query)
        df_filt = pd.DataFrame(l_results, columns=['runcat_id',
                                                   'minmag',
                                                   'maxmag',
                                                   'rb_max',
                                                   'xtrsrc',
                                                   'maxsnr',
                                                   'fwhm'])

        i,i1,i2 = np.intersect1d(df['runcat_id'],
                                 df_filt['runcat_id'],
                                 return_indices = True)

        minmag[i1] = df_filt.loc[i2]['minmag']
        maxmag[i1] = df_filt.loc[i2]['maxmag']
        maxsnr[i1] = df_filt.loc[i2]['maxsnr']
        rb_max[i1] = df_filt.loc[i2]['rb_max']
        xtrsrc[i1] = df_filt.loc[i2]['xtrsrc']
        fwhm[i1] = df_filt.loc[i2]['fwhm']

        df['%s_min'%filt] = minmag
        df['%s_max'%filt] = maxmag
        df['%s_dif'%filt] = maxmag-minmag
        df['snr_zogy'] = np.maximum(df['snr'],maxsnr)
        df['%s_xtrsrc'%filt] = xtrsrc
        df['%s_rb'%filt] = rb_max

    df_new = pd.DataFrame({
        'runcat_id' : df.runcat_id,
        'ra' : df.ra,
        'dec' : df.dec,
        'n_datapoints' : df.n_datapoints,
        'snr' : df.snr,
        'q_min' : df.q_min,
        'q_max' : df.q_max,
        'u_min' : df.u_min,
        'u_max' : df.u_max,
        'i_min' : df.i_min,
        'i_max' : df.i_max,
        'q_rb' : df.q_rb,
        'Gmag' : df.Gmag,
        'BP-RP' : df["BP-RP"],
        'Dist' : df.Dist,
        'b_Dist_cds' : df.b_Dist_cds,
        'B_Dist_cdsa' : df.B_Dist_cdsa,
        'Dist_Lower' : df.Dist-df.b_Dist_cds,
        'Dist_Upper' : df.B_Dist_cdsa-df.Dist,
        # 'RPlx' : df.Dist/np.mean([df.Dist+df.b_Dist_cds, df.Dist-df.B_Dist_cdsa]),
    })

    # RPlx = df_new.Dist/np.mean([df_new.Dist+df_new.b_Dist_cds, df_new.Dist-df_new.B_Dist_cdsa])

    RPlx = np.mean([df_new.Dist+df_new.b_Dist_cds, df_new.Dist-df_new.B_Dist_cdsa])
    df_new["Mean_Plx_Error"] = df_new[['Dist_Lower', 'Dist_Upper']].mean(axis=1)
    df_new["RPlx"] = df_new['Dist']/df_new["Mean_Plx_Error"]
    print(df_new["Mean_Plx_Error"].iloc[0])
    print(df_new["RPlx"].iloc[0])
    df_new = df_new.drop(["Dist_Lower", "Dist_Upper", "Mean_Plx_Error"], axis=1)
    df_new['qui_min'] = df_new[['q_min','u_min','i_min']].min(axis=1)

    return df_new

def plot_nightly_hr_diagram(obs_date, gaia_filename):

    # extended_date = obs_date[:4] + "-" + obs_date[4:6] + "-" + obs_date[6:]

    gaia_transients_filename = gaia_filename

    ## Get Gaia data
    df_gaia = pd.read_csv("./data/gaia_nearbystars_hr.txt", sep=",")
    df_gaia["BP-RP"] = df_gaia["BPmag"] - df_gaia["RPmag"]
    # df_gaia = df_gaia.iloc[::20, :]  ## Only grab 1/5th of it for ease
    df_gaia["Gmag"] = df_gaia["Gmag"] - (5 * np.log10((df_gaia["Dist"]*1000)/10))
    df_gaia = df_gaia.drop(df_gaia[df_gaia.Gmag > 16].index)


    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            name="GCNS",
            x=df_gaia["BP-RP"],
            y=df_gaia["Gmag"],
            # y=Gaia_GMag,
            mode="markers",
            marker=dict(color="mediumslateblue", opacity=0.1),
            # opacity=0.2,
            hoverinfo='skip',
        )
    )

    gemtom_gaia_transients_filename = "./data/history_transients/" + obs_date + "_gaia.csv"
    to_process = True

    if os.path.exists(gemtom_gaia_transients_filename):
        print("Attempting to read in any previously-calculated files for this date...")
        df_transients = pd.read_csv(gemtom_gaia_transients_filename)
        to_process = False
        print("Previously-made transients file found for this date. Reading in.")
        if 'q_rb' in df_transients:  snr = "q_rb"
        else:                        snr = "snr"
    else:
        print("No transients file found for this date. Finding from scratch.")
        print("Looking for file (" + gaia_transients_filename + ")...", sep="")
        try:
            df_transients = pd.read_csv(gaia_transients_filename)

            if "http://xmm-ssc" in gaia_transients_filename:
                snr = "q_rb"
            else:
                print("File found; trying to update...", sep="")
                df_transients = hugo_2_GEMTOM(df_transients)
                snr = "q_rb"

            print("File found!")

        except:
            print("File not found; assuming no observations.")
            df_transients = None


            # try:
            #     df_transients = pd.read_csv(gaia_transients_filename)
            #     print("File found; trying to update...", sep="")
            #     df_transients = hugo_2_GEMTOM(df_transients)
            #     snr = "snr"
            #     print("File found!")
            # except:
            #     try:
            #         print("File not found; looking on the old server...")
            #         df_transients = pd.read_csv("http://xmm-ssc.irap.omp.eu/claxson/BG_images/" + obs_date + "/" + extended_date + "_gw_BlackGEM_transients_gaia.csv")
            #         snr = "q_rb"
            #         print("File found!")
            #     except:
            #         print("File not found; assuming no observations.")
                    # df_transients = None

    if df_transients is not None:
        if to_process:
            df_transients = df_transients[df_transients[snr] > 0.8]
            df_transients = df_transients[df_transients["RPlx"] > 2]
            df_transients["M_G"] = df_transients["Gmag"] - (5 * np.log10(df_transients["Dist"]/10))
            df_transients["url"] = 'http://gemtom.blackgem.org/transients/' + df_transients['runcat_id'].astype(str)

            ## Find the difference. Warning: q_max/min is only actual detections, not upper limits.
            print("Finding qui_diff...")
            # qui_max = []
            qui_diff = []
            for id, q_max, u_max, i_max, qui_min, q_min, u_min, i_min in zip( \
                df_transients["runcat_id"],   \
                df_transients["q_max"],   \
                df_transients["u_max"],   \
                df_transients["i_max"],   \
                df_transients["qui_min"], \
                df_transients["q_min"],   \
                df_transients["u_min"],   \
                df_transients["i_min"]):

                if   qui_min == q_min: qui_diff.append(q_max-q_min)
                elif qui_min == u_min: qui_diff.append(u_max-u_min)
                elif qui_min == i_min: qui_diff.append(i_max-i_min)
                else: qui_diff.append(None) #print("Failed on ", i, id, q_max, u_max, i_max, qui_min, q_min, u_min, i_min)

            # df_transients["qui_max"] = qui_max
            # df_transients["qui_diff"] = df_transients["qui_max"] - df_transients["qui_min"]
            df_transients["qui_diff"] = qui_diff
            print("Found qui_diff.")
            # df_transients["qui_diff"] = df_transients["magmin"]

            ## Drop rows with a maximum mag of 99
            df_transients = df_transients.drop(df_transients[df_transients.qui_diff > 30].index)
            df_transients = df_transients.drop(df_transients[np.isfinite(df_transients.qui_diff) == False].index)
            df_transients.to_csv(gemtom_gaia_transients_filename)


        fig.add_trace(
            go.Scatter(
                name            = "2024-09-28",
                x               = df_transients["BP-RP"],
                y               = df_transients["M_G"],
                mode            = 'markers',
                hovertemplate   =
                    'ID: %{customdata[0]}<br>' +
                    'BP-RP: %{x:.3f}<br>' +
                    'G Mag: %{y:.3f}<br>' +
                    'Dist: %{customdata[1]:.0f}pc (%{customdata[2]:.0f},%{customdata[3]:.0f})<br>'
                    # 'RB Score: %{customdata[4]:.2f}<br>'
                    'Mag Diff: %{customdata[5]:.2f}<br>'
                    ,
                customdata      = [(df_transients['runcat_id'].iloc[i], df_transients["Dist"].iloc[i], df_transients["b_Dist_cds"].iloc[i], df_transients["B_Dist_cdsa"].iloc[i], df_transients[snr].iloc[i], df_transients["qui_diff"].iloc[i]) for i in range(len(df_transients['runcat_id']))],
                # customdata = [(df_2['x.magerr_zogy'].iloc[i], df_2['x.flux_zogy'].iloc[i], df_2['x.fluxerr_zogy'].iloc[i]) for i in range(len(df_2['x.fluxerr_zogy']))]
                text            = df_transients['url'],
            ),
        )
        print("Finding possible CVs...")
        df_cv = find_possible_CVs(df_transients)
        print("Possible CVs found.")

        fig.add_trace(
            go.Scatter(
                name="Potential CVs",
                x = df_cv["BP-RP"],
                y = df_cv["M_G"],
                mode = 'markers',
                marker=dict(color="#ff7f0e"),
                hovertemplate   =
                    'ID: %{customdata[0]}<br>' +
                    'BP-RP: %{x:.3f}<br>' +
                    'G Mag: %{y:.3f}<br>' +
                    'Dist: %{customdata[1]:.0f}pc (%{customdata[2]:.0f},%{customdata[3]:.0f})<br>'
                    # 'RB Score: %{customdata[4]:.2f}<br>'
                    'Mag Diff: %{customdata[5]:.2f}<br>'
                    ,
                customdata      = [(df_cv['runcat_id'].iloc[i], df_cv["Dist"].iloc[i], df_cv["b_Dist_cds"].iloc[i], df_cv["B_Dist_cdsa"].iloc[i], df_cv[snr].iloc[i], df_cv["qui_diff"].iloc[i]) for i in range(len(df_cv['runcat_id']))],
                text            = df_cv['url'],

            )
        )


        fig.update_layout(
            yaxis = dict(autorange="reversed"),
            height=510,
            margin=dict(t=0, l=0, r=0, b=100),  # Set margins to reduce whitespace
            # title="HR Diagram",
            xaxis_title="BP-RP",
            yaxis_title="GMag",
            showlegend=False,
            sliders=[{
                'active': 0,
                'currentvalue': {'prefix': 'Mag Diff > '},
                'pad': {'t': 50},
                'steps': [
                    {
                        'method': 'update',
                        'label': str(threshold),
                        'args': [
                            {
                                'x': [
                                    df_gaia["BP-RP"],
                                    df_transients["BP-RP"][df_transients["qui_diff"] >= threshold],
                                    df_cv["BP-RP"][df_cv["qui_diff"] >= threshold]
                                ],
                                'y': [
                                    df_gaia["Gmag"],
                                    df_transients["M_G"][df_transients["qui_diff"] >= threshold],
                                    df_cv["M_G"][  df_cv["qui_diff"] >= threshold]
                                ],
                                'hovertemplate': [
                                    None,
                                    'ID: %{customdata[0]}<br>' +
                                    'BP-RP: %{x:.3f}<br>' +
                                    'G Mag: %{y:.3f}<br>' +
                                    'Dist: %{customdata[1]:.0f}pc (%{customdata[2]:.0f},%{customdata[3]:.0f})<br>' +
                                    'Mag Diff: %{customdata[5]:.2f}<br>',
                                    'ID: %{customdata[0]}<br>' +
                                    'BP-RP: %{x:.3f}<br>' +
                                    'G Mag: %{y:.3f}<br>' +
                                    'Dist: %{customdata[1]:.0f}pc (%{customdata[2]:.0f},%{customdata[3]:.0f})<br>'
                                    # 'RB Score: %{customdata[4]:.2f}<br>'
                                    'Mag Diff: %{customdata[5]:.2f}<br>'
                                ],
                                'customdata': [
                                    None,
                                    [(df_transients['runcat_id'][df_transients["qui_diff"] >= threshold].iloc[i], df_transients["Dist"][df_transients["qui_diff"] >= threshold].iloc[i], df_transients["b_Dist_cds"][df_transients["qui_diff"] >= threshold].iloc[i], df_transients["B_Dist_cdsa"][df_transients["qui_diff"] >= threshold].iloc[i], df_transients[snr][df_transients["qui_diff"] >= threshold].iloc[i], df_transients["qui_diff"][df_transients["qui_diff"] >= threshold].iloc[i]) for i in range(len(df_transients['runcat_id'][df_transients["qui_diff"] >= threshold]))],
                                    [(df_cv['runcat_id'][df_cv["qui_diff"] >= threshold].iloc[i], df_cv["Dist"][df_cv["qui_diff"] >= threshold].iloc[i], df_cv["b_Dist_cds"][df_cv["qui_diff"] >= threshold].iloc[i], df_cv["B_Dist_cdsa"][df_cv["qui_diff"] >= threshold].iloc[i], df_cv[snr][df_cv["qui_diff"] >= threshold].iloc[i], df_cv["qui_diff"][df_cv["qui_diff"] >= threshold].iloc[i]) for i in range(len(df_cv['runcat_id'][df_cv["qui_diff"] >= threshold]))],
                                ],
                                'text' : [
                                    None,
                                    df_transients['url'][df_transients["qui_diff"] >= threshold],
                                    df_cv['url'][df_cv["qui_diff"] >= threshold],
                                ],
                            },
                        ]
                    } for threshold in range(0, 4, 1)  # Adjust slider granularity as needed
                ]
            }]
        )

        fig['layout']['sliders'][0]['pad']=dict(t=20,)
        print("HR Diagram plotted.")

    else:
        print("No Gaia sources this night.")

    return fig

def find_possible_CVs(df):

    cv_lower = 0.1
    cv_break = 0.4
    cv_upper = 1.8

    def cv_cutoff(x):
        return 10.4-(11*np.exp(-x))

    df_cv_1 = df.copy()
    df_cv_1 = df_cv_1[df_cv_1["BP-RP"] > cv_lower]
    df_cv_1 = df_cv_1[df_cv_1["BP-RP"] < cv_break]
    df_cv_1 = df_cv_1[df_cv_1["M_G"] > cv_cutoff(cv_break)]

    df_cv_2 = df.copy()
    df_cv_2 = df_cv_2[df_cv_2["BP-RP"] > cv_break]
    df_cv_2 = df_cv_2[df_cv_2["BP-RP"] < cv_upper]
    df_cv_2 = df_cv_2[df_cv_2["M_G"] > cv_cutoff(df_cv_2["BP-RP"])]

    df_cv = pd.concat([df_cv_1, df_cv_2]).reset_index(drop=True)

    return df_cv


def download_possible_CVs(request):

    obs_date = request.POST.get('obs_date')

    extended_date = obs_date[:4] + "-" + obs_date[4:6] + "-" + obs_date[6:]

    gaia_transients_filename = get_transients_filenames(obs_date, 'gaia')

    df_transients = pd.read_csv(gaia_transients_filename)
    try:
        df_transients = df_transients[df_transients["q_rb"] > 0.8]
    except:
        df_transients = df_transients[df_transients["snr"] > 0.8]
    df_transients["M_G"] = df_transients["Gmag"] - (5 * np.log10(df_transients["Dist"]/10))

    df_cv = find_possible_CVs(df_transients)

    # Convert DataFrame to CSV string
    csv_string = df_cv.to_csv(index=False)

    # Create the HttpResponse object with appropriate headers for CSV.
    response = HttpResponse(csv_string, content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="BlackGEM_Potential_CVs_' + obs_date + '.csv"'

    # Return the response which prompts the download
    return response


def download_rated_orphans(request):

    df_orphans = blackgem_rated_orphans()

    # Convert DataFrame to CSV string
    csv_string = df_orphans.to_csv(index=False)

    obs_date = date.today()
    obs_date = obs_date.strftime("%Y%m%d")

    # Create the HttpResponse object with appropriate headers for CSV.
    response = HttpResponse(csv_string, content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="GEMTOM_Rated_Orphans_' + obs_date + '.csv"'

    # Return the response which prompts the download
    return response


def scatter_plot_view(request):

    import plotly.io as pio

    x_data = [1, 2, 3, 4, 5]
    y_data = [10, 11, 12, 13, 14]
    urls = [
        "http://127.0.0.1:8000/transients/1",
        "http://127.0.0.1:8000/transients/2",
        "http://127.0.0.1:8000/transients/3",
        "http://127.0.0.1:8000/transients/4",
        "http://127.0.0.1:8000/transients/5"
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers',
        marker=dict(size=12, color='blue'),
        text=urls,
        hoverinfo='text'
    ))

    plot_html = pio.to_html(fig, full_html=False)

    return render(request, 'scatter_plot.html', {'plot_html': plot_html})






def get_any_nights_files(obs_date):

    ## Check if it's more than a day ago
    this_nights_datetime = obs_date_to_datetime(str(obs_date))
    now_datetime = datetime.now()
    time_diff = now_datetime - this_nights_datetime
    time_diff_days = time_diff.total_seconds()/86400
    print(time_diff_days)
    extended_date = obs_date[:4] + "-" + obs_date[4:6] + "-" + obs_date[6:]

    filenames_out = "./data/night_"+str(obs_date)+"/" + str(obs_date) + "_filenames_final.json"

    if time_diff_days < 0.875:
        context = {
            "obs_date"                      : obs_date,
            "extended_date"                 : extended_date,
            "mjd"                           : int(Time(extended_date + "T00:00:00.00", scale='utc').mjd),
        }
        with open("./data/future_filenames.json", 'r') as f:
            context2 = json.load(f)

        context = {**context, **context2}

    else:
        if not os.path.exists("./data/night_"+str(obs_date)+"/"):
            os.makedirs("./data/night_"+str(obs_date)+"/")

        if not os.path.exists(filenames_out) or time_diff_days < 1.6:

            data_length, num_in_gaia, extragalactic_sources_length, extragalactic_sources, images_urls_sorted, \
                transients_filename, gaia_filename, extragalactic_filename = get_blackgem_stats(obs_date)

            if data_length == "1": data_length_plural = ""; data_length_plural_2 = "s"
            else: data_length_plural = "s"; data_length_plural_2 = "ve"
            if num_in_gaia == "1": gaia_plural = ""
            else: gaia_plural = "es"
            if extragalactic_sources_length == "1": extragalactic_sources_plural = ""
            else: extragalactic_sources_plural = "s"

            context = {
                "obs_date"                      : obs_date,
                "extended_date"                 : extended_date,
                "mjd"                           : int(Time(extended_date + "T00:00:00.00", scale='utc').mjd),
                "data_length"                   : data_length,
                "num_in_gaia"                   : num_in_gaia,
                "extragalactic_sources_length"  : extragalactic_sources_length,
                "extragalactic_sources_id"      : extragalactic_sources[0],
                "extragalactic_sources_ra"      : extragalactic_sources[1],
                "extragalactic_sources_dec"     : extragalactic_sources[2],
                "extragalactic_sources_pipe"    : extragalactic_sources[3],
                "extragalactic_sources_check"   : extragalactic_sources[4],
                "old_images"                    : extragalactic_sources[5],
                "data_length_plural"            : data_length_plural,
                "data_length_plural_2"          : data_length_plural_2,
                "gaia_plural"                   : gaia_plural,
                "extragalactic_sources_plural"  : extragalactic_sources_plural,
                "images_urls_sorted"            : images_urls_sorted,
            }

            context['transients_filename']      = transients_filename
            context['gaia_filename']            = gaia_filename
            context['extragalactic_filename']   = extragalactic_filename

            print(context)

            with open(filenames_out, 'w') as f:
                json.dump(context, f)

        else:
            with open(filenames_out, 'r') as f:
                context = json.load(f)


    return context

def get_any_nights_context(obs_date):
    response = "You're looking at BlackGEM date %s."

    context2 = get_any_nights_files(obs_date)

    context = {
        "response"                      : response % obs_date
    }

    context = {**context, **context2}

    data_length = context["data_length"]
    num_in_gaia = context["num_in_gaia"]
    extragalactic_sources_length = context["extragalactic_sources_length"]
    images_urls_sorted = context["images_urls_sorted"]

    extragalactic_sources_id    = context["extragalactic_sources_id"]
    extragalactic_sources_ra    = context["extragalactic_sources_ra"]
    extragalactic_sources_dec   = context["extragalactic_sources_dec"]
    extragalactic_sources_pipe  = context["extragalactic_sources_pipe"]
    extragalactic_sources_check = context["extragalactic_sources_check"]


    if (data_length == "0") and (num_in_gaia == "0") and (extragalactic_sources_length == "0") and (extragalactic_sources_id == "") and (images_urls_sorted == ""):
        observed_string = "No transients were recorded by BlackGEM that night."
        images_daily_text_1 = zip([], ["No transients were recorded by BlackGEM that night."])
    else:
        observed_string = "Yes!"
        images_daily_text_1 = zip(images_urls_sorted, extragalactic_sources_id, extragalactic_sources_ra, extragalactic_sources_dec, extragalactic_sources_pipe, extragalactic_sources_check)
    context['observed']                 = observed_string
    context['images_daily_text_1']      = images_daily_text_1

    field_stats, image_base64 = get_any_nights_sky_plot(obs_date)
    context['num_fields']       = field_stats[0]
    context['green_fields']     = field_stats[1]
    context['yellow_fields']    = field_stats[2]
    context['orange_fields']    = field_stats[3]
    context['red_fields']       = field_stats[4]
    context['plot_image']       = image_base64
    context['prev_night']       = datetime.strftime(obs_date_to_datetime(obs_date) - timedelta(1), '%Y%m%d')
    context['next_night']       = datetime.strftime(obs_date_to_datetime(obs_date) + timedelta(1), '%Y%m%d')

    return context


@login_required
def NightView(request, obs_date):
    '''
    Finds and displays data from a certain date.
    '''
    print("Starting NightView...")
    time_list = []
    time_list.append(time.time())

    obs_date = str(obs_date)

    context = get_any_nights_context(obs_date)

    time_list.append(time.time())

    df_orphans = get_nightly_orphans(obs_date)

    time_list.append(time.time())

    rated_orphans = False
    try:
        df_orphans_all = pd.read_csv("./data/history_transients/rated_orphans.csv")
        rated_orphans = True

    except:
        print("'Rated Orphans' file not present. Please create!")

    if df_orphans is not None:
        if rated_orphans:
            yes_no_list = []
            notes_list = []

            for runcat_id in df_orphans.runcat_id:
                if runcat_id in list(df_orphans_all.runcat_id):
                    index = df_orphans_all.index[df_orphans_all['runcat_id'] == int(runcat_id)]

                    yes_no_list.append(df_orphans_all['yes_no'].values[index][0])
                    notes_list.append(df_orphans_all['notes'].values[index][0])
                else:
                    yes_no_list.append("")
                    notes_list.append("")

            df_orphans['yes_no'] = yes_no_list
            df_orphans['notes'] = notes_list

        else:
            df_orphans['yes_no'] = [""]*len(df_orphans)
            df_orphans['notes']  = [""]*len(df_orphans)

        df_orphans = df_orphans.sort_values(by=['i_rb_avg'], ascending=False)
        df_orphans = df_orphans.sort_values(by=['u_rb_avg'], ascending=False)
        df_orphans = df_orphans.sort_values(by=['q_rb_avg'], ascending=False)
        df_orphans = df_orphans.fillna('')

        if "std_max" not in df_orphans.columns:
            df_orphans['std_max'] = [np.nan for x in df_orphans.det_sep]
            df_orphans['std_min'] = [np.nan for x in df_orphans.det_sep]
            df_orphans['std_frc'] = [np.nan for x in df_orphans.det_sep]
            df_orphans['std_ang'] = [np.nan for x in df_orphans.det_sep]

        if "probabilities" not in df_orphans.columns:
            df_orphans['probabilities'] = [0 for x in df_orphans.det_sep]

        nan_fix = False
        if "real_bogus_probabilities" not in df_orphans.columns:
            nan_fix = True
            df_orphans['real_bogus_probabilities']  = df_orphans['probabilities']
            df_orphans['asteroid_probabilities']    = [0 for x in df_orphans.det_sep]
            df_orphans['diff_spike_probabilities']  = [0 for x in df_orphans.det_sep]

        df_orphans = df_orphans.sort_values(by=['real_bogus_probabilities'], ascending=False)


        # df_orphans.std_min = df_orphans.std_min.fillna(value=np.nan)
        df_orphans.std_min = df_orphans.std_min.replace('', np.nan)
        # df_orphans.std_frc = df_orphans.std_min.fillna(value=np.nan)
        df_orphans.std_frc = df_orphans.std_frc.replace('', np.nan)
        # print(df_orphans['std_min'][2:5])
        # print(type(df_orphans['std_min'][3]))
        # aaa_orphans_std_min = df_orphans.std_min
        # print(df_orphans.std_min)

        # real_bogus_color = df_orphans["real_bogus_probabilities"]
        mediumaquamarine_rgb = tuple(int("099C6C"[i:i+2], 16) for i in (0, 2, 4))
        darkorange_rgb = tuple(int("E88410"[i:i+2], 16) for i in (0, 2, 4))
        lightgrey_rgb = tuple(int("D3D3D3"[i:i+2], 16) for i in (0, 2, 4))
        # real_bogus_blue = df_orphans["real_bogus_probabilities"]*(mediumaquamarine_rgb[2]-lightgrey_rgb[2])+lightgrey_rgb[2]
        real_bogus_red  = df_orphans["real_bogus_probabilities"]*(mediumaquamarine_rgb[0]-lightgrey_rgb[0])+lightgrey_rgb[0]
        real_bogus_grn  = df_orphans["real_bogus_probabilities"]*(mediumaquamarine_rgb[1]-lightgrey_rgb[1])+lightgrey_rgb[1]
        real_bogus_blu  = df_orphans["real_bogus_probabilities"]*(mediumaquamarine_rgb[2]-lightgrey_rgb[2])+lightgrey_rgb[2]
        asteroid_red  = df_orphans["asteroid_probabilities"]*(darkorange_rgb[0]-lightgrey_rgb[0])+lightgrey_rgb[0]
        asteroid_grn  = df_orphans["asteroid_probabilities"]*(darkorange_rgb[1]-lightgrey_rgb[1])+lightgrey_rgb[1]
        asteroid_blu  = df_orphans["asteroid_probabilities"]*(darkorange_rgb[2]-lightgrey_rgb[2])+lightgrey_rgb[2]
        diff_spike_red  = df_orphans["diff_spike_probabilities"]*(darkorange_rgb[0]-lightgrey_rgb[0])+lightgrey_rgb[0]
        diff_spike_grn  = df_orphans["diff_spike_probabilities"]*(darkorange_rgb[1]-lightgrey_rgb[1])+lightgrey_rgb[1]
        diff_spike_blu  = df_orphans["diff_spike_probabilities"]*(darkorange_rgb[2]-lightgrey_rgb[2])+lightgrey_rgb[2]
        df_orphans["real_bogus_color"] = [hex(int(x))[2:]+hex(int(y))[2:]+hex(int(z))[2:] for x,y,z in zip(real_bogus_red, real_bogus_grn, real_bogus_blu)]
        df_orphans["asteroid_color"] = [hex(int(x))[2:]+hex(int(y))[2:]+hex(int(z))[2:] for x,y,z in zip(asteroid_red, asteroid_grn, asteroid_blu)]
        df_orphans["diff_spike_color"] = [hex(int(x))[2:]+hex(int(y))[2:]+hex(int(z))[2:] for x,y,z in zip(diff_spike_red, diff_spike_grn, diff_spike_blu)]

        if nan_fix:
            df_orphans['real_bogus_probabilities']  = df_orphans['probabilities']
            df_orphans['asteroid_probabilities']    = [np.nan for x in df_orphans.det_sep]
            df_orphans['diff_spike_probabilities']  = [np.nan for x in df_orphans.det_sep]

        context['orphans'] = zip(
            list(df_orphans.runcat_id),
            ['%.3f'%x for x in df_orphans.ra_psf],
            ['%.3f'%x for x in df_orphans.dec_psf],
            # ['%.3g'%x for x in df_orphans.ra_std],
            # ['%.3g'%x for x in df_orphans.dec_std],
            ['%.5s'%x for x in df_orphans.q_min],
            ['%.4s'%x for x in df_orphans.q_rb_avg],
            ['%.5s'%x for x in df_orphans.u_min],
            ['%.4s'%x for x in df_orphans.u_rb_avg],
            ['%.5s'%x for x in df_orphans.i_min],
            ['%.4s'%x for x in df_orphans.i_rb_avg],
            ['%.3g'%(x*60*60) for x in df_orphans.std_max],
            ['%.3g'%(x*60*60) for x in df_orphans.std_min],
            ['%.3g'%x for x in df_orphans.std_frc],
            ['%.4g'%x for x in df_orphans.angle_eigs],
            ['%.4s'%x for x in df_orphans.std_ang],
            ['%.3f'%x for x in df_orphans["real_bogus_probabilities"]],
            ['%.3f'%x for x in df_orphans["asteroid_probabilities"]],
            ['%.3f'%x for x in df_orphans["diff_spike_probabilities"]],
            [x for x in df_orphans.real_bogus_color],
            [x for x in df_orphans.asteroid_color],
            [x for x in df_orphans.diff_spike_color],
            [x for x in df_orphans.yes_no],
            [x for x in df_orphans.notes],
        )

        # print(df_orphans["real_bogus_color"])
        # print("BARKBARKBARK")
        # print(["color: "+x for x in df_orphans.real_bogus_color])

        context['orphans_sources_length'] = len(df_orphans)
        if len(df_orphans) == 1:
            context['orphans_sources_plural'] = ""
        else:
            context['orphans_sources_plural'] = "s"

    else:
        context['orphans'] = ""
        context['orphans_sources_length'] = 0
        context['orphans_sources_plural'] = "s"
        # context['orphans_bool'] = False

    time_list.append(time.time())

    if 'history_daily_text_1' in context:
        if field_stats[0] == 0 and "No" not in context['history_daily_text_1']:
            context['blackhub'] = False
        else:
            context['blackhub'] = True
    else:
        context['blackhub'] = True

    time_list.append(time.time())
    print("NightView Times:")
    for i in range(len(time_list)-1):
        print("t"+str(i)+"->t"+str(i+1)+": "+str(time_list[i+1]-time_list[i]))


    return render(request, "history/index.html", context)


@login_required
def NightView_Gaia(request, obs_date):
    '''
    Finds and displays data from a certain date.
    '''
    print("Starting NightView...")
    time_list = []
    time_list.append(time.time())

    obs_date = str(obs_date)

    context = get_any_nights_context(obs_date)

    fig = plot_nightly_hr_diagram(obs_date, context["gaia_filename"])
    time_list.append(time.time())
    lightcurve = plot(fig, output_type='div')
    context['lightcurve']       = lightcurve

    time_list.append(time.time())

    if 'history_daily_text_1' in context:
        if field_stats[0] == 0 and "No" not in context['history_daily_text_1']:
            context['blackhub'] = False
        else:
            context['blackhub'] = True
    else:
        context['blackhub'] = True

    time_list.append(time.time())
    print("NightView Times:")
    for i in range(len(time_list)-1):
        print("t"+str(i)+"->t"+str(i+1)+": "+str(time_list[i+1]-time_list[i]))


    return render(request, "history/index_gaia.html", context)


def all_stats_from_bgem_id(bgem_id):

    ## Get the name, ra, and dec:
    bg = authenticate_blackgem()

    print("Starting query...")

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

    try:
        l_results = bg.run_query(query)
    except Exception as e:
        l_results = None
        print("Error!")
        print(e)

    if l_results:
        return [l_results[0][1], l_results[0][2], l_results[0][3]]
    else:
        return [None, None, None]


@login_required
def url_to_GEMTOM(request, bgem_id, custom_name=False):
    '''
    Imports a target from the bgem_id
    '''
    print("Running url_to_GEMTOM...")

    name, ra, dec = all_stats_from_bgem_id(bgem_id)

    if custom_name:
        name = custom_name

    if name == None or name == "None":
        messages.error(
            request,
            'Upload failed; are you sure that ' + str(bgem_id) + ' is a valid BlackGEM ID? \
            If so, check that the connection to the transient server is online.'
        )
        return redirect(reverse('tom_targets:list'))

    created, existing_target_id = add_to_GEMTOM(bgem_id, name, ra, dec)

    add_to_GEMTOM_message(request, name, created, existing_target_id)

    # return redirect(reverse('tom_targets:list'))
    return redirect(f'/targets/{existing_target_id}')

@login_required
def name_to_GEMTOM(request, bgem_id, custom_name):
    # url_to_GEMTOM(request, bgem_id, custom_name)
    '''
    Imports a target from the bgem_id and adds it as a custom name
    '''
    print("Running url_to_GEMTOM...")

    name, ra, dec = all_stats_from_bgem_id(bgem_id)

    if custom_name:
        name = custom_name
        custom_name = custom_name.replace(" ","")
        custom_name = custom_name.replace("%20","")
        source_is_TNS = False
        if (custom_name[:2] == "AT") or (custom_name[:2] == "SN"):
            tns_prefix = custom_name[:2]
            tns_name = custom_name[2:]
            source_is_TNS = True

    if name == None or name == "None":
        messages.error(
            request,
            'Upload failed; are you sure that ' + str(bgem_id) + ' is a valid BlackGEM ID? \
            If so, check that the connection to the transient server is online.'
        )
        return redirect(reverse('tom_targets:list'))

    if source_is_TNS:   created, existing_target_id = add_to_GEMTOM(bgem_id, name, ra, dec, tns_prefix, tns_name)
    else:               created, existing_target_id = add_to_GEMTOM(bgem_id, name, ra, dec)

    return redirect(f'/targets/')


def history_to_GEMTOM(request):
    '''
    Imports a target from the History tab
    '''
    print("Running history_to_GEMTOM...")


    bgem_id = request.POST.get('id')
    name = request.POST.get('name')
    ra = request.POST.get('ra')
    dec = request.POST.get('dec')
    tns_prefix  = request.POST.get('tns_prefix')
    tns_name    = request.POST.get('tns_name')

    print("TNS Details:")
    print(tns_prefix)
    print(tns_name)

    if tns_prefix and tns_name:
        name = tns_prefix + " " + tns_name
    else:
        name = iau_name_from_bgem_id(bgem_id)


    created, existing_target_id = add_to_GEMTOM(bgem_id, name, ra, dec, tns_prefix, tns_name)

    add_to_GEMTOM_message(request, name, created, existing_target_id)

    # return redirect(reverse('tom_targets:list'))
    return redirect(f'/targets/{existing_target_id}')

def add_to_GEMTOM_message(request, name, created, existing_target_id):

    print("created:")
    print(created)

    if created:
        messages.success(
            request,
            'Target created: ' + name + " (/targets/" + str(existing_target_id) + "/)"
        )
    else:
        messages.warning(
            request,
            'Target already exists: ' + name + " (/targets/" + str(existing_target_id) + "/)"
        )

    return


def rate_target(request):
    '''
    Rates a target as interesting or not
    '''

    id = request.POST.get('id')
    yes_no = request.POST.get('yes_no')
    notes = request.POST.get('notes')
    obs_date = request.POST.get('night')

    true_rate_target(id, yes_no, notes, obs_date)

    return redirect('/history/' + obs_date + "/")

def true_rate_target(id, yes_no, notes, obs_date):

    time_list = []
    time_list.append(time.time())
    # name = iau_name_from_bgem_id(id)

    # df_orphans = pd.read_csv("./data/history_transients/"+obs_date+"_orphans.csv")
    rated_orphans_exists = True
    try:
        df_orphans = pd.read_csv("./data/history_transients/rated_orphans.csv")
    except:
        rated_orphans_exists = False
        print("'Rated orphans' file not present. Please create!")

    time_list.append(time.time())
    if rated_orphans_exists:
        index_list = df_orphans.index[df_orphans['runcat_id'] == int(id)]

        if len(index_list) == 0:
            df_this_night = pd.read_csv("./data/history_transients/"+obs_date+"_orphans.csv")
            this_index = df_this_night.index[df_this_night['runcat_id'] == int(id)][0]
            print(df_this_night.loc[this_index,"runcat_id"])
            # df_orphans = pd.concat([df_orphans, df_this_night.iloc[this_index]]).reset_index(drop=True)
            index = len(df_orphans)
            df_orphans.loc[index] = df_this_night.iloc[this_index]
            print(df_orphans.loc[index,"runcat_id"])

        else:
            index = index_list[0]

        time_list.append(time.time())
        if "yes_no" not in df_orphans.columns:
            df_orphans["yes_no"] = [None]*len(df_orphans)
        if "notes" not in df_orphans.columns:
            df_orphans["notes"] = [None]*len(df_orphans)

        time_list.append(time.time())
        df_orphans.loc[index,"yes_no"] = yes_no
        if len(notes) > 0:
            df_orphans.loc[index,"notes"] = notes
        for column_name in df_orphans.columns:
            if 'Unnamed' in column_name:
                df_orphans = df_orphans.drop(column_name, axis=1)
        time_list.append(time.time())
        df_orphans.to_csv("./data/history_transients/rated_orphans.csv")
        time_list.append(time.time())
        print("\n\n")
        print(df_orphans)
        print(yes_no)
        print(id)
        print(df_orphans.runcat_id)
        print(index)
        print(df_orphans.iloc[index])
        print("\n\n")

    # add_to_GEMTOM(id, name, ra, dec)

    # return redirect(reverse('tom_targets:list'))

    time_list.append(time.time())

    print("rate_target Times:")
    for i in range(len(time_list)-1):
        print("t"+str(i)+"->t"+str(i+1)+": "+str(time_list[i+1]-time_list[i]))
    print("")



    return redirect('/history/' + obs_date + "/")


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
            'first_obs'     : ['2010-01-01'],
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
            # 'iauname_short' : [0],
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


def blackgem_rated_orphans():
    '''
    Fetches BlackGEM's rated orphans and returns as a pandas dataframe
    '''

    filepath = "./data/history_transients/rated_orphans.csv"
    if os.path.isfile(filepath):
        rated_orphans = pd.read_csv(filepath)
    else:
        rated_orphans = pd.DataFrame({
            'runcat_id'             : [0],
            'ra'                    : [0],
            'dec'                   : [0],
            'ra_psf'                : [0],
            'dec_psf'               : [0],
            # 'ra_std'                : [0],
            # 'dec_std'               : [0],
            'xtrsrc'                : [0],
            'n_datapoints'          : [0],
            'q_min'                 : [0],
            'q_max'                 : [0],
            'q_avg'                 : [0],
            'q_rb_avg'              : [0],
            'q_num'                 : [0],
            'u_min'                 : [0],
            'u_max'                 : [0],
            'u_avg'                 : [0],
            'u_rb_avg'              : [0],
            'u_num'                 : [0],
            'i_min'                 : [0],
            'i_max'                 : [0],
            'i_avg'                 : [0],
            'i_rb_avg'              : [0],
            'i_num'                 : [0],
            'all_num_datapoints'    : [0],
            'det_sep'               : [0],
            'yes_no'                : [""],
            'notes'                 : [""],
        })


        rated_orphans.to_csv(filepath, index=False)

    return rated_orphans

def obs_date_to_datetime(obs_date):
    return datetime.strptime(obs_date, '%Y%m%d')

def datetime_to_obsdate(datetime):
    return datetime.strftime('%Y%m%d')

def does_url_exist(url):
    print("Checking url:", url, end="")
    response = requests.get(url)
    if response.status_code == 200:
        print(" - Found!")
        return True
    else:
        print(" - Not found :(")
        return False

def get_transients_filenames(obs_date, url_selection="all"):
    '''
    Gets the URLs for Hugo's server.
    Use url_selection = 'transient', 'gaia', or 'extragal' in order to save on time.
    '''
    if url_selection == "all": url_plural = "s"
    else: url_plural = ""
    print("Looking for ", url_selection, " URL", url_plural, "...", sep="")

    extended_date = obs_date[:4] + "-" + obs_date[4:6] + "-" + obs_date[6:]

    if obs_date_to_datetime(obs_date) < obs_date_to_datetime("20240628"):
        transient_url  = "http://xmm-ssc.irap.omp.eu/claxson/BG_images/" + obs_date + "/" + extended_date + "_BlackGEM_transients.csv"
        gaia_url       = "http://xmm-ssc.irap.omp.eu/claxson/BG_images/" + obs_date + "/" + extended_date + "_BlackGEM_transients_gaia.csv"
        extragal_url   = "http://xmm-ssc.irap.omp.eu/claxson/BG_images/" + obs_date + "/" + extended_date + "_BlackGEM_transients_selected.csv"
        if (url_selection == 'all' or url_selection == 'transient'):
            if does_url_exist(transient_url)   != True: transient_url = "http://xmm-ssc.irap.omp.eu/claxson/BG_images/" + obs_date + "/" + extended_date + "_gw_BlackGEM_transients.csv"
        if (url_selection == 'all' or url_selection == 'gaia'):
            if does_url_exist(gaia_url)        != True: gaia_url      = "http://xmm-ssc.irap.omp.eu/claxson/BG_images/" + obs_date + "/" + extended_date + "_gw_BlackGEM_transients_gaia.csv"
        if (url_selection == 'all' or url_selection == 'extragal'):
            if does_url_exist(extragal_url)    != True: extragal_url  = "http://xmm-ssc.irap.omp.eu/claxson/BG_images/" + obs_date + "/" + extended_date + "_gw_BlackGEM_transients_selected.csv"

    else:
        transient_url  = "http://34.90.13.7/quick_selection/" + extended_date + "_BlackGEM_transients.csv"
        gaia_url       = "http://34.90.13.7/quick_selection/" + extended_date + "_BlackGEM_transients_gaia.csv"
        extragal_url   = "http://34.90.13.7/quick_selection/" + extended_date + "_BlackGEM_transients_selected.csv"

    if   url_selection == "all":
        print("Transient URL:    ", transient_url)
        print("Gaia URL:         ", gaia_url)
        print("Extragalactic URL:", extragal_url)
        return [transient_url, gaia_url, extragal_url]
    elif url_selection == "transient":   print("Transient URL:", transient_url); return transient_url
    elif url_selection == "gaia":        print("Gaia URL:", gaia_url); return gaia_url
    elif url_selection == "extragal":    print("Extragalactic URL:", extragal_url); return extragal_url



# def get_gaia_filename(obs_date):
#
#     extended_date = obs_date[:4] + "-" + obs_date[4:6] + "-" + obs_date[6:]
#
#     if obs_date_to_datetime(obs_date) < obs_date_to_datetime("20240628"):
#         base_gaia_url = "http://xmm-ssc.irap.omp.eu/claxson/BG_images/" + obs_date + "/" + extended_date + "_BlackGEM_transients_gaia.csv"
#         if does_url_exist(base_gaia_url) == True:
#             return base_gaia_url
#         else
#             return "http://xmm-ssc.irap.omp.eu/claxson/BG_images/" + obs_date + "/" + extended_date + "_gw_BlackGEM_transients_gaia.csv"
#     else:
#         return "http://34.90.13.7/quick_selection/" + extended_date + "_BlackGEM_transients_alerts_gaia.csv"


def get_recent_blackgem_transients(days_since_last_update):
    creds_user_file = str(Path.home()) + "/.bg_follow_user_john_creds"
    creds_db_file = str(Path.home()) + "/.bg_follow_transientsdb_creds"
    creds_user_file = "../../.bg_follow_user_john_creds"
    creds_db_file = "../../.bg_follow_transientsdb_creds"
    print(creds_user_file)
    print(creds_db_file)

    # Instantiate the BlackGEM object
    bg = BlackGEM(creds_user_file=creds_user_file, creds_db_file=creds_db_file)

    obs_date = date.today() - timedelta(1)
    extended_obs_date = obs_date.strftime("%Y-%m-%d")
    obs_date = obs_date.strftime("%Y%m%d")
    mjd = int(Time(extended_obs_date + "T00:00:00.00", scale='utc').mjd)

    ## Get previous history:
    previous_history = blackgem_recent_transients()

    if days_since_last_update > 30:
        days_since_last_update = 30

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
        # print("Bark!")
        transients_filename = get_transients_filenames(obs_date, url_selection="transient")
        # print("\n\ntransients_filename\n\n")
        # print(transients_filename)
        try:
            data = pd.read_csv(transients_filename)
            print("Transients read.")
        except Exception as e:
            ## If it doesn't exist, assume BlackGEM didn't observe any transients that night.
            print("No transients on ", obs_date, ":", sep="")
            print(e)
            continue

        d           = pd.Series([extended_date])
        date_column = d.repeat(len(data))
        date_column = date_column.set_axis(range(len(data)))

        data = data.sort_values(by=['runcat_id']).reset_index(drop=True)

        ## Find first time observed:

        qu = """\
            SELECT a.runcat
                  ,MIN(i."date-obs")
                  ,MAX(i."date-obs")
              FROM assoc a
                  ,extractedsource x
                  ,image i
             WHERE a.runcat IN %(runcatids)s
               AND a.xtrsrc = x.id
               AND x.image = i.id
             GROUP BY a.runcat;
        """

        params = {'runcatids' : tuple(list(data.runcat_id))}

        # params['runcatids'] = tuple(data['runcat_id']) + (data['runcat_id'].iloc[0],)
        query = qu % (params)

        l_results = bg.run_query(query)
        df_dates = pd.DataFrame(l_results, columns=['runcat_id','first_obs','last_obs'])
        df_dates = df_dates.sort_values(by=['runcat_id']).reset_index(drop=True)


        # print(len(data))
        # print(len(df_dates))
        #
        # print(data["runcat_id"].iloc[0:5])
        # print(df_dates["runcat_id"].iloc[0:5])

        if len(df_dates) > 0:

            df_dates['first_obs'] = df_dates['first_obs'].dt.strftime('%Y-%m-%d')
            df_dates['last_obs'] = df_dates['last_obs'].dt.strftime('%Y-%m-%d')

            data = pd.merge(data, df_dates, on="runcat_id", how="left")
        # data["last_obs"] = date_column


        # data_list.append(data.iloc[:20])
        data_list.append(data)
        num_sources += len(data)
        print(obs_date, "--", len(data), "\t Total:", num_sources)

    ## If there's any new data, combine it together
    if data_list:
        update_data = True
        df_new = pd.concat(data_list).reset_index(drop=True)
        # df_new = df_new.groupby('runcat_id', as_index=False).max()

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

    ## LEGACY - there are no duplicates in the list! Need to find a way to update the list then...
    # print("\n\nlen(df):")
    # print(len(df))
    # test = df[["runcat_id","ra"]]
    # test = test.astype('str')
    # print(len(test))
    df = df.sort_values('last_obs', ascending=False).drop_duplicates('runcat_id').sort_index()
    # test = test.drop_duplicates(subset=['runcat_id'], keep='first', inplace=False)
    # test.sort_values('runcat_id').reset_index()
    # print(test.iloc[0])
    # print(test.runcat_id[0])
    # print(type(test.runcat_id[0]))
    # print(df.iloc[0:5])
    # print(len(df))
    # print(test.iloc[50:100])
    # print("\n\n")
    df = df.dropna(subset=['first_obs', 'last_obs'])

    if update_data:
        print("Updating Recent History...")
        ## Make new columns and create new, truncated old columns
        ## Remove bugged values
        df['q_max'] = df['q_max'].replace(99,np.nan)
        df['u_max'] = df['u_max'].replace(99,np.nan)
        df['i_max'] = df['i_max'].replace(99,np.nan)

        # print(df.iloc[0])

        ## Round values for displaying
        df['ra_sml']        = round(df['ra'],4)
        df['dec_sml']       = round(df['dec'],4)
        df['snr_zogy_sml']  = round(df['snr'],1)
        # df['iauname_short'] = df['iauname'].str[5:]
        df['q_min_sml']     = round(df['q_min'],1)
        df['u_min_sml']     = round(df['u_min'],1)
        df['i_min_sml']     = round(df['i_min'],1)
        df['q_max_sml']     = round(df['q_max'],1)
        df['u_max_sml']     = round(df['u_max'],1)
        df['i_max_sml']     = round(df['i_max'],1)
        df['q_dif']         = round(df['q_max']-df['q_min'],2)
        df['u_dif']         = round(df['u_max']-df['u_min'],2)
        df['i_dif']         = round(df['i_max']-df['i_min'],2)

        # print("\n\n\nUPDATED\n\n\n")
        # print(df['snr_zogy_sml'])


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

def parse_ra_dec(ra, dec):

    radec = str(ra) + " " + str(dec)

    radec = re.sub(' +', ' ', radec)
    radec = radec.replace(",", "")

    radec_split = radec.split(" ")


    ## Attempt 1: Degrees?
    try:
        radec_split = radec.split(" ")
        ra  = radec_split[0]
        dec = radec_split[1]
        if len(radec_split) == 2:
            c = SkyCoord(ra=float(ra)*u.degree, dec=float(dec)*u.degree, frame='icrs')
            print("Success:", c.ra.degree, c.dec.degree)
        else:
            raise Exception("RA/Dec Degrees failed")

    ## Attempt 2: HMS
    except Exception as e:
        try:
            c = SkyCoord(radec, unit=(u.hourangle, u.deg))
            print("Success:", c.ra.degree, c.dec.degree)
        except:
            print("RA/Dec conversion failed!")

    ra      = c.ra.degree
    dec     = c.dec.degree

    return ra, dec


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
          ,(SELECT i0.id AS imageid
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
            WHERE a.runcat = '%(blackgem_id)s'
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

    t0 = time.time()

    bg = authenticate_blackgem()

    t1 = time.time()


    # Create an instance of the Transients Catalog
    tc = TransientsCatalog(bg)
    df_limiting_mag = get_limiting_magnitudes_from_BGEM_ID(transient_id)

    t2 = time.time()

    # Get all the associated extracted sources for this transient
    # Note that you can specify the columns yourself, but here we use the defaults
    bg_columns, bg_results = tc.get_associations(transient_id)
    df_bgem_lightcurve = pd.DataFrame(bg_results, columns=bg_columns)

    t3 = time.time()

    if len(df_bgem_lightcurve) == 0:
        print("No lightcurve found.")
        return df_bgem_lightcurve, df_limiting_mag

    # print(df_bgem_lightcurve.iloc[0])
    # print(df_limiting_mag.iloc[0])

    ## Remove all points in the limiting_mag lightcurve that have detections.
    # num_removed = 0
    # for i in range(len(df_limiting_mag["date_obs"])):
    #     if df_limiting_mag["date_obs"][i] in df_bgem_lightcurve['i."date-obs"'].unique():
    #         df_limiting_mag = df_limiting_mag.drop([i])
            # num_removed += 1
    # print(num_removed, "points removed.")
    # print(len(df_bgem_lightcurve), "points in lightcurve.")


    t4 = time.time()

    print("get_lightcurve_from_BGEM_ID Times:")
    print("t0->t1:", t1-t0)
    print("t1->t2:", t2-t1)
    print("t2->t3:", t3-t2)
    print("t3->t4:", t4-t3)

    return df_bgem_lightcurve, df_limiting_mag


def BGEM_to_GEMTOM_photometry_2(df_bgem_lightcurve, df_limiting_mag=[]):

    print("df_bgem_lightcurve:")
    print(df_bgem_lightcurve)
    print("df_limiting_mag:")
    print(df_limiting_mag)

    gemtom_photometry = pd.DataFrame({
        'mjd' : df_bgem_lightcurve["i.\"mjd-obs\""],
        'mag' : df_bgem_lightcurve["x.mag_zogy"],
        'magerr' : df_bgem_lightcurve["x.magerr_zogy"],
        'flux' : df_bgem_lightcurve["x.flux_zogy"],
        'fluxerr' : df_bgem_lightcurve["x.fluxerr_zogy"],
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
    # print("TNS API Key:", TNS_API_KEY)

    return response


def fetch_latest_TNS():
    search_url = "https://www.wis-tns.org/system/files/tns_public_objects/tns_public_objects.csv.zip"
    tns_marker = set_bot_tns_marker()
    headers = {'User-Agent': tns_marker}
    json_file = OrderedDict()
    search_data = {'api_key': TNS_API_KEY, 'data': json.dumps(json_file)}
    response = requests.post(search_url, headers = headers, data = search_data)
    # print("TNS API Key:", TNS_API_KEY)

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
    print(source[:11])
    if source[:11] == "The website":
        return False
    parsed = json.loads(source, object_pairs_hook = OrderedDict)
    result = json.dumps(parsed, indent = 4)
    return result

##  --- TNS functions for GEMTOM ---
def get_tns_from_ra_dec(ra, dec, radius):

    search_obj          = [
        ("ra", str(ra)),
        ("dec", str(dec)),
        ("radius", str(radius)),
        ("units", "arcsec"),
        ("objname", ""),
        ("objname_exact_match", 0),
        ("internal_name", ""),
        ("internal_name_exact_match", 0),
        ("objid", ""),
        ("public_timestamp", "")
    ]

    response = search(search_obj)
    try:
        json_data = format_to_json(response.text)
    except:
        return "Website Error"
    if json_data == False:
        return "Website Error"
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
        # print(json_data["data"])
        # print(len(json_data["data"]))
        return json_data


def search_TNS_ID(request):
    '''
    Redirects to a transient page based on a TNS ID
    '''
    tns_object_name = request.GET.get('user_input')
    print(tns_object_name)
    bgem_message, bgem_id, tns_name, tns_ra, tns_dec, tns_success = get_bgem_id_from_tns(tns_object_name)
    if tns_success:
        print("BGEM ID Found:", bgem_id)
        return redirect(f'/transients/{bgem_id}')
    else:
        print("BGEM ID Not Found")
    #     return redirect(f'/GEMTOM/transients/{user_input}')
    # return redirect('/GEMTOM/transients')  # Redirect to the original view if no input
        context = {
            "bgem_message" : bgem_message,
            "bgem_id" : bgem_id,
            "tns_name" : tns_name,
            "tns_object_name" : tns_object_name,
            "tns_ra" : tns_ra,
            "tns_dec" : tns_dec,
        }

        return render(request, "transient/index_noTNS.html", context)

# def TNS_to_GEMTOM(request, TNS_id):
#     '''
#     Adds a target to GEMTOM based on a TNS ID
#     '''
#     tns_object_name = TNS_id
#     print(tns_object_name)
#     bgem_message, bgem_id, tns_name, tns_ra, tns_dec, tns_success = get_bgem_id_from_tns(tns_object_name)
#     if tns_success:
#         print("BGEM ID Found:", bgem_id)
#         return redirect(f'/transients/{bgem_id}')
#     else:
#         print("BGEM ID Not Found")
#         context = {
#             "bgem_message" : bgem_message,
#             "bgem_id" : bgem_id,
#             "tns_name" : tns_name,
#             "tns_object_name" : tns_object_name,
#             "tns_ra" : tns_ra,
#             "tns_dec" : tns_dec,
#         }
#         return render(request, "transient/index_noTNS.html", context)

def get_bgem_id_from_tns(tns_object_name):
    tns_object_data = get_ra_dec_from_tns(tns_object_name)
    print(tns_object_data)
    tns_success     = False
    if tns_object_data == "Too many requests!":
        tns_object_data = "Too many TNS requests. Please check later."
        bgem_message    = tns_object_data
        bgem_id         = ""
        tns_name        = tns_object_name
        tns_ra          = ""
        tns_dec         = ""
    else:
        if tns_object_data["id_message"] == "Bad request":
            bgem_message    = "Bad Request; TNS Object does not exist. Have you checked the name?"
            bgem_id         = ""
            tns_name        = tns_object_name
            tns_ra          = ""
            tns_dec         = ""

        elif tns_object_data["id_code"] == 200:
            tns_ra      = tns_object_data["data"]["radeg"]
            tns_dec     = tns_object_data["data"]["decdeg"]
            tns_name    = tns_object_data["data"]['name_prefix'] + " " + tns_object_data["data"]['objname']

            df = transient_cone_search(tns_ra, tns_dec, radius=100)
            if len(df) == 0:
                print("TNS Target exists. No BlackGEM ID associated.")
                bgem_message = "No BlackGEM Transient ID found within 100 arcseconds. (Is the BlackGEM server up?)"
                bgem_id      = ""
            else:
                print(df)
                print(df.columns)
                df = df.sort_values(by=['distance_arcsec'])
                print("Closest BlackGEM target within 2 arcseconds:")
                print(df.iloc[0])

                if df.distance_arcsec[0] > 2:
                    bgem_message = "Closest BlackGEM transient (%.0f)"%df['id'][0] + " is %.1f arcseconds away."%df.distance_arcsec[0]
                    bgem_id      = df['id'][0]
                    print(bgem_message)
                else:
                    tns_success  = True
                    bgem_message = "Closest BlackGEM transient within 2 arcseconds: " + str(df['id'][0])
                    bgem_id      = df['id'][0]

        else:
            bgem_message = tns_object_data["id_message"]
            bgem_id         = ""
            tns_name        = tns_object_name
            tns_ra          = ""
            tns_dec         = ""


    return bgem_message, bgem_id, tns_name, tns_ra, tns_dec, tns_success




def get_ra_dec_from_tns(tns_object_name):
    get_obj             = [("objname", tns_object_name), ("objid", ""), ("photometry", ""), ("spectra", "")]
    response = get(get_obj)
    try:
        json_data = format_to_json(response.text)
    except:
        return "Website Error"
    json_data = json.loads(json_data)
    # print(json_data)
    if json_data["id_code"] == 429:
        return "Too many requests!"
    else:
        return json_data

## ----- TNS Functions -----
## =========================

def iau_name_from_bgem_id(bgem_id):

    ## Get the name, ra, and dec:
    bg = authenticate_blackgem()

    qu = """\
    SELECT id
          ,iau_name
      FROM runcat
     WHERE id = '%(bgem_id)s'
    """
    params = {'bgem_id': bgem_id}
    query = qu % (params)

    l_results = bg.run_query(query)

    return l_results[0][1]

@login_required
def BGEM_ID_View(request, bgem_id):
    '''
    Displays data of a certain transient
    '''
    print("Beginning BGEM ID View...")


    time_list = []
    time_list.append(time.time())

    ## Get all BlackGEM Target IDs

    df_targets = pd.DataFrame(list(Target.objects.all().values()))
    print(df_targets)
    # target_2 = Target.targetextra_set.all()

    # --- Now handle extras ---
    # Suppose you want keywords like "redshift", "priority", "magnitude"
    desired_extras = ['BlackGEM ID']

    # Query extras for those keywords
    extras = TargetExtra.objects.filter(key__in=desired_extras).values(
        'target_id', 'key', 'value'
    )

    extras_df = pd.DataFrame(list(extras))

    time_list.append(time.time())

    if str(bgem_id) in extras_df['value'].values:
        print("Transient is in GEMTOM!")
        target_num = extras_df["target_id"].values[np.where(extras_df['value'].values == str(bgem_id))]
        # print(target_num[0])
        # print(len(target_num))
        target_details = df_targets[df_targets['id'] == int(target_num[-1])]
        # print(target_details["name"])
        transient_in_GEMTOM = target_num[-1]
        GEMTOM_name = target_details["name"].iloc[0]
    else:
        print("Transient is not in GEMTOM...")
        transient_in_GEMTOM = False
        GEMTOM_name = False


    print("Fetching BGEM Lightcurve...")
    df_bgem_lightcurve, df_limiting_mag = get_lightcurve_from_BGEM_ID(bgem_id)
    # print(df_bgem_lightcurve)
    # print(df_bgem_lightcurve.columns)

    time_list.append(time.time())

    ## Get the name, ra, and dec:
    print("Authenticating with BlackGEM...")
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

        context = {"bgem_id" : bgem_id,}

        return render(request, "transient/index_2.html", context)

    iau_name    = source_data['iau_name'][0]
    ra          = source_data['ra_deg'][0]
    dec         = source_data['dec_deg'][0]

    gal_l, gal_b = ra_dec_to_galactic(ra, dec)

    time_list.append(time.time())

    response = "You're looking at BlackGEM transient %s."

    ## --- Location on Sky ---
    fig = plot_BGEM_location_on_sky(df_bgem_lightcurve, ra, dec)
    location_on_sky = plot(fig, output_type='div')

    # print(df_bgem_lightcurve['x.ra_psf_d'])
    # print(df_bgem_lightcurve['x.dec_psf_d'])

    time_list.append(time.time())

    ## --- Lightcurve ---
    fig = plot_BGEM_lightcurve(df_bgem_lightcurve, df_limiting_mag)
    lightcurve = plot(fig, output_type='div')


    # print(source_data)
    # print(l_results)

    time_list.append(time.time())


    ## Detail each observation:

    app = DjangoDash('EachObservation')


    df_bgem_lightcurve = df_bgem_lightcurve.rename(columns={
        'a.xtrsrc'          : "xtrsrc",
        'x.ra_psf_d'        : "ra_psf_d",
        'x.dec_psf_d'       : "dec_psf_d",
        'x.flux_zogy'       : "flux_zogy",
        'x.fluxerr_zogy'    : "fluxerr_zogy",
        'x.mag_zogy'        : "mag_zogy",
        'x.magerr_zogy'     : "magerr_zogy",
        'i."mjd-obs"'       : "mjd",
        'i."date-obs"'      : "date_obs",
        'i.filter'          : "filter",
    })

    # print("Standard Deviations:")
    # print(np.std(df_bgem_lightcurve['ra_psf_d']))
    # print(np.std(df_bgem_lightcurve['dec_psf_d']))

    # df_new = pd.concat([df_bgem_lightcurve,df_limiting_mag]).reset_index(drop=True)
    # print("\n")
    # print(df_bgem_lightcurve.iloc[0])
    # print(df_limiting_mag.iloc[0])
    # df_bgem_lightcurve = df_bgem_lightcurve.astype({'xtrsrc': 'int32'}).dtypes
    df_ful = pd.concat([df_bgem_lightcurve,df_limiting_mag]).reset_index(drop=True)
    df_ful = df_ful.sort_values(by=['mjd'])
    # print(df_new)
    # print("\n")

    # print(np.std(df_ful['ra_psf_d']))
    # print(np.std(df_ful['dec_psf_d']))

    # df_new.style.format({
    #     # 'runcat_id' : make_runcat_clickable,
    #     'xtrsrc' : make_xtrsrc_clickable
    # })

    df_new = df_bgem_lightcurve

    df_new['xtrsrc'] = df_new['xtrsrc'].apply(lambda x: f'[{int(x)}](https://staging.apps.blackgem.org/transients/blackview/show_xtrsrc/{int(x)})' if x > 0 else 'None')
    df_ful['xtrsrc'] = df_ful['xtrsrc'].apply(lambda x: f'[{int(x)}](https://staging.apps.blackgem.org/transients/blackview/show_xtrsrc/{int(x)})' if x > 0 else 'None')

    # print(df_new)

    getRowStyle = {
        "styleConditions": [
            {
                "condition": "params.data.limiting_mag > 0",
                "style": {"color": "lightgrey"},
            },
        ],
        "defaultStyle": {"color": "black"},
    }

    df_new = df_new[df_new['mag_zogy'] != 99]

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
                        {'headerName': 'xtrsrc', 'field':  'xtrsrc', 'cellRenderer': 'markdown', "linkTarget":"_blank"},
                        {'headerName': 'Filter', 'field': 'filter'},
                        {'headerName': 'Obs. Date', 'field':  'date_obs'},
                        {'headerName': 'Mag', 'field':  'mag_zogy',
                            "valueFormatter": {"function": "d3.format('.3f')(params.value)"}},
                        {'headerName': 'dMag', 'field':  'magerr_zogy', 'maxWidth': 80,
                            "valueFormatter": {"function": "d3.format('.3f')(params.value)"}},
                        {'headerName': 'RA', 'field':  'ra_psf_d', 'maxWidth': 120,
                            "valueFormatter": {"function": "d3.format('.5f')(params.value)"}},
                        {'headerName': 'Dec', 'field':  'dec_psf_d', 'maxWidth': 120,
                            "valueFormatter": {"function": "d3.format('.5f')(params.value)"}},
                        {'headerName': 'MJD', 'field':  'mjd', 'maxWidth': 120,
                            "valueFormatter": {"function": "d3.format('.5f')(params.value)"}},
            ],
            getRowStyle=getRowStyle,
            defaultColDef={
                'sortable': True,
                'filter': True,
                'resizable': True,
                'editable': False,
            },
            columnSize="autoSize",
            dashGridOptions={
                "skipHeaderOnAutoSize": True,
                "rowSelection": "single",
                "enableCellTextSelection": True,
            },
            style={'height': '350px', 'width': '100%'},  # Set explicit height for the grid
            className='ag-theme-balham'  # Add a theme for better appearance
        ),
        dcc.Store(id='selected-row-data'),  # Store to hold the selected row data
        html.Div(id='output-div'),  # Div to display the information
    ], style={'height': '350px', 'width': '100%'}
    )

    app2 = DjangoDash('FullObservation')
    ## Define the layout of the Dash app
    app2.layout = html.Div([
        dag.AgGrid(
            id='full-observation-grid',
            rowData=df_ful.to_dict('records'),
            # rowData=rowData_new,
            # columnDefs=[{'headerName': col, 'field': col} for col in df_new.columns],
            # columnDefs=[
            #             {'headerName': '1', 'field': 'x.ra_psf_d'},
            #             {'headerName': '2', 'field': 'x.dec_psf_d'},
            # ],
            columnDefs=[
                        {'headerName': 'a.xtrsrc', 'field':  'xtrsrc', 'cellRenderer': 'markdown', "linkTarget":"_blank"},
                        {'headerName': 'i.filter', 'field': 'filter'},
                        {'headerName': 'i.date_obs', 'field':  'date_obs'},
                        {'headerName': 'x.ra_psf_d', 'field':  'ra_psf_d',
                            "valueFormatter": {"function": "d3.format('.5f')(params.value)"}},
                        {'headerName': 'x.dec_psf_d', 'field':  'dec_psf_d',
                            "valueFormatter": {"function": "d3.format('.5f')(params.value)"}},
                        {'headerName': 'x.mag_zogy', 'field':  'mag_zogy',
                            "valueFormatter": {"function": "d3.format('.3f')(params.value)"}},
                        {'headerName': 'x.magerr_zogy', 'field':  'magerr_zogy',
                            "valueFormatter": {"function": "d3.format('.3f')(params.value)"}},
                        {'headerName': 'Limiting Mag', 'field':  'limiting_mag',
                            "valueFormatter": {"function": "d3.format('.3f')(params.value)"}},
                        {'headerName': 'i.mjd_obs', 'field':  'mjd',
                            "valueFormatter": {"function": "d3.format('.5f')(params.value)"}},
            ],
            getRowStyle=getRowStyle,
            defaultColDef={
                'sortable': True,
                'filter': True,
                'resizable': True,
                'editable': False,
            },
            columnSize="autoSize",
            dashGridOptions={
                "skipHeaderOnAutoSize": True,
                "rowSelection": "single",
                "enableCellTextSelection": True,
            },
            style={'height': '300px', 'width': '100%'},  # Set explicit height for the grid
            className='ag-theme-balham'  # Add a theme for better appearance
        ),
        dcc.Store(id='selected-row-data'),  # Store to hold the selected row data
        html.Div(id='output-div'),  # Div to display the information
    ], style={'height': '300px', 'width': '100%'}
    )

    time_list.append(time.time())


    ## TNS:
    print("Starting TNS Query...")

    dec_str = str(dec)
    if dec_str[0] != '-':
        dec_str = '+' + dec_str

    context = {
        "bgem_id"               : bgem_id,
        "iau_name"              : iau_name,
        "ra"                    : ra,
        "dec"                   : dec_str,
        "gal_l"                 : gal_l,
        "gal_b"                 : gal_b,
        "dataframe"             : df_bgem_lightcurve,
        "columns"               : df_bgem_lightcurve.columns,
        "location_on_sky"       : location_on_sky,
        "lightcurve"            : lightcurve,
        "transient_in_GEMTOM"   : transient_in_GEMTOM,
        "GEMTOM_name"           : GEMTOM_name,
        # "tns_flag"          : tns_flag,
        # "tns_flag_prefix"   : tns_flag_prefix,
        # "tns_flag_name"     : tns_flag_name,
        # "tns_flag_sep"      : tns_flag_sep,
        # "tns_flag_bgem"     : tns_flag_bgem,
        # "tns_data"          : tns_data,
        # "tns_text"          : tns_text,
        # "tns_list"          : tns_list,
        # "image_name"        : file_name
        # "tns_nearby"        : tns_nearby,
        # "tns_objects_data"  : tns_objects_data,
    }

    # print(context["image_name"])

    time_list.append(time.time())

    print("Transients ID View Times:")
    for i in range(len(time_list)-1):
        print("t"+str(i)+"->t"+str(i+1)+": "+str(time_list[i+1]-time_list[i]))


    return render(request, "transient/index.html", context)

def delayed_search_for_Vizier(request):

    print("Delayed Search for Vizier called!")

    ra = request.GET.get('ra')
    dec = request.GET.get('dec')
    ra = float(ra)
    dec = float(dec)

    # Define search coordinates (e.g., center of interest)
    target_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

    # Define search radius
    search_radius = 1.5 * u.arcsec

    result = Vizier().query_region(target_coord, radius=search_radius)

    if len(result)-1 == 1:  message = "Target is in " + str(len(result)-1) + " VizieR catalog<br>"
    else:                   message = "Target is in " + str(len(result)-1) + " VizieR catalogs<br>"

    ## Find specific catalogs

    GSC = False
    AAVSO = False
    for i in range(len(result)):
        # print(result[i]._meta["name"])
        # print(result[i]._meta)
        if result[i]._meta["name"] == "I/353/gsc242":
            GSC = True
        if result[i]._meta["name"] == "B/vsx/vsx":
            AAVSO = True


    if len(result)-1 > 0:

        if GSC: message += "<a style='color:mediumaquamarine'><em>GSC</em></a>"
        else: message += "<a style='color:lightgrey'><em>GSC</em></a>"

        message += "<a style='color:lightgrey'><em> • </em></a>"

        if AAVSO: message += "<a style='color:mediumaquamarine'><em>AAVSO</em></a>"
        else: message += "<a style='color:lightgrey'><em>AAVSO</em></a>"

        message += '<br><a style="margin:5px" class="btn btn-outline-primary" href="https://vizier.cds.unistra.fr/viz-bin/VizieR-4?-c=' + str(ra) + '%20' + str(dec) + '&-c.u=arcsec&-c.r=1.5&-c.eq=J2000&-c.geom=r&-out.max=50&-out.add=_r" target="_blank">Query VizieR</a><br>'

    print("Returning Vizier response.")
    return JsonResponse({'message': message})


def delayed_search_for_TNS(request):

    print("Delayed Search for TNS called!")

    # print(request)
    # print(request.GET)
    ra = request.GET.get('ra')
    dec = request.GET.get('dec')
    bgem_id = request.GET.get('bgemid')

    ## TNS:
    print("Starting TNS Query...")
    search_radius = 5
    tns_data = get_tns_from_ra_dec(ra, dec, search_radius)
    # print("\n\nTNS data:\n")
    # print(tns_data)
    # print("\n\n")

    if tns_data == "Too many requests!":
        message = '<div style="color:black"><em>Too many TNS requests. Please check later.</em></div>'
        return JsonResponse({'message': message})
    elif tns_data == "Unauthorised!":
        message = '<div style="color:black"><em>Note: TNS Unauthorised. Please check.</em></div>'
        return JsonResponse({'message': message})
    elif tns_data == "Website Error":
        message = '<div style="color:black"><em>TNS site encountered an error. Please try again later.</em></div>'
        return JsonResponse({'message': message})
    else:
        tns_reply = tns_data["data"]
        tns_reply_length = len(tns_data["data"])

    if tns_reply_length == 0:
        message = '<div style="color:lightgrey"><em>No TNS Object within 10 arcseconds.</em></div>'

    else:

        ## If multiple objects, find which is closest
        if tns_reply_length == 1:
            tns_index = 0
        if tns_reply_length > 1:
            tns_sep = []
            for i in range(tns_reply_length):
                tns_object_data     = get_ra_dec_from_tns(tns_reply[i]["objname"])
                bgem_object_radec   = SkyCoord(float(ra)*u.deg, float(dec)*u.deg, frame='icrs')
                this_object_ra      = tns_object_data["data"]["radeg"]
                this_object_dec     = tns_object_data["data"]["decdeg"]
                this_object_radec   = SkyCoord(this_object_ra*u.deg, this_object_dec*u.deg, frame='icrs')
                tns_sep.append(bgem_object_radec.separation(this_object_radec).arcsecond)
            tns_index = tns_sep.index(min(tns_sep))


        tns_name = str(tns_reply[0]["prefix"] + " " + tns_reply[tns_index]["objname"])

        ## Find Separation
        print("Getting object data...")
        tns_object_data     = get_ra_dec_from_tns(tns_reply[tns_index]["objname"])
        bgem_object_radec   = SkyCoord(float(ra)*u.deg, float(dec)*u.deg, frame='icrs')
        this_object_ra      = tns_object_data["data"]["radeg"]
        this_object_dec     = tns_object_data["data"]["decdeg"]
        this_object_radec   = SkyCoord(this_object_ra*u.deg, this_object_dec*u.deg, frame='icrs')
        this_object_sep     = bgem_object_radec.separation(this_object_radec).arcsecond
        # print(this_object_sep)

        message = 'Closeby TNS! • '
        message += '<b><a href=https://www.wis-tns.org/object/' + tns_reply[tns_index]["objname"] + ' target="_blank">' + tns_reply[tns_index]["prefix"] + ' ' + tns_reply[tns_index]["objname"] + '</a></b> • '
        message += '<a style="color:grey"><em>(%.1f")</em></a><br>'%this_object_sep


        # message += '<form method="post" action="name_to_GEMTOM/' + bgem_id + '/' + tns_name + \
            # '/" class="image-form"><button type="submit" class="btn btn-outline-success">Add to GEMTOM with TNS</button></form><br>'

        if "BGEM" in tns_object_data["data"]["internal_names"]:
            # tns_flag_bgem = True
            message += '<a style="color:mediumaquamarine"><em>BlackGEM data reported to TNS</em></a><br>'
        else:
            message += '<a style="color:darkorange"><em>BlackGEM data not in TNS!</em></a><br>'
        # print("tns_flag_bgem:", tns_flag_bgem)

        message += '<a style="margin:5px" class="btn btn-outline-success" href="https://gemtom.blackgem.org/name_to_GEMTOM/' + bgem_id + '/' + tns_name + \
            '/"  target="_blank">Add to GEMTOM with TNS</a><br>'

    print("Returning TNS response.")
    return JsonResponse({'message': message})


## =============================================================================
## ------------------ Codes for the Unified Transients page --------------------

def get_transient_image(bgem_id, ra, dec, df_bgem_lightcurve = False, tns_ra=False, tns_dec=False):

    t0 = time.time()

    # Instantialte the BlackGEM object, with a connection to the database
    bg = authenticate_blackgem()

    t1 = time.time()

    desimoc = MOC.from_fits('./data/MOC_DESI-Legacy-Surveys_DR10.fits')
    coords = SkyCoord(ra,dec,unit='deg',frame='icrs')
    indesi = desimoc.contains_lonlat(coords.ra,coords.dec)

    t2 = time.time()

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

    t3 = time.time()

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


    t4 = time.time()

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

    t5 = time.time()

    print("URL =", url)
    cutout = requests.get(url, stream = True)

    t6 = time.time()

    if cutout.status_code == 200:
        with open(file_name,'wb') as f:
            shutil.copyfileobj(cutout.raw, f)
        print('Image successfully downloaded: ',file_name)

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


    t7 = time.time()

    print("Get Transient Image Times:")
    print("t0->t1:", t1-t0)
    print("t1->t2:", t2-t1)
    print("t2->t3:", t3-t2)
    print("t3->t4:", t4-t3)
    print("t4->t5:", t5-t4)
    print("t5->t6:", t6-t5)
    print("t6->t7:", t7-t6)

    return file_name


def search_fuzzy_iauname(request):
    '''
    Redirects to a transient page based on an IAU name. Search is fuzzy, and can return several options.
    '''
    bg = authenticate_blackgem()

    iauname = request.GET.get('user_input')

    iauname_edited = iauname
    iauname_edited = iauname_edited.replace(" -", "-")
    iauname_edited = iauname_edited.replace(" +", "+")

    iauname = iauname.replace("-", " -")
    iauname = iauname.replace("+", " +")
    iauname = iauname.replace("  ", " ")
    iauname = iauname.replace("BGEMJ", "BGEM J")
    iauname_split = re.split('_| ', iauname)
    if len(iauname_split) == 1:
        context = {"iauname" : iauname_edited}
        return render(request, "transient/index_noIAU.html", context)

    if iauname_split[0][0] == "B": iauname_split = iauname_split[1:]

    if len(iauname_split) == 1:
        context = {"iauname" : iauname_edited}
        return render(request, "transient/index_noIAU.html", context)

    if (len(iauname_split[0]) < 4) or (len(iauname_split[1]) < 4):
        context = {"iauname" : iauname_edited}
        return render(request, "transient/index_noIAU.html", context)

    qu = """\
    SELECT id
          ,iau_name
          ,ra_deg
          ,dec_deg
          ,datapoints
      FROM runcat

    """
    qu = qu + "WHERE iau_name LIKE '%" + iauname_split[0] + "%'\n"
    qu = qu + "  AND iau_name LIKE '%" + iauname_split[1] + "%'\n"

    # query = qu % (params)

    # Call the run_query function that can execute any sql query
    l_results = bg.run_query(qu)

    # print(l_results)

    df = pd.DataFrame(l_results, columns=['id', 'iau_name', 'ra', 'dec', 'datapoints'])


    if len(df) == 1:
        return redirect(f'/transients/{df.id[0]}')

    elif len(df) > 1:
        df_lists = zip(list(df.id),
                    list(df.iau_name),
                    list(df.ra),
                    list(df.dec),
                    list(df.datapoints)
        )

        context = {
            "iauname" : iauname_edited,
            "df" : df_lists
        }

        return render(request, "transient/index_multiIAU.html", context)

    else:
        context = {"iauname" : iauname_edited}
        return render(request, "transient/index_noIAU.html", context)


    return df


def search_GAIA_ID(request):
    '''
    Redirects to a transient page based on a Gaia ID, or if there are multiple, returns several options.
    '''
    bg = authenticate_blackgem()

    gaia_id = request.GET.get('user_input')

    if gaia_id[:4] != "Gaia": gaia_id = "Gaia DR3 " + gaia_id

    ## Get RA/Dec from GAIA Designation
    job = Gaia.launch_job("select top 10 "
                      "designation,ra,dec "
                      "from gaiadr3.gaia_source "
                      "where designation = '" + gaia_id + "' "
                      "order by source_id ")

    if len(job.get_results()) == 0:
        context = {
                "gaia_id" : gaia_id,
                "success" : False
                }
        return render(request, "transient/index_noGaia.html", context)


    ra = job.get_results()["ra"].data.data[0]
    dec = job.get_results()["dec"].data.data[0]

    df = transient_cone_search(ra, dec, 10)

    if len(df) == 0:
        context = {
                "gaia_id" : gaia_id,
                "ra" : ra,
                "dec" : dec,
                "success" : True
                }
        return render(request, "transient/index_noGaia.html", context)
    elif len(df) == 1:
        return redirect(f'/transients/{df.id[0]}')
    elif len(df) > 1:
        print(df.columns)
        df_lists = zip(
                    list(df.id),
                    list(df.ra_deg),
                    list(df.dec_deg),
                    list(df.datapoints),
                    list(df.distance_arcsec),
        )

        context = {
            "gaia_id" : gaia_id,
            "ra" : ra,
            "dec" : dec,
            "success" : True,
            "df" : df_lists
        }

        return render(request, "transient/index_multiGaia.html", context)


def ra_dec_checker(ra, dec):

    success = True
    message = []

    try:
        ra      = float(ra)
    except:
        success = False
        message.append("RA cannot be interpreted as a float.")

    try:
        dec     = float(dec)
    except:
        success = False
        message.append("Dec cannot be interpreted as a float.")

    if success:
        if   (ra < 0) or (ra > 360):   success = False;    message.append("RA outside of 0 < RA < 360 degrees.")
        if (dec < -90) or (dec > 90):  success = False;    message.append("Dec outside of -90 < Dec < 90 degrees.")

    return success, message


def search_BGEM_RA_Dec(request):
    '''
    Redirects to a transient page based on a BlackGEM RA/Dec, or if there are multiple, returns several options.
    '''
    bg = authenticate_blackgem()

    ra      = request.GET.get('ra')
    dec     = request.GET.get('dec')
    radius  = request.GET.get('radius')

    ra, dec = parse_ra_dec(ra, dec)

    success, message = ra_dec_checker(ra, dec)

    try:
        radius  = float(radius)
        if (radius < 0) or (radius > 1000):  success = False;    message.append("Radius outside of 0 < radius < 1000 arcseconds.")
    except:
        success = False
        message.append("Radius cannot be interpreted as a float.")



    if not success:
        context = {
                "ra" : ra,
                "dec" : dec,
                "radius" : radius,
                "message" : message,
                "success" : False
                }
        return render(request, "transient/index_noRadec.html", context)

    ## Find the last time this RA/Dec was searched
    skytile_message = search_skytiles_from_RA_Dec(ra,dec)

    df = transient_cone_search(ra, dec, radius)

    if len(df) == 0:
        context = {
                "ra" : ra,
                "dec" : dec,
                "radius" : radius,
                "skytile_message" : skytile_message,
                "success" : True
                }
        return render(request, "transient/index_noRadec.html", context)
    elif len(df) == 1:
        return redirect(f'/transients/{df.id[0]}')
    elif len(df) > 1:
        print(df.columns)
        df_lists = zip(
                    list(df.id),
                    list(df.ra_deg),
                    list(df.dec_deg),
                    list(df.datapoints),
                    list(df.distance_arcsec),
        )

        context = {
            "ra" : ra,
            "dec" : dec,
            "radius" : radius,
            "skytile_message" : skytile_message,
            "success" : True,
            "df" : df_lists
        }

        return render(request, "transient/index_multiRadec.html", context)

    return df


def search_skytiles_from_RA_Dec(ra,dec):
    '''
    From an RA and Dec, find any fields it exists in.
    '''

    bg = authenticate_blackgem()
    tc = TransientsCatalog(bg)

    l_columns, l_results = tc.nearest_skytiles(ra, dec, 1.35*1.41)
    df_nearest_fields = pd.DataFrame(l_results, columns=l_columns)

    message = ["- - -"]

    if len(df_nearest_fields) == 0:
        message.append("RA/Dec is not in any BlackGEM fields.")
        return message

    df_nearest_fields["status"] = "Green"
    df_nearest_fields["status"][df_nearest_fields["distance_deg"] > 1.35] = "Yellow"

    last_observations = [tc.skytile_observing_times(i) for i in df_nearest_fields['field_id']]
    df_nearest_fields["num"] = [len(i[1]) for i in last_observations]
    df_nearest_fields["num"] = [len(i[1]) for i in last_observations]

    df_observations = pd.DataFrame(last_observations[0][1], columns = list(last_observations[0][0]))
    df_observations["field_id"] = df_nearest_fields["field_id"].iloc[0]
    df_observations["status"]   = df_nearest_fields["status"].iloc[0]
    df_observations["num"]      = df_nearest_fields["num"].iloc[0]

    for i in range(1, len(last_observations)):
        df_new_observations = pd.DataFrame(last_observations[i][1], columns = list(last_observations[i][0]))
        df_new_observations["field_id"] = df_nearest_fields["field_id"].iloc[i]
        df_new_observations["status"]   = df_nearest_fields["status"].iloc[i]
        df_new_observations["num"]      = df_nearest_fields["num"].iloc[i]
        df_observations = pd.concat([df_observations, df_new_observations]).reset_index(drop=True)

    df_observations = df_observations.sort_values(by=['"mjd-obs"'], ascending=False).reset_index(drop=True)

    if len(df_observations) == 0:
        message.append("RA/Dec is in the following BlackGEM fields:")
        message.append(list(df_nearest_fields["field_id"]))
        message.append("but it has not yet been observed.")
        return message

    if df_observations['status'].iloc[0] == "Green":# or df_observations['status'].iloc[0] == "Yellow":
        message.append("RA/Dec was last observed on " + str(df_observations['"date-obs"'].iloc[0])[:-7])
        days_ago = (datetime.today()-df_observations['"date-obs"'].iloc[0]).days + 1
        if days_ago == 1: message.append("(Last night!)")
        else: message.append("(That was " + str(days_ago) + " nights ago)")

    else:

        message.append("RA/Dec may have been observed on " + str(df_observations['"date-obs"'].iloc[0])[:-7] + "; it may be outside the field.")

        days_ago = (datetime.today()-df_observations['"date-obs"'].iloc[0]).days + 1
        if days_ago == 1: message.append("(Last night!)")
        else: message.append("(That was " + str(days_ago) + " nights ago)")

        message.append("- - -")

        # df_confirmed_observations = df_observations[df_observations["status"] == "Green" or df_observations["status"] == "Yellow"].reset_index(drop=True)
        df_confirmed_observations = df_observations[df_observations["status"] == "Green"].reset_index(drop=True)
        # df_confirmed_observations= df_observations.loc[df_observations["status"].isin(["Green","Yellow"])].reset_index(drop=True)
        if len(df_confirmed_observations) > 0:
            message.append("RA/Dec was last definitely observed on " + str(df_confirmed_observations['"date-obs"'].iloc[0])[:-7])
            days_ago = (datetime.today()-df_confirmed_observations['"date-obs"'].iloc[0]).days + 1

            if days_ago == 1: message.append("(That was last night!)")
            else: message.append("(That was " + str(days_ago) + " nights ago)")

        else:
            message.append("RA/Dec has not been definitely observed.")


    return message


def search_skytiles_from_RA_Dec_orig(request):
    '''
    From an RA and Dec, find any fields it exists in.
    '''
    bg = authenticate_blackgem()

    tc = TransientsCatalog(bg)

    ra      = request.GET.get('ra')
    dec     = request.GET.get('dec')

    ra, dec = parse_ra_dec(ra, dec)

    ## Check RA/Dec:
    success, message = ra_dec_checker(ra, dec)

    if not success:
        context = {
                "ra" : ra,
                "dec" : dec,
                "message" : message,
                "success" : False
        }
        return render(request, "transient/index_Skytile.html", context)


    skytile_message = search_skytiles_from_RA_Dec(ra,dec)

    df = transient_cone_search(ra, dec, 60)

    if len(df) == 0:
        context = {
                "ra" : ra,
                "dec" : dec,
                "message" : skytile_message,
                "success" : True,
                "sources" : False,
                }

    else:
        df_lists = zip(
                    list(df.id),
                    list(df.ra_deg),
                    list(df.dec_deg),
                    list(df.datapoints),
                    list(df.distance_arcsec),
        )

        context = {
            "ra" : ra,
            "dec" : dec,
            "message" : skytile_message,
            "success" : True,
            "sources" : True,
            "df" : df_lists,
        }

    return render(request, "transient/index_Skytile.html", context)

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
    days_since_last_update = difference.days + 1
    print("Days since last update:", days_since_last_update)

    yesterday_date_string = yesterday_date.strftime("%Y%m%d")

    if days_since_last_update > 0:
        get_recent_blackgem_transients(days_since_last_update)
        return blackgem_recent_transients()
    else:
        return recent_transients


class UnifiedTransientsView(LoginRequiredMixin, TemplateView):
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




    # app = DjangoDash('ConeSearchTable')
    #
    # style_dict = {
    #   "font-size": "16px",
    #   "margin-right": "10px",
    #   'width': '200px',
    #   'height': '30px',
    #   # 'scale':'2'
    # }
    #
    # button_style_dict = {
    #     "font-size": "16px",
    #     "margin-right": "10px",
    #     "padding": "7px 24px",
    #     # 'border-radius': '5px',
    #     'color':'#027bff',
    #     'background-color': '#ffffff',
    #     'border': '2px solid #027bff',
    #     'border-radius': '5px',
    #     'cursor': 'pointer',
    # }
    #
    # app.layout = html.Div([
    #     html.Div([
    #         # html.A("Link to external site", href='https://plot.ly', target="_blank"),
    #         dcc.Input(id='ra-input',        type='number', min=0, max=360,  placeholder=' RA (deg)',        style=style_dict),
    #         dcc.Input(id='dec-input',       type='number', min=-90, max=90,   placeholder=' Dec (deg)',       style=style_dict),
    #         dcc.Input(id='radius-input',    type='number', min=0, max=600,  placeholder=' Radius (arcseconds)',    style=style_dict),
    #         html.Button('Search', id='submit-button', n_clicks=0, style=button_style_dict),
    #         # html.Button('Search', id='submit-button', n_clicks=0, style={"font-size": "16px","margin-right": "10px",})
    #     ], style={'margin-bottom': '20px', "text-align":"center"}),
    #     html.Div(id='results-container', children=[]),
    #     html.Div(id='redirect-trigger', style={'display': 'none'})
    # ])
    #
    # # Define the callback to update the table based on input coordinates
    # @app.callback(
    #     Output('results-container', 'children'),
    #     Input('submit-button', 'n_clicks'),
    #     State('ra-input', 'value'),
    #     State('dec-input', 'value'),
    #     State('radius-input', 'value'),
    #     prevent_initial_call=False
    # )
    # def update_results(n_clicks, ra, dec, radius):
    #     print(ra, dec, radius)
    #
    #     # tns_object_name = "SN2024zmm"
    #     # tns_message, tns_bgem_id = get_bgem_id_from_tns(tns_object_name)
    #     # print(tns_message)
    #     # print(tns_bgem_id)
    #
    #     if ra is not None and dec is not None:
    #         df = transient_cone_search(ra, dec, radius)
    #         if len(df) == 0:
    #             return html.Div([
    #                         html.P("RA: " +str(ra)),
    #                         html.P("Dec: " +str(dec)),
    #                         html.P("Radius: " +str(radius)),
    #                         html.Em("No targets found"),
    #                     ], style={"text-align":"center", "font-size": "18px", "font-family":"sans-serif"}
    #                 )
    #         else:
    #             df['id'] = df['id'].apply(lambda x: f'[{x}](/transients/{x})')
    #             # df['id'] = df['id'].apply(lambda x: f'<a href="/transients/{x}" target="_blank">{x}</a>)')
    #             message = html.Div([html.P(html.Em("Ctrl/Cmd-click on links to open the transient in a new tab"))], style={"text-align":"center", "font-size": "18px", "font-family":"sans-serif"})
    #             table = dag.AgGrid(
    #                 id='results-table',
    #                 columnDefs=[
    #                     {'headerName': 'ID', 'field': 'id', 'cellRenderer': 'markdown'},
    #                     # {'headerName': 'ID', 'field': 'id', 'cellRenderer': 'htmlCellRenderer'},
    #                     # {'headerName': 'ID', 'field': 'id'},
    #                     {'headerName': 'Datapoints', 'field': 'datapoints'},
    #                     {'headerName': 'RA', 'field': 'ra_deg'},
    #                     {'headerName': 'Dec', 'field': 'dec_deg'},
    #                     {'headerName': 'Dist (")', 'field': 'distance_arcsec'},
    #                 ],
    #                 rowData=df.to_dict('records'),
    #                 dashGridOptions={"rowSelection": "single"},
    #                 style={'height': '200px', 'width': '100%'},
    #                 className='ag-theme-balham'  # Add a theme for better appearance
    #             )
    #             return message, table
    #     else:
    #         message = html.Div([html.P(html.Em("Enter co-ordinates to search."))], style={"text-align":"center", "font-size": "18px", "font-family":"sans-serif", "color":"grey"})
    #
    #         return message

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

    # if hidden_error == "Yes":
    #     print("I'm an error!")

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
                # {'headerName': 'IAU Name', 'field': 'iauname_short'},
                {'headerName': 'RA', 'field': 'ra_sml'},
                {'headerName': 'Dec', 'field': 'dec_sml'},
                {'headerName': 'Datapoints', 'field': 'n_datapoints',   'minWidth': 105, 'maxWidth': 105},
                {'headerName': 'S/N', 'field': 'snr_zogy_sml'},
                {'headerName': 'q min', 'field': 'q_min_sml',   'minWidth': 75, 'maxWidth': 75},
                {'headerName': 'q dif', 'field': 'q_dif',       'minWidth': 75, 'maxWidth': 75},
                {'headerName': 'u min', 'field': 'u_min_sml',   'minWidth': 75, 'maxWidth': 75},
                {'headerName': 'u dif', 'field': 'u_dif',       'minWidth': 75, 'maxWidth': 75},
                {'headerName': 'i min', 'field': 'i_min_sml',   'minWidth': 75, 'maxWidth': 75},
                {'headerName': 'i dif', 'field': 'i_dif',       'minWidth': 75, 'maxWidth': 75},
                {'headerName': 'First Obs', 'field': 'first_obs', 'maxWidth': 110},
                {'headerName': 'Last Obs',  'field': 'last_obs',  'maxWidth': 110},
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
                # html.P(str(row_data["iauname"]), style={'font-size':'17px'}),
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

            if name == None or name == "None":
                all_stats = all_stats_from_bgem_id(id)
                name = all_stats[0]

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
                columns=[{'name': k, 'id': k} for k in row_data.keys() if k in ['q_min', 'q_max', 'q_xtrsrc', 'q_rb', 'q_fwhm', 'q_dif']],#[i] for i in [0,1,2]],
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
                columns=[{'name': k, 'id': k} for k in row_data.keys() if k in ['q_dif', 'u_dif', 'i_dif']],
                style_table={'margin': 'auto'}, style_cell={'textAlign': 'center', 'padding': '5px'}, style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
            # table_5 = dash_table.DataTable(
            #     data=[row_data],
            #     columns=[{'name': k, 'id': k} for k in row_data.keys()][23:29],
            #     style_table={'margin': 'auto'}, style_cell={'textAlign': 'center', 'padding': '5px'}, style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            # )
            # ## Extra 2
            # table_6 = dash_table.DataTable(
            #     data=[row_data],
            #     columns=[{'name': k, 'id': k} for k in row_data.keys()][30:36],
            #     style_table={'margin': 'auto'}, style_cell={'textAlign': 'center', 'padding': '5px'}, style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            # )

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
                [table_1] + [table_2] + [table_3] + [table_4] + [table_5],# + [table_6],
                style={'font-family': 'Arial', 'text-align': 'center'})

        return



## =============================================================================
## ---------------------- Code for the Discoveries Page ------------------------

class DiscoveriesView(TemplateView):
    template_name = 'discoveries.html'

    discoveries_filename = "./data/blackgem_tns_discoveries.csv"

    try:
        df_blackgem_table = pd.read_csv(discoveries_filename)

        df_blackgem_table['name'] = df_blackgem_table['name'].apply(lambda x: f'[{str(x)}](https://www.wis-tns.org/object/{str(x)})')
        df_blackgem_table['id'] = df_blackgem_table['id'].apply(lambda x: f'[{str(x)}](https://gemtom.blackgem.org/transients/{str(x)})')
        df_blackgem_table['discoveryday'] = [str(x).split(" ")[0] for x in df_blackgem_table['discoverydate']]
        df_blackgem_table['Discovery_ADS_bibcode'] = df_blackgem_table['Discovery_ADS_bibcode'].apply(lambda x: f'[ADS](https://ui.adsabs.harvard.edu/abs/{str(x)}/abstract)')

        df_blackgem_table['last_obs'] = Time(df_blackgem_table['last_obs'], format='mjd').iso
        df_blackgem_table['last_obs'] = [x.split(" ")[0] for x in df_blackgem_table['last_obs']]

    except Exception as e:
        print("Error! Discoveries table not read.")
        print(e)

        df_blackgem_table = pd.DataFrame(data={
            'name_prefix':[],
            'name':[],
            'ra':[],
            'declination':[],
            'discoveryday':[],
            'reporters':[],
            'Discovery_ADS_bibcode':[],
            'id':[],
            'iau_name':[],
        })


    app = DjangoDash("discoveries_table")

    getRowStyle = {
        "styleConditions": [
            {
                "condition": "params.data.limiting_mag > 0",
                "style": {"color": "lightgrey"},
            },
        ],
        "defaultStyle": {"color": "black"},
    }

    ## Define the layout of the Dash app
    app.layout = html.Div([
        dag.AgGrid(
            id='observation-grid',
            rowData=df_blackgem_table.to_dict('records'),
            columnDefs=[
                        {'headerName': 'Pre', 'field':  'name_prefix', 'maxWidth' : 110},
                        {'headerName': 'Name', 'field': 'name', 'cellRenderer': 'markdown', "linkTarget":"_blank"},
                        {'headerName': 'Runcat ID', 'field': 'id', 'cellRenderer': 'markdown', "linkTarget":"_blank"},
                        # {'headerName': 'IAU Name', 'field': 'iau_name', 'cellRenderer': 'markdown', "linkTarget":"_blank"},
                        {'headerName': 'RA', 'field': 'ra',
                            "valueFormatter": {"function": "d3.format('.2f')(params.value)"}},
                        {'headerName': 'Dec', 'field': 'declination',
                            "valueFormatter": {"function": "d3.format('.2f')(params.value)"}},
                        {'headerName': 'Latest Mag', 'field': 'latest_mag',
                            "valueFormatter": {"function": "d3.format('.1f')(params.value)"}},
                        {'headerName': 'Latest Obs', 'field': 'last_obs'},
                        {'headerName': 'Discovery', 'field': 'discoveryday'},
                        {'headerName': 'Reporters', 'field': 'reporters'},#, 'suppressSizeToFit': True},
                        {'headerName': 'ADS', 'field': 'Discovery_ADS_bibcode', 'cellRenderer': 'markdown', "linkTarget":"_blank"},
            ],
            getRowStyle=getRowStyle,
            defaultColDef={
                'sortable': True,
                'filter': True,
                'resizable': True,
                'editable': False,
            },
            # columnSize="sizeToFit",
            columnSize="autoSize",
            dashGridOptions={
                "skipHeaderOnAutoSize": True,
                "rowSelection": "single",
                "enableCellTextSelection": True,
            },
            style={'height': '500px', 'width': '100%'},  # Set explicit height for the grid
            # className='ag-theme-balham'  # Add a theme for better appearance
            className='ag-theme-alpine'  # Add a theme for better appearance
        ),
        dcc.Store(id='selected-row-data'),  # Store to hold the selected row data
        html.Div(id='output-div'),  # Div to display the information
    ], style={'height': '500px', 'width': '100%'}
    )




    # 'number_of_discoveries': len(df_blackgem_table)

    # print(context)
    def get_context_data(self, **kwargs):

        context = super().get_context_data(**kwargs)

        tns_latest_filepath = "./data/tns_latest.csv"
        df_bgem_contrib_filepath = "./data/bgem_tns_contributions.csv"
        df_bgem_discoveries_filepath = "./data/bgem_tns_discoveries.csv"
        df_bgem_sn_discoveries_filepath = "./data/bgem_tns_sn_discoveries.csv"

        if not os.path.exists(df_bgem_contrib_filepath):
            print("Warning: No TNS data. Save TNS data in " + tns_latest_filepath)
        else:
            # df_tns = pd.read_csv(tns_latest_filepath)
            # df_tns = df_tns.sort_values(by=['discoverydate']).reset_index(drop=True)
            #
            # ## Find contributed sources
            # df_bgem_contrib = df_tns[df_tns["reporting_group"] != "BlackGEM"]
            # df_bgem_contrib = df_bgem_contrib.dropna(subset=['internal_names'])
            # df_bgem_contrib = df_bgem_contrib[df_bgem_contrib["internal_names"].str.contains("BGEM")]
            #
            # df_bgem_discoveries = df_tns[df_tns["reporting_group"] == "BlackGEM"]
            # df_bgem_sn_discoveries = df_bgem_discoveries[df_bgem_discoveries["name_prefix"] == "SN"]

            df_bgem_contrib = pd.read_csv(df_bgem_contrib_filepath)
            df_bgem_discoveries = pd.read_csv(df_bgem_discoveries_filepath)
            df_bgem_sn_discoveries = pd.read_csv(df_bgem_sn_discoveries_filepath)


            fig = go.Figure()

            fig.add_trace(go.Scatter(
                        name            = "TNS Discoveries",
                        x               = pd.to_datetime(df_bgem_discoveries['discoverydate']),
                        y               = np.arange(1, len(df_bgem_discoveries)+1),
                        mode            = 'lines',
                        line            = dict(color='teal'),
            ))
            fig.add_trace(go.Scatter(
                        name            = "TNS Contributions",
                        x               = pd.to_datetime(df_bgem_contrib['discoverydate']),
                        y               = np.arange(1, len(df_bgem_contrib)+1),
                        mode            = 'lines',
                        line            = dict(color='darkorange'),
            ))
            fig.add_trace(go.Scatter(
                        name            = "Confirmed Supernovae",
                        x               = pd.to_datetime(df_bgem_sn_discoveries['discoverydate']),
                        y               = np.arange(1, len(df_bgem_sn_discoveries)+1),
                        mode            = 'lines',
                        line            = dict(color='darkred'),
            ))
            fig.add_vline(x=datetime(2024,10,1).timestamp() * 1000,
                    line_width=1, line_dash="dash", line_color="grey",
                    annotation_text=" GEMTOM Released",
                    annotation_position="top left",
                    annotation_font_color = "grey",
                    annotation_textangle = 90,
            )

            fig.update_layout(
                width=1200,
                height=400,
                hovermode="x",
                margin=dict(t=0, b=50, l=2, r=2),  # Set margins to reduce whitespace
                xaxis_title="Discovery Date",
                yaxis_title="Cumulative Number",)

            max_time = max([df_bgem_contrib['discoverydate'].iloc[-1], df_bgem_discoveries['discoverydate'].iloc[-1], df_bgem_sn_discoveries['discoverydate'].iloc[-1]])

            fig.update_xaxes(
                range=(datetime(2023,2,1), max_time),
            )

            cumulative_graph = plot(fig, output_type='div')



            context['number_of_discoveries'] = len(df_bgem_discoveries)
            context['number_of_contributions'] = len(df_bgem_contrib)
            context['number_of_supernovae'] = len(df_bgem_sn_discoveries)
            context['targets'] = Target.objects.all()
            context['cumulative_graph'] = cumulative_graph
            context['testing'] = 'Test successful!'

        return context
    # return render(request, "discoveries.html", context)


## =============================================================================
## ------------------- Codes for the Orphaned Transients page ------------------

class OrphanedTransientsView(LoginRequiredMixin, TemplateView):
    template_name = 'orphaned_transients.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        ## --- Update Orphaned Transients ---
        df = blackgem_rated_orphans()

        return context


    ## ===== Plot orphaned transients =====

    ## --- Step 1: The 'Orphaned Transients' Table ---
    ## Uses a Dash AG Grid

    # Initialize the Dash app
    app = DjangoDash('OrphanedTransients')

    # Read CSV data
    df = blackgem_rated_orphans()
    df = df.sort_values(by=['yes_no', 'runcat_id'], ascending=False)
    # df = df.sort_values(by=['probabilities'], ascending=False)
    # df = df.sort_values(by=['yes_no'], ascending=False)

    ## Round values for displaying
    df['ra_sml']        = round(df['ra'],3)
    df['dec_sml']       = round(df['dec'],3)
    # df['ra_std_sml']        = round(df['ra_std']*360*60,4)
    # df['dec_std_sml']       = round(df['dec_std']*360*60,4)
    df['det_sep_sml']   = round(df['det_sep'],2)
    df['qavg_sml']      = round(df['q_avg'],2)
    df['uavg_sml']      = round(df['u_avg'],2)
    df['iavg_sml']      = round(df['i_avg'],2)
    df['qrbavg_sml']    = round(df['q_rb_avg'],2)
    df['urbavg_sml']    = round(df['u_rb_avg'],2)
    df['irbavg_sml']    = round(df['i_rb_avg'],2)
    df['std_max_sml']   = round(df['std_max'],7)
    df['std_frc_sml']   = round(df['std_frc'],2)
    df['std_ang_sml']   = round(df['std_ang'],2)
    # df['probabilities'] = round(df['probabilities'],3)


    getRowStyle = {
        "styleConditions": [
            {
                "condition": "params.data.yes_no == 'No'",
                "style": {"color": "lightgrey"},
            },
        ],
        "defaultStyle": {"color": "black"},
    }

    ## Define the layout of the Dash app
    app.layout = html.Div([
        dag.AgGrid(
            id='csv-grid',
            rowData=df.to_dict('records'),
            columnDefs=[
                {'headerName': 'BGEM ID', 'field': 'runcat_id',         'minWidth': 95, 'maxWidth': 95},

                {'headerName': 'RA', 'field': 'ra_sml',                 'minWidth': 75, 'maxWidth': 75},
                {'headerName': 'Dec', 'field': 'dec_sml',               'minWidth': 75, 'maxWidth': 75},
                # {'headerName': 'RA StDev', 'field': 'ra_std_sml',       'minWidth': 95, 'maxWidth': 95},
                # {'headerName': 'Dec StDev', 'field': 'dec_std_sml',     'minWidth': 95, 'maxWidth': 95},
                {'headerName': '#Datapoints', 'field': 'n_datapoints',  'minWidth': 48, 'maxWidth': 48},

                # {'headerName': 'q min',     'field': 'q_min',       'minWidth': 75, 'maxWidth': 75},
                # {'headerName': 'q max',     'field': 'q_max',       'minWidth': 75, 'maxWidth': 75},
                {'headerName': 'q avg',     'field': 'qavg_sml',      'minWidth': 73, 'maxWidth': 73},
                {'headerName': 'q R/B', 'field': 'qrbavg_sml',    'minWidth': 75, 'maxWidth': 75},
                # {'headerName': 'q num',     'field': 'q_num',       'minWidth': 75, 'maxWidth': 75},

                # {'headerName': 'u min',     'field': 'u_min',       'minWidth': 75, 'maxWidth': 75},
                # {'headerName': 'u max',     'field': 'u_max',       'minWidth': 75, 'maxWidth': 75},
                {'headerName': 'u avg',     'field': 'uavg_sml',      'minWidth': 73, 'maxWidth': 73},
                {'headerName': 'u R/B', 'field': 'urbavg_sml',    'minWidth': 75, 'maxWidth': 75},
                # {'headerName': 'u num',     'field': 'u_num',       'minWidth': 75, 'maxWidth': 75},

                # {'headerName': 'i min',     'field': 'i_min',       'minWidth': 75, 'maxWidth': 75},
                # {'headerName': 'i max',     'field': 'i_max',       'minWidth': 75, 'maxWidth': 75},
                {'headerName': 'i avg',     'field': 'iavg_sml',      'minWidth': 73, 'maxWidth': 73},
                {'headerName': 'i R/B', 'field': 'irbavg_sml',    'minWidth': 75, 'maxWidth': 75},
                # {'headerName': 'i num',     'field': 'i_num',       'minWidth': 75, 'maxWidth': 75},

                # {'headerName': 'sum(Datapoints)', 'field': 'all_num_datapoints'},
                # {'headerName': 'Separation', 'field': 'det_sep_sml',    'minWidth': 75, 'maxWidth': 75},

                {'headerName': 'std_max', 'field': 'std_max_sml',    'minWidth': 125, 'maxWidth': 125},
                # {'headerName': 'Prob.', 'field': 'probabilities',    'minWidth': 75, 'maxWidth': 75},
                {'headerName': 'std_frc', 'field': 'std_frc_sml',    'minWidth': 75, 'maxWidth': 75},
                {'headerName': 'std_ang', 'field': 'std_ang_sml',    'minWidth': 75, 'maxWidth': 75},

                {'headerName': 'Interest', 'field': 'yes_no',       'minWidth': 90, 'maxWidth': 90},
                {'headerName': 'Notes', 'field': 'notes',               'minWidth': 75},
                # {'headerName': 'TNS?', 'field': 'tns_classification',               'minWidth': 75, 'maxWidth': 100},
                # {'headerName': 'TNS Name', 'field': 'tns_name',               'minWidth': 75, 'maxWidth': 100},


            ],
            getRowStyle=getRowStyle,
            defaultColDef={
                'sortable': True,
                'filter': True,
                'resizable': True,
            },
            columnSize="autoSize",
            dashGridOptions = {"skipHeaderOnAutoSize": True, "rowSelection": "single"},
            style={'height': '400px', 'width': '100%'},  # Set explicit height for the table
            className='ag-theme-balham'  # Add a theme for better appearance
        ),
        dcc.Store(id='selected-row-data'),  # Store to hold the selected row data, for when a row is clicked

        ## The following are sections that show information based on the row clicked
        html.Hr(),
        html.Div([
            html.Div(id='information-div', style={"flex": "3", "border": "1px solid #ccc"}),     ## For Step 2: Displays the Object ID, IAU Name, RA, and Dec
            html.Div(id='detections-list', children=[], style={"flex": "2"}),
        ], style={"display": "flex"}),
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

        primary_button_dict = {
            'font-family': 'Arial',
            'font-size': '16px',
            'color': '#007bff',
            'background-color': '#white',
            'border': '2px solid #007bff',
            'padding': '10px 20px',
            'text-align': 'center',
            'text-decoration': 'none',
            'display': 'inline-block',
            'margin': '4px 2px',
            'cursor': 'pointer',
            'border-radius': '12px'
        }

        add_to_gemtom_dict = {
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
        }

        if row_data:
            if (row_data["yes_no"] == "Yes"):
                this_colour = "green"
            else:
                this_colour = "grey"
            return html.Div([
                        html.Br(),
                        html.A(str(row_data["runcat_id"]) + ' - "' + str(row_data["notes"]) + '"', style={'color':this_colour, 'font-size':'20px'}),
                        html.Br(), html.Br(),
                        html.A("BlackView", href="https://staging.apps.blackgem.org/transients/blackview/show_runcat?runcatid=" + str(row_data['runcat_id']), target="_blank", style=primary_button_dict),
                        html.A("BlackPEARL", href="https://blackpearl.blackgem.org/analyze.php", target="_blank", style=primary_button_dict),
                        html.A("GEMTOM", href='/transients/'+str(row_data['runcat_id']), target="_blank", style=primary_button_dict),
                        html.A("SIMBAD", href="https://simbad.u-strasbg.fr/simbad/sim-coo?Coord=" + str(row_data['ra']) + "d" + str(row_data['dec']) + "d&CooFrame=FK5&CooEpoch=2000&CooEqui=2000&CooDefinedFrames=none&Radius=10&Radius.unit=arcsec&submit=submit+query&CoordList=", target="_blank", style=primary_button_dict),
                        html.A("Vizier", href="https://vizier.cds.unistra.fr/viz-bin/VizieR-4?-c=" + str(row_data['ra']) + "%20" + str(row_data['dec']) + "&-c.u=arcsec&-c.r=1.5&-c.eq=J2000&-c.geom=r&-out.max=50&-out.add=_r", target="_blank", style=primary_button_dict),
                        # html.Br(), html.Br(),
                        html.Div(html.Button('Add to GEMTOM', id='call-function-button', n_clicks=0, style=add_to_gemtom_dict)),
                        html.P(id='button-click-message'),  # Div to display the message when button is clicked

                        # html.Form(method="post", action="rate_target", className = "image-form", id='rate-target-form',
                        #    children=[
                        #        dbc.Input(id="notes", name="notes", type="text", placeholder="Notes"),
                        #        dbc.Input(id="submit_yes", name="yes_no", type="submit", value="Yes"),
                        #        dbc.Input(id="submit_no",  name="yes_no", type="submit", value="No"),
                        #    ],
                        # ),

                                # First column with text field and buttons
                        html.Div(
                            children=[
                                html.Label("Update Notes:", style={'font-family': 'Arial', 'text-align': 'center', 'color': 'grey', "font-style": "italic"}),
                                dcc.Input(
                                    id="notes",
                                    type="text",
                                    placeholder="Notes",
                                    style={"marginLeft": "10px", "marginRight": "10px", "padding" : "12px 20px", 'font-size': '16px',},
                                ),
                                html.Button(
                                    "Yes",
                                    id="Yes",
                                    n_clicks=0,
                                    # style={"marginRight": "10px"},
                                    style={**primary_button_dict, ** {'background-color': '#ffffff'}},
                                ),
                                html.Button(
                                    "No",
                                    id="No",
                                    n_clicks=0,
                                    style={**primary_button_dict, ** {'background-color': '#ffffff'}},
                                ),
                                html.Div(id="feedback", style={"marginTop": "10px", "color": "black"}),
                            ],
                        ),

                        ], style={'font-family': 'Arial', 'text-align': 'center'}
            )

    # <form method="post" action="{% url 'rate_target' %}" class="image-form">
    #     {% csrf_token %}
    #     <input type="hidden" name="id" value="{{ bgem_id }}">
    #     <input type="hidden" name="night" value="{{ obs_date }}">
    #     <!-- <input type="hidden" name="yes_no" value="Real"> -->
    #     <input type="text" name="notes" placeholder="Notes">
    #     <br>
    #     <button type="submit" name="yes_no" value="Yes" class="btn btn-outline-primary">Yes</button>
    #     <button type="submit" name="yes_no" value="No" class="btn btn-outline-primary">No</button>
    # </form>

        return html.Div(
            html.Em(html.P("Select a row")), style={"text-align":"center", "font-size": "18px", "font-family":"sans-serif", "color":"grey"}
        )


    ## --- Re-rate the Transient ---

    @app.callback(
        Output("feedback", "children"),
        [Input("Yes", "n_clicks"),
         Input("No", "n_clicks"),
         Input('selected-row-data', 'data'),
         ],
        [State("notes", "value")]
    )
    def handle_buttons(yes, no, row_data, notes):
        # if not input_value:
        #     return "Please enter a value."
        #
        ctx = dash.callback_context
        if not ctx.triggered:
            return ""
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if notes == None:
            notes = ""

        # print(row_data)
        id = row_data['runcat_id']

        if button_id == "Yes":
            true_rate_target(id, button_id, notes, "19700101")
            if len(notes) == 0:
                return f"Target updated: Interest = Yes. (This will take a while to update)"
            else:
                return f"Target updated: Interest = Yes; '{notes}'. (This will take a while to update)"

        elif button_id == "No":
            true_rate_target(id, button_id, notes, "19700101")
            if len(notes) == 0:
                return f"Target updated: Interest = No. (This will take a while to update)"
            else:
                return f"Target updated: Interest = No; '{notes}'. (This will take a while to update)"

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


    ## --- Find the individual detections of a given source ---
    @app.callback(
        Output('detections-list', 'children'),
        Input('selected-row-data', 'data'),
        prevent_initial_call=True  # Prevent the callback from being called when the app loads
    )
    def update_new_grid(row_data):
        if not row_data:
            return []

        print(row_data)

        bgem_id = row_data['runcat_id']

        df_bgem_lightcurve, df_limiting_mag = get_lightcurve_from_BGEM_ID(bgem_id)

        df_new = df_bgem_lightcurve.rename(columns={
            'a.xtrsrc'          : "xtrsrc",
            'x.ra_psf_d'        : "ra_psf_d",
            'x.dec_psf_d'       : "dec_psf_d",
            'x.flux_zogy'       : "flux_zogy",
            'x.fluxerr_zogy'    : "fluxerr_zogy",
            'x.mag_zogy'        : "mag_zogy",
            'x.magerr_zogy'     : "magerr_zogy",
            'i."mjd-obs"'       : "mjd",
            'i."date-obs"'      : "date_obs",
            'i.filter'          : "filter",
        })

        df_new['xtrsrc'] = df_new['xtrsrc'].apply(lambda x: f'[{int(x)}](https://staging.apps.blackgem.org/transients/blackview/show_xtrsrc/{int(x)})' if x > 0 else 'None')

        new_grid = dag.AgGrid(
            id='observation-grid',
            rowData=df_new.to_dict('records'),
            columnDefs=[
                        {'headerName': 'a.xtrsrc', 'field':  'xtrsrc', 'cellRenderer': 'markdown', "linkTarget":"_blank"},
                        {'headerName': 'MJD', 'field':  'mjd',
                            "valueFormatter": {"function": "d3.format('.5f')(params.value)"}},
                        {'headerName': 'Filter', 'field': 'filter'},
                        {'headerName': 'mag', 'field':  'mag_zogy',
                            "valueFormatter": {"function": "d3.format('.3f')(params.value)"}},
            ],
            defaultColDef={
                'sortable': True,
                'filter': True,
                'resizable': True,
                'editable': False,
            },
            columnSize="autoSize",
            dashGridOptions={
                "skipHeaderOnAutoSize": True,
                "rowSelection": "single",
            },
            style={'height': '100%', 'width': '100%'},  # Set explicit height for the grid
            className='ag-theme-balham'  # Add a theme for better appearance
        )

        return [new_grid]  # Return the new grid as the content of the container

    ## --- Display the link to Transients, the 'Add to GEMTOM' button, and the full data ---

    ## First, create the 'Add to GEMTOM' button
    ## Callback to handle button click:
    @app.callback(
        Output('button-click-message', 'children'),  # Allow multiple outputs to the same component
        Input('call-function-button', 'n_clicks'),
        State('selected-row-data', 'data'),
        prevent_initial_call=True  # Prevent the callback from being called when the app loads
    )
    # Function to add the transient to GEMTOM:
    def transient_to_GEMTOM(n_clicks, row_data):
        if n_clicks > 0 and row_data:

            id      = str(row_data['runcat_id'])
            name    = all_stats_from_bgem_id(id)[0]
            ra      = str(row_data['ra'])
            dec     = str(row_data['dec'])

            if name == None or name == "None":
                all_stats = all_stats_from_bgem_id(id)
                name = all_stats[0]

            add_to_GEMTOM(id, name, ra, dec)

            return [html.A(f"Transient added to GEMTOM as " + name, style={'display': 'inline-block'}), html.A("Please see the Targets page", href="/targets/", target="_blank", style={'text-decoration':'None', 'display': 'inline-block'}), html.P(".", style={'display': 'inline-block'})]
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
                # columns=[{'name': k, 'id': k} for k in row_data.keys() if k in ['runcat_id', 'ra', 'dec', 'ra_std', 'dec_std', 'xtrsrc', 'n_datapoints']],
                columns=[{'name': k, 'id': k} for k in row_data.keys() if k in ['runcat_id', 'ra', 'dec', 'xtrsrc', 'n_datapoints']],
                style_table={'margin': 'auto'}, style_cell={'textAlign': 'center', 'padding': '5px'}, style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
            ## q mag
            table_2 = dash_table.DataTable(
                data=[row_data],
                columns=[{'name': k, 'id': k} for k in row_data.keys() if 'q_' in k],
                style_table={'margin': 'auto'}, style_cell={'textAlign': 'center', 'padding': '5px'}, style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
            ## u mag
            table_3 = dash_table.DataTable(
                data=[row_data],
                # columns=[[{'name': k, 'id': k} for k in row_data.keys()][i] for i in [13,14,15,16,17,44]],
                columns=[{'name': k, 'id': k} for k in row_data.keys() if 'u_' in k],
                style_table={'margin': 'auto'}, style_cell={'textAlign': 'center', 'padding': '5px'}, style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
            ## i mag
            table_4 = dash_table.DataTable(
                data=[row_data],
                columns=[{'name': k, 'id': k} for k in row_data.keys() if 'i_' in k],
                style_table={'margin': 'auto'}, style_cell={'textAlign': 'center', 'padding': '5px'}, style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
            table_5 = dash_table.DataTable(
                data=[row_data],
                columns=[{'name': k, 'id': k} for k in row_data.keys() if k in ['std_max', 'std_min', 'std_frc', 'std_ang']],
                style_table={'margin': 'auto'}, style_cell={'textAlign': 'center', 'padding': '5px'}, style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
            ## Extra 1
            table_6 = dash_table.DataTable(
                data=[row_data],
                columns=[{'name': k, 'id': k} for k in row_data.keys() if k in ['all_num_datapoints', 'det_sep', 'yes_no', 'notes']],
                style_table={'margin': 'auto'}, style_cell={'textAlign': 'center', 'padding': '5px'}, style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
            ## Extra 2
            return [table_1] + [table_2] + [table_3] + [table_4] + [table_5] + [table_6]

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

def mjd_to_datetime(mjd):
    t = Time(mjd, format='mjd')

    return t.iso

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


class LiveFeed(LoginRequiredMixin, TemplateView):
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
                dcc.Graph(id='live-update-graph_1', figure=go.Figure(layout={'margin': dict(l=20, r=20, t=40, b=30), 'height': 260})),
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

                    if np.isfinite(min_x) == False: min_x = np.min(np.min(df_x))
                    if np.isfinite(max_x) == False: max_x = np.max(np.min(df_x))
                    if np.isfinite(max_y) == False: max_y = np.max(np.min(df_y))


                    fig = plot_BGEM_lightcurve(df_bgem_lightcurve, df_limiting_mag)
                    time_last_update = Time(df_bgem_lightcurve['i."mjd-obs"'].iloc[-1], format='mjd')

                ## Get time since last update
                print("\n\nTesting...")
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
                    margin=dict(l=20, r=0, t=25, b=30),  # Set margins to reduce whitespace
                    title_x=0.5,  # Center the title
                    title_y=0.97,  # Adjust title position
                    height=263
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



@login_required
def LiveFeed_BGEM_ID_View(request, bgem_id):
    '''
    Finds and displays data from a certain date.
    '''

    df_bgem_lightcurve, df_limiting_mag = get_lightcurve_from_BGEM_ID(bgem_id)
    print(df_bgem_lightcurve)
    print(df_bgem_lightcurve.columns)

    response = "You're looking at BlackGEM transient %s."


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

    ## --- Location on Sky ---
    fig = plot_BGEM_location_on_sky(df_bgem_lightcurve, ra, dec)
    location_on_sky = plot(fig, output_type='div')

    ## --- Lightcurve ---
    fig = plot_BGEM_lightcurve(df_bgem_lightcurve, df_limiting_mag)
    lightcurve = plot(fig, output_type='div')



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

        ## Save Data
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

        # except InvalidFileFormatException as e:
        #     print("Invalid File Format Exception!")
        #     iffe = "File Format Invalid"
        #     print(e)
        #     ReducedDatum.objects.filter(data_product=dp).delete()
        #     dp.delete()

        except Exception as e:
            print("Exception!")
            print(e)
            if e: iffe2 = e
            else: iffe2 = "Unknown Error"
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
                messages.error(self.request, 'There was a problem processing your file: {0} -- Error: {1}. Is the transient server online?'.format(str(dp), iffe2))
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





## =============================================================================
## -------------------- Functions for the Observation Night --------------------

from .models import Observation
from .forms import ObservationForm

## Observatories:
observatory_list = [
    "Roque de los Muchachos Observatory (La Palma, Spain)",
    "Teide Observatory (Tenerife, Spain)",
    "Georgian National Astrophysical Observatory (Abastumani, Georgia)",
    "Abruzzo Astronomical Observatory - TNT (INAF-OA Abruzzo, Italy)",
    "Apache Point Observatory (New Mexico, USA)",
    "Aras de los Olmos Observatory (Spain)",
    "Asiago Astrophysical Observatory (Italy)",
    "Assy-Turgen Observatory (Kazakhstan)",
    "Bohyunsan Optical Astronomy Observatory (Korea)",
    "Brorfelde Observatory (Denmark)",
    "Byurakan Observatory (Armenia)",
    "Calar Alto Observatory (Almeria, Spain)",
    "Campo Imperatore Observatory (INAF-OA Abruzzo, Italy)",
    "Cerro Pachon Observatory (Chile)",
    "Cerro Paranal Observatory (Chile)",
    "Cerro Tololo Observatory (Chile)",
    "Dominion Astrophysical Observatory (Canada)",
    "El Leoncito Observatory (Argentina)",
    "Fred Lawrence Whipple Observatory (Arizona, USA)",
    "Guillermo Haro Astrophysical Observatory (Sonora, Mexico)",
    "Haute-Provence Observatory (France)",
    "Helmos Observatory (Greece)",
    "Iranian National Observatory, Mt. Gargash (Iran)",
    "Javalambre Astronomical Observatory (Teruel, Spain)",
    "Kitt Peak Observatory (Arizona, USA)",
    "Kryoneri Observatory (Greece)",
    "La Luz Observatory (Mexico)",
    "La Silla Observatory (Chile)",
    "Large Millimeter Telescope (Mexico)",
    "Las Campanas Observatory (Chile)",
    "Lick Observatory (California, USA)",
    "Lijiang Observatory (China)",
    "Loiano Observatory (Italy)",
    "Lulin Observatory (Taiwan)",
    "Mauna Kea Observatory (Hawaii, USA)",
    "McDonald Observatory (Texas, USA)",
    "Montsec Astronomical Observatory (Lleida, Spain)",
    "Mount Graham International Observatory (Arizona, USA)",
    "Mount John Observatory (Tekapo, New Zeland)",
    "Mount Kent Observatory (Queensland, Australia)",
    "Mount Lemon Optical Astronomy Observatory (Arizona, USA)",
    "Nanshan Station, Xinjiang Astronomical Observatory (China)",
    "National Astronomical Observatory Rozhen (Bulgaria)",
    "Observatoire de la Cote dAzur (France)",
    "Ondrejov Observatory",
    "Palomar Observatory (California, USA)",
    "Pearl Station (Canada)",
    "Pico dos Dias Observatory (Brazil)",
    "Pine Mountain Observatory (USA)",
    "Piszkesteto Observatory (Hungary)",
    "San Pedro Martir Observatory (Mexico)",
    "Santa Martina Observatory (Chile)",
    "Seimei Telescope (Japan)",
    "Sertao de Itaparica Observatory (Brazil)",
    "Siding Spring Observatory (Australia)",
    "Astronomical Observatory of the University of Siena (Italy)",
    "Serra La Nave Observatory (INAF - OA Catania, Italy)",
    "Sierra Nevada Observatory (Granada, Spain)",
    "Skinakas Observatory (Crete, Greece)",
    "Sobaeksan Optical Astronomy Observatory (Korea)",
    "Special Astrophysical Observatory (Russia)",
    "Sutherland Observatory (South Africa)",
    "Tian Shan Observatory (Kazakhstan)",
    "Thai National Observatory, NARI (Thailand)",
    "Thueringer Landessternwarte Tautenburg (Germany)",
    "Tubitak National Observatory (Turkey)",
    "Vidojevica Astronomical Station (Serbia)",
    "Virgin Island Robotic Telescope (US Virgin Islands)",
    "Wendelstein Observatory (Germany)",
    "Wise Observatory (Israel)",
    "Xinglong Observatory (China)",
    "Zimmerwald Observatory (Bern, Switzerland)",
]



def plot_altitude_graph(name, target_ra, target_dec, night, location):

    output_dir = "./data/AltitudeGraphs/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ## ===== Find the times of the night =====
    today_time    = Time(night)+1                                                            ## Set Time

    now_time    = Time(datetime.utcnow(), scale='utc')                                        ## Find current time

    location_name = location

    try:
        location    = EarthLocation.of_site(location_name)                                            ## Set Location
        valid_location = True
    except:
        print("Location not recognised:")
        print(location)
        valid_location = False

    if valid_location:
        sunset        = Observer.at_site(location_name).sun_set_time(today_time, which="previous")    ## Find Sunset
        # sunrise        = Observer.at_site(location_name).sun_rise_time(today_time, which="next")        ## Find Sunrise
        sunrise        = Observer.at_site(location_name).sun_rise_time(sunset, which="next")        ## Find Sunrise
        print(sunset)
        print(sunrise)

        ## Define the location
        # location = EarthLocation(lat=-29.25738889*u.deg, lon=-70.73791667*u.deg, height=2200*u.m)
        observer = Observer(location=location, timezone='UTC')

        # Evening twilight times
        civil_twilight_evening          = observer.twilight_evening_civil(          today_time+0.1, which='previous')
        nautical_twilight_evening       = observer.twilight_evening_nautical(       today_time+0.1, which='previous')
        astronomical_twilight_evening   = observer.twilight_evening_astronomical(   today_time+0.1, which='previous')

        # Morning twilight times
        civil_twilight_morning = observer.twilight_morning_civil(sunset, which='next')
        nautical_twilight_morning = observer.twilight_morning_nautical(sunset, which='next')
        astronomical_twilight_morning = observer.twilight_morning_astronomical(sunset, which='next')

        print("")
        print("")

        # print("Sunset at:  {0.iso}".format(sunset))
        # print("Sunrise at: {0.iso}".format(sunrise))
        ## -- Find each hour section --
        print("Plotting altitude graph...")

        # utcoffset        = -5*u.hour
        midnight        = today_time
        now_time_rel    = (now_time.mjd - today_time.mjd)*24
        sunset_rel      = (sunset.mjd - today_time.mjd)*24
        sunrise_rel     = (sunrise.mjd - today_time.mjd)*24
        delta_midnight  = np.linspace(-24, 24, 2000)*u.hour
        delta_midnight2 = np.linspace(-24, 24, 2000)

        ## Get the sun position
        print("Getting Sun position...")

        times            = midnight+delta_midnight
        frame            = AltAz(obstime=times, location=location)
        sunaltazs        = get_sun(times).transform_to(frame)

        ## Get the moon position
        print("Getting Moon position...")
        moon            = get_moon(times)
        moonaltazs         = moon.transform_to(frame)

        ## Get the positions of our taargets
        print("Getting Target positions...")
        target_coords = SkyCoord(target_ra, target_dec, unit=(u.deg), frame='icrs')
        target_altazs = target_coords.transform_to(frame)

        ## Find sunrise and sunset
        sunset_hour     = (today_time.mjd-sunset.mjd)*24
        sunrise_hour    = (sunrise.mjd-today_time.mjd)*24

        civ_twilight_sunset = (civil_twilight_evening.mjd-today_time.mjd)*24
        nau_twilight_sunset = (nautical_twilight_evening.mjd-today_time.mjd)*24
        ast_twilight_sunset = (astronomical_twilight_evening.mjd-today_time.mjd)*24

        civ_twilight_sunrise = (civil_twilight_morning.mjd-today_time.mjd)*24
        nau_twilight_sunrise = (nautical_twilight_morning.mjd-today_time.mjd)*24
        ast_twilight_sunrise = (astronomical_twilight_morning.mjd-today_time.mjd)*24


        moon_distance = target_altazs.separation(moonaltazs)

        moon_dist_x = delta_midnight[0::50]
        moon_dist_y = moonaltazs.alt[0::50]
        moon_dist_t = moon_distance [0::50]


        ## ----- Plot! -----
        fig, ax1 = plt.subplots(figsize=(8, 8))

        xlimits = [-np.ceil(sunset_hour), np.ceil(sunrise_hour)]
        # xlimits = [-np.ceil(sunset_rel), np.ceil(sunrise_rel)]
        # xlimits = [sunset_rel-1, sunrise_rel+1]

        plt.title(str(name) + "\nLocation: " + str(location_name) + "     RA: " + str(target_ra) + "  Dec: " + str(target_dec) + "     Night of " + night)
        ax1 = plt.gca()

        ax1.plot(delta_midnight, moonaltazs.alt, color=[0.75]*3, ls='--', zorder=7, label='Moon')
        # plt.text(moon_dist_x.value, moon_dist_y.value, moon_dist_t.value)
        for x, y, text in zip(moon_dist_x.value, moon_dist_y.value, moon_dist_t.value):
            if (x > xlimits[0]) and (y > 0) and (x < xlimits[1]):
                ax1.text(x, y, "%.0f°"%text, ha="center", va="center", backgroundcolor="white", zorder=7)

        colour            = "darkviolet"

        ## Plot the full altitude plot
        ax1.plot(delta_midnight, target_altazs.alt, lw=2, ls="-", color=colour, zorder=18, label='Target')

        # ## Plot when it's right now!
        # plt.vlines(now_time_rel, 0, 90, linestyle="--", linewidth=1, zorder=9, color="r")

        ax1.vlines(sunset_rel,           0, 90, linestyle="-",  linewidth=2, zorder=5, color="k")
        ax1.vlines(civ_twilight_sunset,  0, 90, linestyle="--", linewidth=2, zorder=5, color="grey")
        ax1.vlines(nau_twilight_sunset,  0, 90, linestyle="--", linewidth=1, zorder=5, color="grey")
        ax1.vlines(ast_twilight_sunset,  0, 90, linestyle=":",  linewidth=1, zorder=5, color="grey")
        ax1.vlines(ast_twilight_sunrise, 0, 90, linestyle=":",  linewidth=1, zorder=5, color="grey")
        ax1.vlines(nau_twilight_sunrise, 0, 90, linestyle="--", linewidth=1, zorder=5, color="grey")
        ax1.vlines(civ_twilight_sunrise, 0, 90, linestyle="--", linewidth=2, zorder=5, color="grey")
        ax1.vlines(sunrise_rel,          0, 90, linestyle="-",  linewidth=2, zorder=5, color="k")

        ax1.text(sunset_rel,           20, "Sunset",                rotation='vertical', ha="center", va="center", color="grey", backgroundcolor="white", zorder=6)
        ax1.text(civ_twilight_sunset,  20, "Civil Twilight",        rotation='vertical', ha="center", va="center", color="grey", backgroundcolor="white", zorder=6)
        ax1.text(nau_twilight_sunset,  20, "Nautical Twilight",     rotation='vertical', ha="center", va="center", color="grey", backgroundcolor="white", zorder=6)
        ax1.text(ast_twilight_sunset,  20, "Astronomical Twilight", rotation='vertical', ha="center", va="center", color="grey", backgroundcolor="white", zorder=6)
        ax1.text(ast_twilight_sunrise, 20, "Astronomical Twilight", rotation='vertical', ha="center", va="center", color="grey", backgroundcolor="white", zorder=6)
        ax1.text(nau_twilight_sunrise, 20, "Nautical Twilight",     rotation='vertical', ha="center", va="center", color="grey", backgroundcolor="white", zorder=6)
        ax1.text(civ_twilight_sunrise, 20, "Civil Twilight",        rotation='vertical', ha="center", va="center", color="grey", backgroundcolor="white", zorder=6)
        ax1.text(sunrise_rel,          20, "Sunrise",               rotation='vertical', ha="center", va="center", color="grey", backgroundcolor="white", zorder=6)


        ## Various legends and axes properties
        ax1.legend(loc='lower center', ncol=3, shadow=True).set_zorder(15)
        ax1.set_xlim(xlimits)
        ax1.set_xticks(np.arange(xlimits[0], xlimits[1], 1))
        ax1.grid(linestyle=":", zorder=5)
        ax1.set_ylim(0, 90)
        ax1.set_xlabel('Hours from UTC Midnight')
        ax1.set_ylabel('Altitude [deg]')

        ## Plot Airmass
        yticks = np.arange(10,91,10)
        yticks_true = 1/np.cos(np.arange((8/9*np.pi)/2,-0.1,-np.pi/18))
        yticks_true = ["%.2f"%x for x in yticks_true]
        secax = ax1.secondary_yaxis('right')
        secax.set_ylabel('Airmass')
        secax.set_yticks(yticks, yticks_true)

        savename = output_dir + "/AltitudePlot_%.5f"%float(target_ra) + "_%.5f"%float(target_dec) + "_" + night + "_" + str(location_name) + ".png"
        plt.savefig(savename, bbox_inches='tight')

        print("Plotted.")





class ObservationNightView(TemplateView):
    template_name = 'observations.html'

    def observation_form_view(self, request):
        form = ObservationForm(request.POST or None)

        print(form)

        if request.method == "POST" and form.is_valid():


            return redirect('observations')  # Redirect to refresh the form after submission

        return render(request, 'observations.html', {'form': form})


    def get_context_data(self, **kwargs):
        return {'targets': Target.objects.all()}


    def post(self, request, **kwargs):
        source_ra  = float(request.POST['num1'])
        source_dec = float(request.POST['num2'])

        print("-- ZTF: Looking for target...", end="\r")
        lcq = lightcurve.LCQuery.from_position(source_ra, source_dec, 5)
        ZTF_data_full = pd.DataFrame(lcq.data)
        ZTF_data = pd.DataFrame({'JD' : lcq.data.mjd, 'Magnitude' : lcq.data.mag, 'Magnitude_Error' : lcq.data.magerr})

        if len(ZTF_data) == 0:
            raise Exception("-- ZTF: Target not found. Try AAVSO instead?")

        print("-- ZTF: Looking for target... target found.")
        print(lcq.__dict__)

        df = ZTF_data # replace with your own data source

        fig = px.scatter(df, x='JD', y='Magnitude')
        fig.update_layout(
            yaxis = dict(autorange="reversed")
        )
        return HttpResponse("Closest target within 5 arcseconds:" + fig.to_html() + ZTF_data_full.to_html())


@login_required
def set_observed(request):
    '''
    Rates a target as observed or not
    '''

    print("\n\n\nFull Request:")
    print(request.POST)
    print("\n\n\n")

    num = request.POST.get('Num')
    night = request.POST.get('Night')
    block_num = request.POST.get('block_num')
    if len(num) > 0: num = np.int64(num)

    observed_new = request.POST.get('observed_new')

    data_file = "./data/planned_observations_data.csv"
    obs_data = pd.read_csv(data_file)

    print("Night:", night)

    print(type(obs_data['targ_num'][0]))
    # print(type(int(num)))
    print(obs_data['targ_num'][0])
    print(num)
    print(obs_data['targ_num'][0] == num)

    index = obs_data.index[obs_data['targ_num'] == num]
    obs_data.loc[index,"observed"] = observed_new

    obs_data.to_csv(data_file, index=False)

    ## Are we looking at a certain night?
    if night and block_num: return redirect('/observations/?show_night=' + night + '&show_block=' + block_num)
    elif night: return redirect('/observations/?show_night=' + night)
    elif block_num: return redirect('/observations/?show_block=' + block_num)
    else:
        return redirect('/observations/')

@login_required
def delete_observation(request):
    '''
    Deletes observation
    '''

    num = request.POST.get('Num')
    night = request.POST.get('Night')
    block_num = request.POST.get('block_num')
    if len(num) > 0: num = np.int64(num)

    data_file = "./data/planned_observations_data.csv"
    obs_data = pd.read_csv(data_file)

    index = obs_data.index[obs_data['targ_num'] == num]

    obs_data = obs_data.drop(index)

    obs_data.to_csv(data_file, index=False)

    ## Are we looking at a certain night?
    if night and block_num: return redirect('/observations/?show_night=' + night + '&show_block=' + block_num)
    elif night:             return redirect('/observations/?show_night=' + night)
    elif block_num:         return redirect('/observations/?show_block=' + block_num)
    else:
        return redirect('/observations/')


def df_to_lists(obs_data):

    altitude_plot_list = []
    telescope_list = []
    for ra, dec, night, start_night, telescope, location in zip(obs_data.ra, obs_data.dec, obs_data.night, obs_data.start_night, obs_data.telescope, obs_data.location):
        if night != "Any": used_night = night
        else: used_night = start_night
        # altitude_path = "./data/AltitudeGraphs/AltitudePlot_" + str(ra) + "_" + str(dec) + "_" + str(night) + "_" + str(location) + ".png"
        altitude_path = "./data/AltitudeGraphs/AltitudePlot_%.5f"%float(ra) + "_%.5f"%float(dec) + "_" + str(used_night) + "_" + str(location) + ".png"
        print(altitude_path)
        if os.path.exists(altitude_path):
            altitude_plot_list.append(altitude_path)
        else:
            altitude_plot_list.append("")
        if telescope == telescope:
            telescope_list.append(telescope)
        else:
            telescope_list.append(" ")

    print("altitude_plot_list:")
    print(altitude_plot_list)
    # print("telescope_list:")
    # print(telescope_list)

    df_lists = zip(
                list(obs_data.id.astype(int)),
                list(obs_data.name),
                list(obs_data.targ_num),
                list(obs_data.ra),
                list(obs_data.dec),
                list(obs_data.notes),
                list(obs_data.night),
                list(obs_data.priority),
                telescope_list,
                list(obs_data.location),
                list(obs_data.submitter),
                list(obs_data.observed),
                list(obs_data.gemtom_id),
                altitude_plot_list,
                # list("./data/AltitudeGraphs/AltitudePlot_" + obs_data["ra"].astype(str) + "_" + obs_data["dec"].astype(str) + "_" + obs_data["night"].astype(str) + ".png"),

    )

    return df_lists

@login_required
def submit_observation(request):

    ## Get Initial Data
    data_file = "./data/planned_observations_data.csv"
    too_file = "./data/too_data.csv"

    if os.path.exists(data_file):
        obs_data = pd.read_csv(data_file)

        df_lists = df_to_lists(obs_data)

        nights = obs_data.night
        nights = pd.concat([pd.Series(["All"]), nights])
        nights = nights.drop_duplicates()
        print(nights)
        if "Any" in nights.unique():
            print("BARRKKARABBARKBARBKABRKBARK!")
            nights = nights[nights != "Any"]

    else:
        df_lists = []
        nights = []

    if os.path.exists(too_file):
        ToO_data = pd.read_csv(too_file)
        num         = list(ToO_data.num)
        telescopes  = list(ToO_data.Telescope)
        # locations   = list(ToO_data.Location)
        date_start  = list(ToO_data.date_start)
        date_close  = list(ToO_data.date_close)
        PI          = list(ToO_data.Name)

        observations = zip(num, telescopes, date_start, date_close, PI)

    else:
        observations = zip([],[],[],[],[])



    # print(nights.drop_duplicates())

    context = {
            "df"    : df_lists,
            "message" : [""],
            "nights" : nights,
            "observations" : list(observations),
    }


    ## Remake Altitude Plots
    remake_altitude = False
    if remake_altitude:
        print("Remaking Altitude Plots...")
        for name, ra, dec, night, start_night, telescope, location in zip(obs_data.name, obs_data.ra, obs_data.dec, obs_data.night, obs_data.start_night, obs_data.telescope, obs_data.location):
            if night != "Any": used_night = night
            else: used_night = start_night
            plot_altitude_graph(name, ra, dec, used_night, location)



    ## Handle selecting a single night
    if 'show_night' in request.GET:
        print("Request: Show Night or Block")
        show_night = request.GET.get('show_night')
        show_block = request.GET.get('show_block')

        print(show_night)
        print(show_block)

        if not show_block:          obs_data = pd.read_csv(data_file)
        elif show_block == "All":   obs_data = pd.read_csv(data_file)
        else:
            # print("block_nums:", obs_data["block_num"].values)
            # for value in obs_data["block_num"].values:
            #     print(value)
            #     print(type(value))
            #     print(type(show_block))
            #     print(show_block == value)

            obs_data = obs_data[obs_data["block_num"]==np.int64(show_block)]

        if show_night and show_night != "All":
            # obs_data = obs_data[obs_data["night"]==show_night]
            obs_data = obs_data[(obs_data["night"]==show_night) | (obs_data["night"]=="Any") ]

            if show_night != "Any":
                drop_list = []
                for index in range(len(obs_data)):
                    if obs_data["night"].iloc[index] == "Any":
                        night_datetime = datetime.strptime(show_night, '%Y-%m-%d')
                        start_datetime = datetime.strptime(obs_data["start_night"].iloc[index], '%Y-%m-%d')
                        close_datetime = datetime.strptime(obs_data["close_night"].iloc[index], '%Y-%m-%d')
                        print(show_night)
                        print(start_datetime)
                        print(close_datetime)
                        print(night_datetime > start_datetime)
                        print(night_datetime < close_datetime)
                        if night_datetime < start_datetime or night_datetime > close_datetime:
                            drop_list.append(index)

                obs_data = obs_data.drop(obs_data.index[drop_list])

            # df[(df['age'] < 25) & df['name'].str.endswith('e')]

        # print("obs_data:")
        # print(obs_data)

        df_lists = df_to_lists(obs_data)

        context["df"] = df_lists
        context["show_night"] = show_night
        context["show_block"] = show_block

        return render(request, "observations.html", context)
        # return redirect('/observations/')


    elif 'ra' in request.GET:
        print("Request: Submit New Target")

        # Get the data from the form
        name        = request.GET.get('name')
        ra          = request.GET.get('ra')
        dec         = request.GET.get('dec')
        notes       = request.GET.get('notes')
        night       = request.GET.get('night')
        priority    = request.GET.get('priority')
        submitter   = request.GET.get('submitter')
        gemtom_id   = int(request.GET.get('gemtom_id'))
        num         = request.GET.get('observation')

        if not night:
            night = "Any"
            any_night = True
            print("\n\nAny Night!\n\n")
            if not num or int(num) == 0:

                start_night = datetime.today().strftime("%Y-%m-%d")
                close_night = (datetime.today()+timedelta(30)).strftime("%Y-%m-%d")
                print(start_night)
                print(close_night)

            else:
                index = ToO_data.index[ToO_data['num'] == int(num)]
                start_night = ToO_data.loc[index,"date_start"].values[0]
                close_night = ToO_data.loc[index,"date_close"].values[0]
                print(start_night)
                print(close_night)
        else:
            start_night = night
            close_night = night


        print("Num:")
        print(num)

        if num and (int(num) > 0):
            index = ToO_data.index[ToO_data['num'] == int(num)]
            print(ToO_data.loc[index,"Name"].values[0])
            print(ToO_data.loc[index,"Telescope"].values[0])
            print(ToO_data.loc[index,"Location"].values[0])

            block_num = str(ToO_data.loc[index,"num"].values[0])
            telescope = str(ToO_data.loc[index,"Telescope"].values[0])
            location = str(ToO_data.loc[index,"Location"].values[0])
            start_night = ToO_data.loc[index,"date_start"].values[0]
            close_night = ToO_data.loc[index,"date_close"].values[0]
            try:
                print("Location:")
                print(location)
                telescope_lon = coords.EarthLocation.of_site(location).geodetic.lon.deg
                telescope_lat = coords.EarthLocation.of_site(location).geodetic.lat.deg

                # print("Telescope:", telescope)
                # print("Location:", location)
                print("Lon:", telescope_lon, " Lat:", telescope_lat)
                make_altitude = True
            except:
                print("Telescope location not understood.")
                make_altitude = False

        else:
            block_num = " "
            telescope = " "
            location = " "
            make_altitude = False

        success, message  = ra_dec_checker(ra, dec)
        if not success:
            context["message"] = message
            print(message)
            return render(request, "observations.html", context)

        if submitter:
            if len(submitter) <= 1:
                submitter = "Unknown"
        else:
            submitter = "Unknown"

        df_nearest_targets = transient_cone_search(ra, dec, radius=10)
        if len(df_nearest_targets) > 0:
            nearest_id = str(int(df_nearest_targets["id"][0]))
        else:
            nearest_id = 0
        print(nearest_id)
        # 186.2724, 12.8787, 60

        # print("len(obs_data.targ_num):")
        # print(len(obs_data.targ_num))
        # print(max(obs_data.targ_num))
        if os.path.exists(data_file):
            if len(obs_data.targ_num) > 0:   new_num = max(obs_data.targ_num)+1
            else:                       new_num = 1
        else:
            new_num = 1
        # print("new_num:", new_num)

        if name:
            if len(name) == 0:
                if nearest_id:  name = nearest_id
                else:           name = "Source #" + new_num
        else:
            if nearest_id:  name = nearest_id
            else:           name = "Source #" + new_num

        new_output = pd.DataFrame({
            'id'            : nearest_id,
            'name'          : name,
            'targ_num'      : [new_num],
            'ra'            : [str(float(ra))[:10]],
            'dec'           : [str(float(dec))[:10]],
            'notes'         : [notes],
            'night'         : [night],
            'start_night'   : [start_night],
            'close_night'   : [close_night],
            'priority'      : [priority],
            'block_num'     : [block_num],
            'telescope'     : [telescope],
            'location'      : [location],
            'submitter'     : [submitter],
            'gemtom_id'     : [gemtom_id],
            'observed'      : [False],
        })

        if os.path.exists(data_file):
            output = pd.read_csv(data_file)
            full_output = pd.concat([output,new_output]).reset_index(drop=True)
            full_output = full_output.sort_values(by=["night"])

            full_output.to_csv(data_file, index=False)

        else:
            new_output.to_csv(data_file, index=False)

        obs_data = pd.read_csv(data_file)
        # print(obs_data.columns)
        # for col in obs_data.columns:
        #     print(obs_data[col])


        if make_altitude:
            print("Making Altitude Plot...")
            if night != "Any": used_night = night
            else: used_night = start_night
            ## Plot altitude graph
            altitude_path = "./data/AltitudeGraphs/AltitudePlot_" + ra + "_" + dec + "_" + used_night + "_" + location + ".png"

            if not os.path.exists(altitude_path):
                plot_altitude_graph(name, ra, dec, used_night, location)
            else:
                print("Altitude Plot already exists.")
        else:
            print("Not making Altitude Plot.")


        df_lists = df_to_lists(obs_data)

        context = {
                "df"    : df_lists,
                "nights" : nights,
        }


        # return render(request, "observations.html", context)
        return redirect('/observations/')

    else:
        print("No Request submitted.")
        return render(request, "observations.html", context)




## ========== Send Email Testing ==========

# @login_required
def email_test_page(request):

    print("\n\n ===== Updating Watched Targets... ===== \n")

    # QueryDict is immutable, and we want to append the remaining params to the redirect URL
    print("Fetching BlackGEM data:")

    df_watched_targets = pd.read_csv("./data/watched_targets.csv")
    num_list = df_watched_targets.num

    unique_num_list = df_watched_targets['num'].unique()


    for num in unique_num_list:
        try:
            Target_2 = Target.objects.get(id=num)
            Target_2a = TargetExtra.objects.get(Q(target_id=num) & Q(key = 'BlackGEM ID'))
            target_name = Target_2.name
            bgem_id = Target_2a.value

            saved_filename = "./data/" + target_name + "/none/" + target_name + "_BGEM_Data.csv"
            df_saved_lightcurve = pd.read_csv(saved_filename)
            df_saved_lightcurve_orig = df_saved_lightcurve
            df_bgem_lightcurve, df_limiting_mag = get_lightcurve_from_BGEM_ID(bgem_id)

            df_saved_lightcurve = df_saved_lightcurve[np.isfinite(df_saved_lightcurve.limit) == False]

            df_saved_lightcurve = df_saved_lightcurve.round(6)
            df_bgem_lightcurve  = df_bgem_lightcurve.round(6)

            df_new_detections = df_bgem_lightcurve[['i."mjd-obs"', 'x.mag_zogy', 'x.magerr_zogy', 'i.filter']]
            df_new_detections = df_new_detections.rename(columns={'i."mjd-obs"' : 'MJD', 'x.mag_zogy' : 'Mag', 'x.magerr_zogy' : 'Mag_Err', 'i.filter' : 'Filter'})
            df_new_detections_saved = df_new_detections
            df_new_detections = df_new_detections[~df_new_detections['MJD'].isin(df_saved_lightcurve.mjd)]

            av_mag_new_detection = np.mean(df_new_detections.Mag)
            if len(df_new_detections) > 1: plural_bool = True
            else: plural_bool = False

            df_to_alert = df_watched_targets[df_watched_targets.num == num]
            df_to_alert = df_to_alert[df_to_alert['limit'] >= av_mag_new_detection]

            email_list = list(df_to_alert[df_to_alert['num'] == num].person)

            # print("These:")
            # print(df_saved_lightcurve)
            # print(df_bgem_lightcurve)
            # print(df_new_detections)

            # if len(df_bgem_lightcurve) > len(df_saved_lightcurve):
            if len(df_new_detections) > 0:
                print("New datapoint!")
                print("\nEmailing the following users about " + target_name + ":")
                print(email_list)
                email_new_detection(bgem_id, target_name, df_new_detections, av_mag_new_detection, plural_bool, email_list)

                df_new_detections = df_new_detections.rename(columns={'MJD' : 'mjd', 'Mag' : 'mag', 'Mag_Err' : "magerr", 'Filter' : 'filter'})
                # df_new_detections = df_new_detections[['mjd', 'mag', 'Mag_Err' : "mag", 'Filter' : 'filter']]
                df_new_detections["limit"] = [np.nan]*len(df_new_detections)
                print("df_new_detections")
                print(df_new_detections)
                df_new_lightcurve = pd.concat([df_saved_lightcurve_orig, df_new_detections]).reset_index(drop=True)


                df_new_lightcurve.to_csv(saved_filename, index=False)
                #
                # print("Saved Lightcurve:")
                # print(df_saved_lightcurve)
                # print("Saved Limits:")
                # print(df_limiting_mag)
                # print("New BGEM Lightcurve:")
                # print(df_new_detections_saved)
            else:
                print("No new datapoint.")

        except Exception as e:
            print(e)
            print("There's been a problem. See above!")

    return render(request, "email_test.html")

def email_new_detection(bgem_id, target_name, df_new_detections, av_mag_new_detection, plural_bool, email_list):
    if plural_bool:
        subject_line = "New Detections of " + target_name + " (Av. %.1f mag)"%av_mag_new_detection
        line_1 = "There are new detections of one of your targets!\n\n"
    else:
        subject_line = "New Detection of " + target_name + " (%.1f mag)"%av_mag_new_detection
        line_1 = "There's a new detection of one of your targets!\n\n"
    send_mail(
        subject_line,
        line_1+
        "Target Name: \t" + target_name + "\n"+
        "BlackGEM ID: \t" + bgem_id + "\n\n"+
        "New detections:\n\n"+
        df_new_detections.to_string() + "\n\n"
        "See all detections here: https://gemtom.blackgem.org/transients/" + bgem_id + "\n\n",
        # "love your self <3",
        "john@blackgem.org",
        email_list,
        fail_silently=False,
    )

@login_required
def add_to_watchlist(request):

    print("\n\n\nFull Request:")
    print(request.GET)
    print("\n\n\n")

    target_id   = int(request.GET.get("gemtom_id"))
    user_email  = request.GET.get("user_email")
    limit       = request.GET.get("limit")
    subscribe       = request.GET.get("subscribe")

    Target_2 = Target.objects.get(id=target_id)
    target_name = Target_2.name
    Target_2a = TargetExtra.objects.get(Q(target_id=target_id) & Q(key = 'BlackGEM ID'))
    bgem_id = Target_2a.value

    if not os.path.exists("./data/" + target_name + "/none/"):
        os.makedirs("./data/" + target_name + "/none/")

        df_bgem_lightcurve, df_limiting_mag = get_lightcurve_from_BGEM_ID(bgem_id)

        df_bgem_lightcurve = df_bgem_lightcurve[['i."mjd-obs"', "x.mag_zogy", "x.magerr_zogy", "i.filter"]]
        df_limiting_mag = df_limiting_mag[['mjd','limiting_mag','filter']]

        df_bgem_lightcurve = df_bgem_lightcurve.rename(columns={
            'i."mjd-obs"'       : "mjd",
            'x.mag_zogy'        : "mag",
            'x.magerr_zogy'     : "magerr",
            'i.filter'          : "filter",
        })

        df_limiting_mag = df_limiting_mag.rename(columns={
            'limiting_mag'      : "limit",
        })

        df_full = pd.concat([df_bgem_lightcurve,df_limiting_mag]).reset_index(drop=True)
        df_full = df_full[['mjd','mag','magerr','limit','filter']]

        filepath = "./data/" + target_name + "/none/" + target_name + "_BGEM_Data.csv"
        df_full.to_csv(filepath)


    if not limit:   limit = 98.0
    else:           limit = float(limit)

    df_watchlist = pd.read_csv("./data/watched_targets.csv")

    df_watchlist = df_watchlist.loc[~((df_watchlist['num'] == target_id) & (df_watchlist['person'] == user_email))]


    if subscribe == "Yes":

        df_new_row = pd.DataFrame([[target_id, user_email, limit]], columns=["num", "person", "limit"])
        df_watchlist = pd.concat([df_watchlist, df_new_row])

    df_watchlist.to_csv("./data/watched_targets.csv", index=False)

    if subscribe == "Yes":
        if limit == 98.0:
            messages.success(
                request,
                'Watchlist updated! Any future detections of this source will be sent to ' + user_email + '.'
            )
        else:
            messages.success(
                request,
                'Watchlist updated! Any future detections of this source above ' + str(limit) + ' mag will be sent to ' + user_email + '.'
            )
    else:
        messages.success(
            request,
            'Watchlist updated! ' + user_email + ' has been unsubscribed from this target.'
        )


    return redirect('/targets/' + str(target_id))

@login_required
def send_email_test(request):
    '''
    Send an Email
    '''

    print("\n\n\nFull Request:")
    print(request.POST)
    print("\n\n\n")

    from django.core.mail import send_mail

    send_mail(
        "Test Email Please Read",
        "love your self <3",
        "john@blackgem.org",
        ["johnapaice@gmail.com"],
        fail_silently=False,
    )

    return redirect('/EmailTest/')
