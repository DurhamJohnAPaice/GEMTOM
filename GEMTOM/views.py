# from .forms import UploadFileForm
# from django.core.files.uploadedfile import SimpleUploadedFile

import os
import sys
import pandas as pd
from astropy.coordinates import SkyCoord, Galactocentric
from astropy import units as u
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
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

## For the Transients View
from django_plotly_dash import DjangoDash
import dash_ag_grid as dag

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

        print("\n\n\n\n\n\n\n")
        print("Barkbarkbark!")
        print("\n\n\n\n\n\n\n")

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
        for error in result['errors']:
            messages.warning(request, error)
        return redirect(reverse('tom_targets:list'))


def status_to_GEMTOM(request):
    '''
    Imports a target from the Status tab
    '''

    id = request.POST.get('id')
    name = request.POST.get('name')
    ra = request.POST.get('ra')
    dec = request.POST.get('dec')
    # print(ra, dec)

    get_lightcurve(id)

    gemtom_dataframe = pd.DataFrame({
        'name' : [name],
        'ra' : [ra],
        'dec' : [dec],
        'BlackGEM ID' : [int(id)],
        'type' : ['SIDEREAL'],
        'public' : ['Public']
    })

    gemtom_dataframe = gemtom_dataframe.reindex(gemtom_dataframe.index)

    # print(gemtom_dataframe)

    gemtom_dataframe.to_csv("./Data/processed_file.csv", index=False)
    csv_stream = StringIO(open(os.getcwd()+"/Data/processed_file.csv", "rb").read().decode('utf-8'), newline=None)

    ## And finally, read them in!
    result = import_targets(csv_stream)
    for target in result['targets']:
        target.give_user_access(request.user)
    messages.success(
        request,
        'Targets created: {}'.format(len(result['targets']))
    )
    for error in result['errors']:
        messages.warning(request, error)
    return redirect(reverse('tom_targets:list'))

def ID_to_GEMTOM(request):
    '''
    Imports a target from the BlackGEM ID
    '''

    transient_id = request.POST.get('id')

    print(transient_id)

    # import pandas as pd
    # from pathlib import Path
    # from blackpy import BlackGEM
    #
    # user_home = str(Path.home())
    # creds_user_file = user_home + "/.bg_follow_user_john_creds"
    # creds_db_file = user_home + "/.bg_follow_transientsdb_creds"
    # bg = BlackGEM(creds_user_file=creds_user_file, creds_db_file=creds_db_file)
    #
    # params = {'transient_id': transient_id,}
    #
    # qu = """\
    # SELECT iau_name
    #       ,ra_deg
    #       ,dec_deg
    #       ,id
    #   FROM runcat
    #  WHERE id = %(transient_id)s
    # """
    #
    # query = qu % (params)
    #
    # l_results = bg.run_query(query)
    # source_data = pd.DataFrame(l_results, columns=['name','ra','dec','id',])
    #
    # gemtom_dataframe_latter = pd.DataFrame({
    #     'type' : ['SIDEREAL'],
    #     'public' : ['Public']
    # })
    #
    # gemtom_dataframe = pd.concat([source_data, gemtom_dataframe_latter]).reset_index(drop=True)
    #
    # gemtom_dataframe.to_csv("./Data/processed_file.csv", index=False)
    # csv_stream = StringIO(open(os.getcwd()+"/Data/processed_file.csv", "rb").read().decode('utf-8'), newline=None)
    #
    # ## And finally, read them in!
    # result = import_targets(csv_stream)
    # for target in result['targets']:
    #     target.give_user_access(request.user)
    # messages.success(
    #     request,
    #     'Targets created: {}'.format(len(result['targets']))
    # )
    # for error in result['errors']:
    #     messages.warning(request, error)
    return redirect(reverse('tom_targets:list'))


class AboutView(TemplateView):
    template_name = 'about.html'


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



class BlackGEMView(TemplateView):
    template_name = 'blackGEM.html'

    def get_context_data(self, **kwargs):
        return {'targets': Target.objects.all()}




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
    Fetches BlackGEM's history and returns as several lists, in order to make a table
    '''

    # get_recent_blackgem_history()

    # history = pd.read_csv("./data/BlackGEM_History.csv")
    history = pd.read_csv("./data/Recent_BlackGEM_History.csv")
    # print(history)

    return history

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


class TransientsView(TemplateView):
    template_name = 'transients.html'

        # Initialize the Dash app
    app = DjangoDash('CSVDataApp')

    # Read CSV data
    # df = pd.read_csv('./data/BlackGEM_Transients_Last30Days.csv')
    df = pd.read_csv('./data/BlackGEM_Transients_Last30Days_Test.csv')

    ## Round values for displaying
    df['ra'] = round(df['ra'],5)
    df['dec'] = round(df['dec'],5)
    df['snr_zogy'] = round(df['snr_zogy'],2)
    df['q_max'] = round(df['q_max'],2)
    df['u_max'] = round(df['u_max'],2)
    df['i_max'] = round(df['i_max'],2)
    # df['lc_req'] = 'http://xmm-ssc.irap.omp.eu/claxson/lcrequest.php?runcatid=' + str(df.runcat_id)
    # df['lc_req'] = df.runcat_id.str[:]

    lc_req = ['http://xmm-ssc.irap.omp.eu/claxson/lcrequest.php?runcatid=' + str(i) for i in df['runcat_id']]
    lc_view = ['http://xmm-ssc.irap.omp.eu/claxson/BG_images/lcrequests/' + str(i) + '_lc.jpg' for i in df['runcat_id']]
    # lc_req = ['<a href="' + i + '">Request Lightcurve</a>' for i in lc_req]
    lc_req = ['[Request Lightcurve](' + i + ')' for i in lc_req]
    lc_view = ['[View Lightcurve](' + i + ')' for i in lc_view]
    df['lc_req'] = lc_req
    df['lc_view'] = lc_view

    # s       = pd.Series(['http://xmm-ssc.irap.omp.eu/claxson/lcrequest.php?runcatid='])
    # lc_req  = s.repeat(len(df))
    # lc_req  = lc_req.set_axis(range(len(df)))
    # df['lc_req'] = lc_req + df['runcat_id']



    # print(df.columns)

    # Define the layout of the Dash app
    app.layout = html.Div([
        dag.AgGrid(
            id='csv-grid',
            rowData=df.to_dict('records'),
            # columnDefs=[{'headerName': col, 'field': col} for col in df.columns[1:]],
            columnDefs=[
                {'headerName': 'BGEM ID', 'field': 'runcat_id'},
                # {'headerName': 'IAU Name', 'field': 'iauname'},
                {'headerName': 'RA', 'field': 'ra'},
                {'headerName': 'Dec', 'field': 'dec'},
                {'headerName': '#Datapoints', 'field': 'datapoints'},
                {'headerName': 'S/N', 'field': 'snr_zogy'},
                {'headerName': 'q', 'field': 'q_max', 'minWidth': 30, 'maxWidth': 60},
                {'headerName': 'u', 'field': 'u_max', 'minWidth': 30, 'maxWidth': 60},
                {'headerName': 'i', 'field': 'i_max', 'minWidth': 30, 'maxWidth': 60},
                {'headerName': 'Last Observation', 'field': 'last_obs', 'maxWidth': 110},
                {'headerName': 'Request LC', 'field': 'lc_req', "cellRenderer": "markdown", 'maxWidth': 130},
                {'headerName': 'View LC', 'field': 'lc_view', "cellRenderer": "markdown"},
                # {'headerName': 'G', 'field': 'GMag'},
            ],
            defaultColDef={
                'sortable': True,
                'filter': True,
                'resizable': True,
                'editable': True,
            },
            dangerously_allow_code=True,
            columnSize="autoSize",
            dashGridOptions = {"skipHeaderOnAutoSize": True},
            style={'height': '400px', 'width': '100%'},  # Set explicit height for the grid
            # style={'resize': 'both', 'overflow': 'hidden'},
            className='ag-theme-balham'  # Add a theme for better appearance
        )
    ], style={'height': '200px', 'width': '100%'}
    )
    # ], style={'width': '100%', 'height': False})  # Set the parent container's style

    # ], style={'width': '100%', 'height': 1200, 'display': 'flex'})  # Also adjust the parent container style
    # ], style={'resize': 'both', 'overflow': 'hidden'})
    # ], className='dash-container')  # Also adjust the parent container style

    def dash_view(request):
        return render(request, 'myapp/dash_template.html')




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

def handle_input(request):
    '''
    Redirects to a status page about a certain date.
    '''
    user_input = request.GET.get('user_input')
    print(user_input)
    if user_input:
        return redirect(f'/status/{user_input}')
    return redirect('status')  # Redirect to the original view if no input


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
    extragalactic_sources_jpg   = []

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
                        extragalactic_sources_jpg.append(get_lightcurve(file[2:10]))
                    else:
                        ## If it's not, state they're all unknown.
                        extragalactic_sources_name.append("Unknown")
                        extragalactic_sources_ra.append("(Unknown)")
                        extragalactic_sources_dec.append("(Unknown)")
                        extragalactic_sources_jpg.append("")

    ## Combine these together.
    extragalactic_sources = [extragalactic_sources_id, extragalactic_sources_name, extragalactic_sources_ra, extragalactic_sources_dec, extragalactic_sources_jpg]
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
    extragalactic_ids   = extragalactic_sources
    extragalactic_urls  = images_urls_sorted

    return num_new_transients, num_in_gaia, num_extragalactic, extragalactic_ids, extragalactic_urls, \
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
        "extragalactic_sources_jpg"     : extragalactic_sources[4],
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

    context['observed']                 = observed_string
    context['images_daily_text_1']      = images_daily_text_1
    context['transients_filename']      = transients_filename
    context['gaia_filename']            = gaia_filename
    context['extragalactic_filename']   = extragalactic_filename

    return render(request, "status/index.html", context)






## ======== RANDOM TESTING STUFF BELOW THIS LINE =========

    # if request.method == 'POST' and 'run_script' in request.POST:
    #
    #     def test_function():
    #         return "Bark!"
    #
    #     # call function
    #     test_function()
    #
    #     # return user to required page
    #     # return HttpResponseRedirect(reverse(template_name))


# def post(self, request, **kwargs):
#
#     if request.method == 'POST' and 'run_script' in request.POST:
#
#         # # import function to run
#         # from path_to_script import function_to_run
#
#         # call function
#         print("Bark!")
#
#         # return HttpResponse('')
#
#         data = {
#         "subject": "hello",
#         "message": "Hi there",
#         "sender": "foo@example.com",
#         "cc_myself": True,
#         }
#         file_data = {"mugshot": SimpleUploadedFile("face.jpg", b"file data")}
#         print(data)
#         # f = ContactFormWithMugshot(data, file_data)
#         # f = ContactFormWithMugshot(request.POST, request.FILE)
#
#         # # return user to required page
#         return HttpResponseRedirect(reverse('about'))

# def handle_uploaded_file(f):
#     with open("some/file/name.txt", "wb+") as destination:
#         for chunk in f.chunks():
#             destination.write(chunk)
#
# def upload_file(request):
#     if request.method == "POST":
#         form = UploadFileForm(request.POST, request.FILES)
#         if form.is_valid():
#             handle_uploaded_file(request.FILES["file"])
#             return HttpResponseRedirect("/success/url/")
#     else:
#         form = UploadFileForm()
#     return render(request, "upload.html", {"form": form})


    #
    # def simple_upload(request):
    #     if request.method == 'POST' and request.FILES['myfile']:
    #         myfile = request.FILES['myfile']
    #         fs = FileSystemStorage()
    #         filename = fs.save(myfile.name, myfile)
    #         uploaded_file_url = fs.url(filename)
    #         return render(request, 'core/simple_upload.html', {
    #             'uploaded_file_url': uploaded_file_url
    #         })
    #     return render(request, 'core/simple_upload.html')


# class UploadView(TemplateView):
#     template_name = 'simple_upload.html'


            #
            # data = {
            # "subject": "hello",
            # "message": "Hi there",
            # "sender": "foo@example.com",
            # "cc_myself": True,
            # }
            # file_data = {"mugshot": SimpleUploadedFile("face.jpg", b"file data")}
            # print(data)
            # print(file_data)
            # # f = ContactFormWithMugshot(data, file_data)
            # # f = ContactFormWithMugshot(request.POST, request.FILE)


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

class UpdateBlackGEMView(LoginRequiredMixin, RedirectView):
    """
    View that handles the updating of BlackGEM data. Requires authentication.
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
