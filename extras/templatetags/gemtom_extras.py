import logging
# from urllib.parse import urlencode

from django import template
from django.conf import settings
from plotly import offline
import plotly.graph_objs as go
import numpy as np
from django import forms
import pandas as pd
from GEMTOM import views
# from ...GEMTOM.views import plot_BGEM_lightcurve
# from ...forms import *
# from django import forms
# from django.contrib.auth.models import Group
# from django.core.paginator import Paginator
# from django.shortcuts import reverse
# from django.utils import timezone
# from datetime import datetime, timedelta
# from guardian.shortcuts import get_objects_for_user
# from io import BytesIO
# from PIL import Image, ImageDraw
# import base64

from tom_dataproducts.models import ReducedDatum
# from tom_dataproducts.forms import DataProductUploadForm, DataShareForm
# from tom_dataproducts.models import DataProduct, ReducedDatum
# from tom_dataproducts.processors.data_serializers import SpectrumSerializer
# from tom_dataproducts.forced_photometry.forced_photometry_service import get_service_classes
# from tom_observations.models import ObservationRecord
from tom_targets.models import Target

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

register = template.Library()

@register.inclusion_tag('tom_dataproducts/partials/ztf_for_target.html', takes_context=True)
def ztf_for_target(context, target, width=700, height=600, background=None, label_color=None, grid=True):
    """
    Renders a photometric plot for a target.

    This templatetag requires all ``ReducedDatum`` objects with a data_type of ``photometry`` to be structured with the
    following keys in the JSON representation: magnitude, error, filter

    :param width: Width of generated plot
    :type width: int

    :param height: Height of generated plot
    :type width: int

    :param background: Color of the background of generated plot. Can be rgba or hex string.
    :type background: str

    :param label_color: Color of labels/tick labels. Can be rgba or hex string.
    :type label_color: str

    :param grid: Whether to show grid lines.
    :type grid: bool
    """

    color_map = {
        'r': 'red',
        'g': 'green',
        'i': 'black'
    }

    try:
        photometry_data_type = settings.DATA_PRODUCT_TYPES['ztf_data'][0]
    except (AttributeError, KeyError):
        photometry_data_type = 'ztf_data'
    photometry_data = {}
    if settings.TARGET_PERMISSIONS_ONLY:
        datums = ReducedDatum.objects.filter(target=target, data_type=photometry_data_type)
    else:
        datums = get_objects_for_user(context['request'].user,
                                      'tom_dataproducts.view_reduceddatum',
                                      klass=ReducedDatum.objects.filter(
                                        target=target,
                                        data_type=photometry_data_type))
    plot_data = []
    all_ydata = []

    if len(datums) > 0:
        for datum in datums:
            photometry_data.setdefault('jd', []).append(datum.timestamp)
            photometry_data.setdefault('magnitude', []).append(datum.value.get('magnitude'))
            photometry_data.setdefault('error', []).append(datum.value.get('error'))



        # print(photometry_data['jd'])
        # print(photometry_data['magnitude'])
        # print(photometry_data['error'])
        # print(photometry_data['error'][0] != None)

        photometry_data['magnitude'] = [float(i) for i in photometry_data['magnitude']]
        for i in range(0, len(photometry_data['error'])):
            if photometry_data['error'][i] == None:
                photometry_data['error'][i] = 0
            else:
                photometry_data['error'][i] = float(photometry_data['error'][i])
        # photometry_data['error'] = [float(i) for i in photometry_data['error']]

        # print("\n\n\n\n\n\n\n")
        # print(photometry_data['jd'])
        # print(photometry_data['magnitude'])
        # print(photometry_data['error'])

        series = go.Scatter(
            x=photometry_data['jd'],
            y=photometry_data['magnitude'],
            mode='markers',
            error_y=dict(
                type='data',
                array=photometry_data['error'],
                visible=True
            )
        )
        plot_data.append(series)
        mags = np.array(photometry_data['magnitude'], float)  # converts None --> nan (as well as any strings)
        errs = np.array(photometry_data['error'], float)
        errs[np.isnan(errs)] = 0.  # missing errors treated as zero
        all_ydata.append(mags + errs)
        all_ydata.append(mags - errs)

    # scale the y-axis manually so that we know the range ahead of time and can scale the secondary y-axis to match
    if all_ydata:
        all_ydata = np.concatenate(all_ydata)
        ymin = np.nanmin(all_ydata)
        ymax = np.nanmax(all_ydata)
        yrange = ymax - ymin
        ymin_view = ymin - 0.05 * yrange
        ymax_view = ymax + 0.05 * yrange
    else:
        ymin_view = 0.
        ymax_view = 0.
    yaxis = {
        'title': 'Apparent Magnitude',
        'range': (ymax_view, ymin_view),
        'showgrid': grid,
        'color': label_color,
        'showline': True,
        'linecolor': label_color,
        'mirror': True,
        'zeroline': False,
    }
    if target.distance is not None:
        dm = 5. * (np.log10(target.distance) - 1.)  # assumes target.distance is in parsecs
        yaxis2 = {
            'title': 'Absolute Magnitude',
            'range': (ymax_view - dm, ymin_view - dm),
            'showgrid': False,
            'overlaying': 'y',
            'side': 'right',
            'zeroline': False,
        }
        plot_data.append(go.Scatter(x=[], y=[], yaxis='y2'))  # dummy data set for abs mag axis
    else:
        yaxis2 = None

    layout = go.Layout(
        xaxis={
            'showgrid': grid,
            'color': label_color,
            'showline': True,
            'linecolor': label_color,
            'mirror': True,
        },
        yaxis=yaxis,
        yaxis2=yaxis2,
        height=height,
        width=width,
        paper_bgcolor=background,
        plot_bgcolor=background,
        legend={
            'font_color': label_color,
            'xanchor': 'center',
            'yanchor': 'bottom',
            'x': 0.5,
            'y': 1.,
            'orientation': 'h',
        },
        clickmode='event+select',
    )
    fig = go.Figure(data=plot_data, layout=layout)

    return {
        'target': target,
        'plot': offline.plot(fig, output_type='div', show_link=False),
    }

class ClassificationForm(forms.Form):
    """

    """
    CHOICES = [
        ("Other",                "Other"),
        ("SN",                   "SN"),
        ("SN I",                 "SN I"),
        ("SN Ia",                "SN Ia"),
        ("SN Ib",                "SN Ib"),
        ("SN Ic",                "SN Ic"),
        ("SN Ib/c",              "SN Ib/c"),
        ("SN Ic-BL",             "SN Ic-BL"),
        ("SN Ibn",               "SN Ibn"),
        ("SN II",                "SN II"),
        ("SN IIP",               "SN IIP"),
        ("SN IIL",               "SN IIL"),
        ("SN IIn",               "SN IIn"),
        ("SN IIb",               "SN IIb"),
        ("SN I-faint",           "SN I-faint"),
        ("SN I-rapid",           "SN I-rapid"),
        ("SLSN-I",               "SLSN-I"),
        ("SLSN-II",              "SLSN-II"),
        ("SLSN-R",               "SLSN-R"),
        ("Afterglow",            "Afterglow"),
        ("LBV",                  "LBV"),
        ("ILRT",                 "ILRT"),
        ("Nova",                 "Nova"),
        ("CV",                   "CV"),
        ("Varstar",              "Varstar"),
        ("AGN",                  "AGN"),
        ("Galaxy",               "Galaxy"),
        ("QSO",                  "QSO"),
        ("Blazar",               "Blazar"),
        ("Light-Echo",           "Light-Echo"),
        ("Std-spec",             "Std-spec"),
        ("Gap",                  "Gap"),
        ("Gap I",                "Gap I"),
        ("Gap II",               "Gap II"),
        ("LRN",                  "LRN"),
        ("FBOT",                 "FBOT"),
        ("Kilonova",             "Kilonova"),
        ("Imposter-SN",          "Imposter-SN"),
        ("SN Ia-pec",            "SN Ia-pec"),
        ("SN Ia-SC",             "SN Ia-SC"),
        ("SN Ia-91bg-like",      "SN Ia-91bg-like"),
        ("SN Ia-91T-like",       "SN Ia-91T-like"),
        ("SN Iax[02cx-like]",    "SN Iax[02cx-like]"),
        ("SN Ia-CSM",            "SN Ia-CSM"),
        ("SN Ib-pec",            "SN Ib-pec"),
        ("SN Ic-pec",            "SN Ic-pec"),
        ("SN Icn",               "SN Icn"),
        ("SN Ibn/Icn",           "SN Ibn/Icn"),
        ("SN II-pec",            "SN II-pec"),
        ("SN IIn-pec",           "SN IIn-pec"),
        ("SN Ib-Ca-rich",        "SN Ib-Ca-rich"),
        ("SN Ib/c-Ca-rich",      "SN Ib/c-Ca-rich"),
        ("SN Ic-Ca-rich",        "SN Ic-Ca-rich"),
        ("SN Ia-Ca-rich",        "SN Ia-Ca-rich"),
        ("TDE",                  "TDE"),
        ("TDE-H",                "TDE-H"),
        ("TDE-He",               "TDE-He"),
        ("TDE-H-He",             "TDE-H-He"),
        ("FRB",                  "FRB"),
        ("WR",                   "WR"),
        ("WR-WN",                "WR-WN"),
        ("WR-WC",                "WR-WC"),
        ("WR-WO",                "WR-WO"),
        ("M dwarf",              "M dwarf"),
        ("NA/Unknown",           "NA/Unknown"),
        ("Computed-Ia",          "Computed-Ia"),
        ("Computed-IIP",         "Computed-IIP"),
        ("Computed-IIb",         "Computed-IIb"),
        ("Computed-PISN",        "Computed-PISN"),
        ("Computed-IIn",         "Computed-IIn"),
    ]
    dropdown = forms.ChoiceField(
        choices=CHOICES,
        label="",
        widget=forms.Select(attrs={'class': 'custom-dropdown'})
    )

@register.inclusion_tag('tom_dataproducts/partials/update_classification.html')
def update_classification(target):
    """
    Handles updating a classification
    """
    target_name = target.name
    target_id = target.id

    form = ClassificationForm()

    return {
        'target_name' : target_name,
        'target_id'   : target_id,
        'form'        : form
        }

@register.inclusion_tag('tom_dataproducts/partials/other_pages.html')
def other_pages(target):
    """
    Links to GEMTOM and BlackView pages
    """
    target_name = target.name
    target_id = target.id
    # print(target.__dict__)
    # bgem_id = target|target_extra_field:"bgem_id"
    bgem_id = target.targetextra_set.get(key='BlackGEM ID').value
    print(bgem_id)

    return {
        'target_name' : target_name,
        'target_id'   : target_id,
        'bgem_id'     : bgem_id,
        }

@register.inclusion_tag('tom_dataproducts/partials/query_forced_photometry.html')
def query_forced_photometry(target):
    target_name = target.name
    target_id = target.id
    return {
        'target_name' : target_name,
        'target_id'   : target_id,
    }

@register.inclusion_tag('tom_dataproducts/partials/observe_staralt.html')
def observe_staralt(target):
    """
    Includes a link to staralt
    """
    target_name = target.name
    # print("\n\n\n\n\n")
    # print(target.ra)
    # print(target.dec)
    # print(target_name[:4])
    # print("\n\n\n\n\n")
    if target_name[:4] == 'BGEM': target_name = target.name[5:12]
    else:
        target_name = target_name.replace(" ","")
        target_name = target_name[:9]
    print(target_name)
    return {
        'target_name' : target_name,
        'staralt_ra'  : round(target.ra, 3),
        'staralt_dec' : round(target.dec, 3)
        }
    # return target

## Testing for updating BGEM LC:

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
        photometry = BGEM_to_GEMTOM_photometry_2(df_bgem_lightcurve)
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



@register.inclusion_tag('tom_dataproducts/partials/update_blackgem_data.html', takes_context=True)
def update_blackgem_data(context, target, width=700, height=400, background=None, label_color=None, grid=True):
    print(target.__dict__)
    print(context.__dict__)
    target_name = target.name
    target_id = target.id
    # target_blackgemid = target.id

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


# @register.inclusion_tag('tom_dataproducts/partials/blackgem_for_target.html', takes_context=True)
# def blackgem_for_target(context, target, width=700, height=400, background=None, label_color=None, grid=True):
#     """
#     Renders a photometric plot for a target.
#
#     This templatetag requires all ``ReducedDatum`` objects with a data_type of ``photometry`` to be structured with the
#     following keys in the JSON representation: magnitude, error, filter
#
#     :param width: Width of generated plot
#     :type width: int
#
#     :param height: Height of generated plot
#     :type width: int
#
#     :param background: Color of the background of generated plot. Can be rgba or hex string.
#     :type background: str
#
#     :param label_color: Color of labels/tick labels. Can be rgba or hex string.
#     :type label_color: str
#
#     :param grid: Whether to show grid lines.
#     :type grid: bool
#     """
#
#     color_map = {
#         'u': 'darkviolet',
#         'g': 'forestgreen',
#         'q': 'darkorange',
#         'r': 'orangered',
#         'i': 'crimson',
#         'z': 'dimgrey'
#     }
#
#     try:
#         photometry_data_type = settings.DATA_PRODUCT_TYPES['blackgem_data'][0]
#     except (AttributeError, KeyError):
#         photometry_data_type = 'blackgem_data'
#     photometry_data = {}
#     if settings.TARGET_PERMISSIONS_ONLY:
#         datums = ReducedDatum.objects.filter(target=target, data_type=photometry_data_type)
#     else:
#         datums = get_objects_for_user(context['request'].user,
#                                       'tom_dataproducts.view_reduceddatum',
#                                       klass=ReducedDatum.objects.filter(
#                                         target=target,
#                                         data_type=photometry_data_type))
#     plot_data = []
#     all_ydata = []
#
#     if len(datums) > 0:
#         for datum in datums:
#             photometry_data.setdefault('jd', []).append(datum.timestamp)
#             photometry_data.setdefault('magnitude', []).append(datum.value.get('magnitude'))
#             photometry_data.setdefault('error', []).append(datum.value.get('error'))
#             photometry_data.setdefault('filter', []).append(datum.value.get('filter'))
#
#             # print(photometry_data)
#         # print("Bark!")
#
#
#         photometry_data['magnitude'] = [float(i) for i in photometry_data['magnitude']]
#         for i in range(0, len(photometry_data['error'])):
#             if photometry_data['error'][i] == None:
#                 photometry_data['error'][i] = 0
#             else:
#                 photometry_data['error'][i] = float(photometry_data['error'][i])
#         # photometry_data['error'] = [float(i) for i in photometry_data['error']]
#
#         # print("\n\n\n\n\n\n\n")
#         # print(photometry_data['jd'])
#         # print(photometry_data['magnitude'])
#         # print(photometry_data['error'])
#         # print(photometry_data['filter'])
#
#         print(photometry_data)
#         df = pd.DataFrame.from_dict(photometry_data)
#         print(df)
#
#         ## Seperate out the data based on filter
#         q_indices = [i for i, f in enumerate(photometry_data['filter']) if f == 'q']
#         u_indices = [i for i, f in enumerate(photometry_data['filter']) if f == 'u']
#         g_indices = [i for i, f in enumerate(photometry_data['filter']) if f == 'g']
#         r_indices = [i for i, f in enumerate(photometry_data['filter']) if f == 'r']
#         i_indices = [i for i, f in enumerate(photometry_data['filter']) if f == 'i']
#         z_indices = [i for i, f in enumerate(photometry_data['filter']) if f == 'z']
#
#         filtered_data = [
#             {key: [value[i] for i in q_indices] for key, value in photometry_data.items()},
#             {key: [value[i] for i in u_indices] for key, value in photometry_data.items()},
#             {key: [value[i] for i in g_indices] for key, value in photometry_data.items()},
#             {key: [value[i] for i in r_indices] for key, value in photometry_data.items()},
#             {key: [value[i] for i in i_indices] for key, value in photometry_data.items()},
#             {key: [value[i] for i in z_indices] for key, value in photometry_data.items()},
#         ]
#
#         # print(filtered_data)
#         # print("Bark!")
#
#         for filter in filtered_data:
#             print(len(filter['jd']))
#             if len(filter['jd']) > 0:
#                 series = go.Scatter(
#                     x=filter['jd'],
#                     y=filter['magnitude'],
#                     mode='markers',
#                     name=filter['filter'][0],
#                     marker_color=color_map[filter['filter'][0]],
#                     error_y=dict(
#                         type='data',
#                         array=filter['error'],
#                         visible=True
#                     )
#                 )
#                 plot_data.append(series)
#                 mags = np.array(filter['magnitude'], float)  # converts None --> nan (as well as any strings)
#                 errs = np.array(filter['error'], float)
#                 errs[np.isnan(errs)] = 0.  # missing errors treated as zero
#                 all_ydata.append(mags + errs)
#                 all_ydata.append(mags - errs)
#
#     # scale the y-axis manually so that we know the range ahead of time and can scale the secondary y-axis to match
#     if all_ydata:
#         all_ydata = np.concatenate(all_ydata)
#         ymin = np.nanmin(all_ydata)
#         ymax = np.nanmax(all_ydata)
#         yrange = ymax - ymin
#         ymin_view = ymin - 0.05 * yrange
#         ymax_view = ymax + 0.05 * yrange
#     else:
#         ymin_view = 0.
#         ymax_view = 0.
#     yaxis = {
#         'title': 'Apparent Magnitude',
#         'range': (ymax_view, ymin_view),
#         'showgrid': grid,
#         'color': label_color,
#         'showline': True,
#         'linecolor': label_color,
#         'mirror': True,
#         'zeroline': False,
#     }
#     if target.distance is not None:
#         dm = 5. * (np.log10(target.distance) - 1.)  # assumes target.distance is in parsecs
#         yaxis2 = {
#             'title': 'Absolute Magnitude',
#             'range': (ymax_view - dm, ymin_view - dm),
#             'showgrid': False,
#             'overlaying': 'y',
#             'side': 'right',
#             'zeroline': False,
#         }
#         plot_data.append(go.Scatter(x=[], y=[], yaxis='y2'))  # dummy data set for abs mag axis
#     else:
#         yaxis2 = None
#
#     layout = go.Layout(
#         xaxis={
#             'showgrid': grid,
#             'color': label_color,
#             'showline': True,
#             'linecolor': label_color,
#             'mirror': True,
#         },
#         yaxis=yaxis,
#         yaxis2=yaxis2,
#         height=height,
#         width=width,
#         paper_bgcolor=background,
#         plot_bgcolor=background,
#         legend={
#             'font_color': label_color,
#             'xanchor': 'center',
#             'yanchor': 'bottom',
#             'x': 0.5,
#             'y': 1.,
#             'orientation': 'h',
#         },
#         clickmode='event+select',
#     )
#     fig = go.Figure(data=plot_data, layout=layout)
#
#     return {
#         'target': target,
#         'plot': offline.plot(fig, output_type='div', show_link=False),
#     }


@register.inclusion_tag('tom_dataproducts/partials/blackgem_for_target.html', takes_context=True)
def blackgem_for_target(context, target, width=700, height=400, background=None, label_color=None, grid=True):
    """
    Renders a photometric plot for a target.

    This templatetag requires all ``ReducedDatum`` objects with a data_type of ``photometry`` to be structured with the
    following keys in the JSON representation: magnitude, error, filter

    :param width: Width of generated plot
    :type width: int

    :param height: Height of generated plot
    :type width: int

    :param background: Color of the background of generated plot. Can be rgba or hex string.
    :type background: str

    :param label_color: Color of labels/tick labels. Can be rgba or hex string.
    :type label_color: str

    :param grid: Whether to show grid lines.
    :type grid: bool
    """

    # views.test_print()

    color_map = {
        'u': 'darkviolet',
        'g': 'forestgreen',
        'q': 'darkorange',
        'r': 'orangered',
        'i': 'crimson',
        'z': 'dimgrey'
    }

    ## First, read in the photometry daata from the context and target variables.
    try:
        photometry_data_type = settings.DATA_PRODUCT_TYPES['blackgem_data'][0]
    except (AttributeError, KeyError):
        photometry_data_type = 'blackgem_data'
    photometry_data = {}
    if settings.TARGET_PERMISSIONS_ONLY:
        datums = ReducedDatum.objects.filter(target=target, data_type=photometry_data_type)
    else:
        datums = get_objects_for_user(context['request'].user,
                                      'tom_dataproducts.view_reduceddatum',
                                      klass=ReducedDatum.objects.filter(
                                        target=target,
                                        data_type=photometry_data_type))

    ## If there's data, start the plot!
    if len(datums) > 0:

        ## For each datum, grab the right bit of the data.
        for datum in datums:
            photometry_data.setdefault('jd', []).append(datum.timestamp)
            photometry_data.setdefault('magnitude', []).append(datum.value.get('magnitude'))
            photometry_data.setdefault('limit', []).append(datum.value.get('limit'))
            photometry_data.setdefault('filter', []).append(datum.value.get('filter'))

            ## Deal with datums with no errors
            if datum.value.get('error') == None:
                photometry_data.setdefault('error', []).append(0)
            else:
                photometry_data.setdefault('error', []).append(float(datum.value.get('error')))

        # print(photometry_data)

        ## Get ready for plotting!
        ## First, split the data up into detections and limiting magnitudes.
        df = pd.DataFrame.from_dict(photometry_data)
        df_lightcurve = df.loc[df['magnitude'].notnull()]
        df_limiting_mag = df.loc[df['magnitude'].isnull()]

        # print(df_lightcurve)
        # print(df_limiting_mag)

        ## Rename for compatibility
        df_lightcurve = df_lightcurve.rename(columns={
            'filter'    : 'i.filter',
            'magnitude' : 'x.mag_zogy',
            'error'     : 'x.magerr_zogy',
        })

        ## Create (empty) flux columns
        df_lightcurve["x.flux_zogy"] = ['']*len(df_lightcurve)
        df_lightcurve["x.fluxerr_zogy"] = ['']*len(df_lightcurve)


        ## Create (empty) flux columns
        df_limiting_mag = df_limiting_mag.rename(columns={
            'limit' : 'limiting_mag',
        })

        ## Convert Datetime to MJD
        df_lightcurve['i."mjd-obs"']    = pd.DatetimeIndex(df_lightcurve['jd']).to_julian_date() - 2400000.5
        df_limiting_mag['mjd']          = pd.DatetimeIndex(df_limiting_mag['jd']).to_julian_date() - 2400000.5
        print(df_lightcurve)

        ## Plot using my usual graph function!
        fig = views.plot_BGEM_lightcurve(df_lightcurve, df_limiting_mag)

    ## If there's no data, plot an empty graph.
    else:
        fig = go.Figure(data=[])


    return {
        'target': target,
        'plot': offline.plot(fig, output_type='div', show_link=False),
    }

# @register.inclusion_tag('tom_dataproducts/partials/photometry_for_target.html', takes_context=True)
# def photometry_for_target(context, target, width=700, height=600, background=None, label_color=None, grid=True):
#     """
#     Renders a photometric plot for a target.
#
#     This templatetag requires all ``ReducedDatum`` objects with a data_type of ``photometry`` to be structured with the
#     following keys in the JSON representation: magnitude, error, filter
#
#     :param width: Width of generated plot
#     :type width: int
#
#     :param height: Height of generated plot
#     :type width: int
#
#     :param background: Color of the background of generated plot. Can be rgba or hex string.
#     :type background: str
#
#     :param label_color: Color of labels/tick labels. Can be rgba or hex string.
#     :type label_color: str
#
#     :param grid: Whether to show grid lines.
#     :type grid: bool
#     """
#
#     color_map = {
#         'r': 'red',
#         'g': 'green',
#         'i': 'black'
#     }
#
#     try:
#         photometry_data_type = settings.DATA_PRODUCT_TYPES['photometry'][0]
#     except (AttributeError, KeyError):
#         photometry_data_type = 'photometry'
#     photometry_data = {}
#     if settings.TARGET_PERMISSIONS_ONLY:
#         datums = ReducedDatum.objects.filter(target=target, data_type=photometry_data_type)
#     else:
#         datums = get_objects_for_user(context['request'].user,
#                                       'tom_dataproducts.view_reduceddatum',
#                                       klass=ReducedDatum.objects.filter(
#                                         target=target,
#                                         data_type=photometry_data_type))
#
#     for datum in datums:
#         photometry_data.setdefault(datum.value['filter'], {})
#         photometry_data[datum.value['filter']].setdefault('time', []).append(datum.timestamp)
#         photometry_data[datum.value['filter']].setdefault('magnitude', []).append(datum.value.get('magnitude'))
#         photometry_data[datum.value['filter']].setdefault('error', []).append(datum.value.get('error'))
#         photometry_data[datum.value['filter']].setdefault('limit', []).append(datum.value.get('limit'))
#
#     plot_data = []
#     all_ydata = []
#     for filter_name, filter_values in photometry_data.items():
#         if filter_values['magnitude']:
#             # print(filter_values['magnitude'])
#             series = go.Scatter(
#                 x=filter_values['time'],
#                 y=filter_values['magnitude'],
#                 mode='markers',
#                 marker=dict(color=color_map.get(filter_name)),
#                 name=filter_name,
#                 error_y=dict(
#                     type='data',
#                     array=filter_values['error'],
#                     visible=True
#                 )
#             )
#             plot_data.append(series)
#             mags = np.array(filter_values['magnitude'], float)  # converts None --> nan (as well as any strings)
#             errs = np.array(filter_values['error'], float)
#             errs[np.isnan(errs)] = 0.  # missing errors treated as zero
#             all_ydata.append(mags + errs)
#             all_ydata.append(mags - errs)
#         if filter_values['limit']:
#             # print(filter_values['limit'])
#             series = go.Scatter(
#                 x=filter_values['time'],
#                 y=filter_values['limit'],
#                 mode='markers',
#                 opacity=0.5,
#                 marker=dict(color=color_map.get(filter_name)),
#                 marker_symbol=6,  # upside down triangle
#                 name=filter_name + ' non-detection',
#             )
#             plot_data.append(series)
#             all_ydata.append(np.array(filter_values['limit'], float))
#
#     # scale the y-axis manually so that we know the range ahead of time and can scale the secondary y-axis to match
#     if all_ydata:
#         all_ydata = np.concatenate(all_ydata)
#         ymin = np.nanmin(all_ydata)
#         ymax = np.nanmax(all_ydata)
#         yrange = ymax - ymin
#         ymin_view = ymin - 0.05 * yrange
#         ymax_view = ymax + 0.05 * yrange
#     else:
#         ymin_view = 0.
#         ymax_view = 0.
#     yaxis = {
#         'title': 'Apparent Magnitude',
#         'range': (ymax_view, ymin_view),
#         'showgrid': grid,
#         'color': label_color,
#         'showline': True,
#         'linecolor': label_color,
#         'mirror': True,
#         'zeroline': False,
#     }
#     if target.distance is not None:
#         dm = 5. * (np.log10(target.distance) - 1.)  # assumes target.distance is in parsecs
#         yaxis2 = {
#             'title': 'Absolute Magnitude',
#             'range': (ymax_view - dm, ymin_view - dm),
#             'showgrid': False,
#             'overlaying': 'y',
#             'side': 'right',
#             'zeroline': False,
#         }
#         plot_data.append(go.Scatter(x=[], y=[], yaxis='y2'))  # dummy data set for abs mag axis
#     else:
#         yaxis2 = None
#
#     layout = go.Layout(
#         xaxis={
#             'showgrid': grid,
#             'color': label_color,
#             'showline': True,
#             'linecolor': label_color,
#             'mirror': True,
#         },
#         yaxis=yaxis,
#         yaxis2=yaxis2,
#         height=height,
#         width=width,
#         paper_bgcolor=background,
#         plot_bgcolor=background,
#         legend={
#             'font_color': label_color,
#             'xanchor': 'center',
#             'yanchor': 'bottom',
#             'x': 0.5,
#             'y': 1.,
#             'orientation': 'h',
#         },
#         clickmode='event+select',
#     )
#     fig = go.Figure(data=plot_data, layout=layout)
#
#     return {
#         'target': target,
#         'plot': offline.plot(fig, output_type='div', show_link=False),
#     }
#
