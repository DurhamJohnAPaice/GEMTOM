import logging
# from urllib.parse import urlencode

from django import template
from django.conf import settings
from plotly import offline
import plotly.graph_objs as go
import numpy as np
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

# @register.inclusion_tag('tom_dataproducts/partials/ztf_for_target.html', takes_context=True)
# def ztf_for_target(context, target, width=700, height=600, background=None, label_color=None, grid=True):
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
#         photometry_data_type = settings.DATA_PRODUCT_TYPES['ztf_data'][0]
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

@register.inclusion_tag('tom_dataproducts/partials/observe_staralt.html')
def observe_staralt(target):
    """
    Includes a link to staralt
    """
    target_name = target.name
    print("\n\n\n\n\n")
    # print(target.ra)
    # print(target.dec)
    print(target_name[:4])
    print("\n\n\n\n\n")
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

    color_map = {
        'u': 'blue',
        'g': 'green',
        'r': 'red',
        'i': 'black',
        'z': 'purple'
    }

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
