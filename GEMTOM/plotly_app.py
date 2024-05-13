from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from django_plotly_dash import DjangoDash
from ztfquery import lightcurve


app = DjangoDash('SimpleExample')

#
# app.layout = html.Div([
#     html.H4('Test plot...'),
#     dcc.Graph(id="scatter-plot"),
#     html.P("Filter by petal width:"),
#     dcc.RangeSlider(
#         id='range-slider',
#         min=0, max=2.5, step=0.1,
#         marks={0: '0', 2.5: '2.5'},
#         value=[0.5, 2]
#     ),
# ])
#
# ## ----- ZTF -----
#
# source_ra   = 306.01249999999993
# source_dec  = 33.867222222222225
#
# print("-- ZTF: Looking for target...", end="\r")
# lcq = lightcurve.LCQuery.from_position(source_ra, source_dec, 5)
# ZTF_data = pd.DataFrame({'JD' : lcq.data.mjd+2400000.5, 'Magnitude' : lcq.data.mag, 'Magnitude_Error' : lcq.data.magerr})
#
# if len(ZTF_data) == 0:
#     raise Exception("-- ZTF: Target not found. Try AAVSO instead?")
#
# print("-- ZTF: Looking for target... target found.")
#
# @app.callback(
#     Output("scatter-plot", "figure"),
#     Input("range-slider", "value")
# )
# def update_bar_chart(slider_range):
#     df = ZTF_data # replace with your own data source
#     low, high = slider_range
#     # mask = (df['petal_width'] > low) & (df['petal_width'] < high)
#     fig = px.scatter(df, x='JD', y='Magnitude')
#     return fig


## ----- ZTF -----

# source_ra   = 306.01249999999993
# source_dec  = 33.867222222222225
#
# print("-- ZTF: Looking for target...", end="\r")
# lcq = lightcurve.LCQuery.from_position(source_ra, source_dec, 5)
# ZTF_data = pd.DataFrame({'JD' : lcq.data.mjd+2400000.5, 'Magnitude' : lcq.data.mag, 'Magnitude_Error' : lcq.data.magerr})
#
# if len(ZTF_data) == 0:
#     raise Exception("-- ZTF: Target not found. Try AAVSO instead?")
#
# print("-- ZTF: Looking for target... target found.")
#
#
# df = ZTF_data # replace with your own data source
#
# fig = px.scatter(df, x='JD', y='Magnitude')
#
# app.layout = html.Div([
#     html.H4('Test plot...'),
#     dcc.Graph(figure=fig),
# ])
#
# @app.callback(
#     Output("scatter-plot", "figure"),
# )
# def update_bar_chart():
#     return fig



# app.run_server(debug=False)
