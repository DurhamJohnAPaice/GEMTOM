import plotly.express as px
import pandas as pd
from jinja2 import Template
import os

# print("-------------------BAAAAAAAAAAAAAAAAARK!-------------------")
# print(os.getcwd())

# test_data = pd.read_csv("./GEMTOM/BlackGEM_Test_Data.csv")
#
# # data_canada = px.data.gapminder().query("country == 'Canada'")
# fig = px.scatter(test_data, x='i."mjd-obs"', y='x.mag_zogy')
#
# output_html_path=r"./templates/about2.html"
# input_template_path = r"./templates/test_about.html"
#
# plotly_jinja_data = {"fig":fig.to_html(full_html=True)}
# #consider also defining the include_plotlyjs parameter to point to an external Plotly.js as described above
#
# with open(output_html_path, "w", encoding="utf-8") as output_file:
#     with open(input_template_path) as template_file:
#         j2_template = Template(template_file.read())
#         output_file.write(j2_template.render(plotly_jinja_data))
