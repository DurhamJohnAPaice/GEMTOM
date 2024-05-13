from django.views.generic import TemplateView
from tom_observations.models import Target
from django.http import HttpResponse, HttpResponseRedirect, FileResponse
from django.urls import reverse
from django.shortcuts import render
# from .forms import UploadFileForm
# from django.core.files.uploadedfile import SimpleUploadedFile
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os
import sys
import pandas as pd
from astropy.coordinates import SkyCoord, Galactocentric
from astropy import units as u
from .BlackGEM_to_GEMTOM import *
from . import plotly_test
from . import plotly_app
from ztfquery import lightcurve    ## <≈- Uncomment if using ZTF
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

class AboutView(TemplateView):
    template_name = 'about.html'


    def get_context_data(self, **kwargs):
        return {'targets': Target.objects.all()}

    # def post(self, request, **kwargs):
    #     ra  = float(request.POST['num1'])
    #     dec = float(request.POST['num2'])
    #     ra  *= 2
    #     dec *= 3
    #     # return HttpResponse('entered text:' + request.POST['text'])
    #     return HttpResponse('Chosen RA x2: ' + str(ra) + '  |  Chosen Dec x3: ' + str(dec))

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


        # return HttpResponse('entered text:' + request.POST['text'])
        # return HttpResponse(ZTF_data)
        # return HttpResponse(ZTF_data.to_html())

    # def post(self, request, **kwargs):
    #
    #     if request.method == 'POST' and 'run_script' in request.POST and request.FILES['file']:
    #
    #         print("\n\n\n\n\n\n\n\n")
    #
    #         print(sys.version)
    #
    #         myfile = request.FILES['file']
    #         print(myfile.__dict__)
    #         # print(myfile.file)
    #
    #         fs = FileSystemStorage()
    #         filename = fs.save(myfile.name, myfile)
    #         uploaded_file_url = fs.url(filename)
    #         # print(os.getcwd()+uploaded_file_url)
    #
    #         processed_file = GEM_to_TOM(os.getcwd()+uploaded_file_url)
    #         # processed_file = GEM_to_TOM(myfile.file)
    #         processed_file.to_csv("./Data/processed_file.csv", index=False)
    #         os.remove(os.getcwd()+uploaded_file_url)
    #
    #
    #         # text = text.iloc[1]
    #
    #         # f = open(os.getcwd()+uploaded_file_url, "r")
    #         # # print(f.read())
    #         # text = str(f.read())
    #         # f.close()
    #
    #         print("Bark!")
    #
    #         # # return user to required page
    #         return FileResponse(open(os.getcwd()+"/Data/processed_file.csv", "rb"), as_attachment=True)
    #         # return FileResponse(text, as_attachment=True)
    #         # return HttpResponse(text)
    #         # return HttpResponseRedirect(reverse('about'))



class BlackGEMView(TemplateView):
    template_name = 'blackGEM.html'

    def get_context_data(self, **kwargs):
        return {'targets': Target.objects.all()}




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
