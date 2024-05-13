import requests

# from tom_observations.facilities.lco import LCOFacility, LCOSettings
# from tom_observations.facilities.lco import LCOImagingObservationForm, LCOSpectroscopyObservationForm
# from tom_common.exceptions import ImproperCredentialsException
# from tom_observations.facility import BaseRoboticObservationFacility, BaseRoboticObservationForm


# class BlackGEMSettings(LCOSettings):
#     def get_sites(self):
#         return {
#             'La Silla': {
#                 'sitecode': '809',
#                 'latitude': -29.2552104,
#                 'longitude': -70.739507,
#                 'elevation': 2400
#             }
#         }
#
#     def get_weather_urls(self):
#         return {
#             'code': 'BlackGEM',
#             'sites': [
#                 {
#                     'code': site['sitecode'],
#                     'weather_url': 'https://www.eso.org/public/ireland/outreach/webcams/'
#                                    # 'cerro-pachon/environmental-conditions'
#                 }
#                 for site in self.get_sites().values()]
#         }


# def make_request(*args, **kwargs):
#     response = requests.request(*args, **kwargs)
#     if 400 <= response.status_code < 500:
#         raise ImproperCredentialsException('SOAR: ' + str(response.content))
#     response.raise_for_status()
#     return response


# class SOARImagingObservationForm(LCOImagingObservationForm):
#
#     def get_instruments(self):
#         instruments = super()._get_instruments()
#         return {
#             code: instrument for (code, instrument) in instruments.items() if (
#                 'IMAGE' == instrument['type'] and 'SOAR' in code)
#         }
#
#     def configuration_type_choices(self):
#         return [('EXPOSE', 'Exposure')]
#
#
# class SOARSpectroscopyObservationForm(LCOSpectroscopyObservationForm):
#     def get_instruments(self):
#         instruments = super()._get_instruments()
#         return {
#             code: instrument for (code, instrument) in instruments.items() if (
#                 'SPECTRA' == instrument['type'] and 'SOAR' in code)
#         }
#
#     def configuration_type_choices(self):
#         return [('SPECTRUM', 'Spectrum'), ('ARC', 'Arc'), ('LAMP_FLAT', 'Lamp Flat')]


# class BlackGEMFacility(LCOFacility):
#     """
#     The ``BlackGEM Facility`` is the interface to the BlackGEM Telescope. For information regarding BlackGEM observing and the
#     available parameters, please ask someone.
#     """
#     name = 'BlackGEM'
    # observation_forms = {
    #     'IMAGING': SOARImagingObservationForm,
    #     'SPECTRA': SOARSpectroscopyObservationForm
    # }

    # def __init__(self, facility_settings=BlackGEMSettings('LCO')):
    #     super().__init__(facility_settings=facility_settings)

    # def get_form(self, observation_type):
    #     return self.observation_forms.get(observation_type, SOARImagingObservationForm)


# # custom_ocs.py
# from tom_observations.facilities.ocs import OCSFacility
#
#
# class BlackGEMFacility(BlackGEMFacility):
#    name = 'BlackGEM'
#    # observation_forms = {
#    #     'Instrument1': Instrument1ObservationForm,
#    #     'Spectra': SpectraObservationForm
#    # }
