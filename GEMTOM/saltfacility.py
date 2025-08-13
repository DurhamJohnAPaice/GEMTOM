from tom_observations.facility import BaseRoboticObservationFacility, BaseRoboticObservationForm
from astropy import units


class SALTFacilityForm(BaseRoboticObservationForm):
    pass


class SALTFacility(BaseRoboticObservationFacility):
    print("SALTFacility Class")
    name = 'SALT'
    # observation_types = [('OBSERVATION', 'Custom Observation')]
    observation_forms = {
        'OBSERVATION': SALTFacilityForm
    }

    SITES = {
        'South African Astronomical Observatory': {
            'latitude': -32.375918,
            'longitude': 20.810757,
            'elevation': 1798.00
        }
    }

    def get_flux_constant(self):
        # print("Getting Flux Constant")
        return units.erg / units.angstrom
        # return 2

    def get_wavelength_units(self):
        return units.angstrom

    def data_products(self):
        return

    def get_form(self, observation_type):
        return SALTFacilityForm

    def get_observation_status(self):
        return

    def get_observation_url(self):
        return

    def get_observing_sites(self):
        return self.SITES

    def get_terminal_observing_states(self):
        return

    def submit_observation(self):
        return

    def validate_observation(self):
        return
