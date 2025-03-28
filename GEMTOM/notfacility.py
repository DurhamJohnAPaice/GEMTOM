from tom_observations.facility import BaseRoboticObservationFacility, BaseRoboticObservationForm
from astropy import units


class NOTFacilityForm(BaseRoboticObservationForm):
    pass


class NOTFacility(BaseRoboticObservationFacility):
    print("NOTFacility Class")
    name = 'NOT'
    # observation_types = [('OBSERVATION', 'Custom Observation')]
    observation_forms = {
        'OBSERVATION': NOTFacilityForm
    }

    SITES = {
        'Roque de los Muchachos': {
            'latitude': 28.75611111,
            'longitude': 17.89166667,
            'elevation': 2396.00
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
        return

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
