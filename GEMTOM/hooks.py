import logging
from .views import *

logger = logging.getLogger(__name__)

def after_uploading_target(target, created):
    print('New target made:', target)
    print(type(target))
    print(vars(target))

    target_name = target.name
    target_id = str(target.id)
    target_blackgemid = get_blackgem_id_from_iauname(target_name)
    print(target_blackgemid)
    add_bgem_lightcurve_to_GEMTOM(target_name, target_id, target_blackgemid)



def observation_change_state(observation, previous_status):
    logger.info(
        'Sending email, observation %s changed state from %s to %s',
        observation, previous_status, observation.status
    )
    print("\n\n\n\nObservation state changed!\n\n\n\n")
