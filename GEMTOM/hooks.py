import logging

logger = logging.getLogger(__name__)

def after_uploading_target(target):
    logger.info(
        'New target created: %s',
        target
    )
    print('New target created:', target)


def observation_change_state(observation, previous_status):
    logger.info(
        'Sending email, observation %s changed state from %s to %s',
        observation, previous_status, observation.status
    )
    print("\n\n\n\nObservation state changed!\n\n\n\n")
