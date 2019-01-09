import numpy as np
import requests
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def url_creator(address, url='https://maps.googleapis.com/maps/api/geocode/json?address=', api_key=''):
    logger.debug('Creating API request URL')
    add_list = address.split()
    add_list = [line + '+' for line in add_list[:-1]]
    add_list.append(address.split()[-1])
    add_url = "".join(add_list)
    return url + add_url + '&key=' + api_key


def lat_lng(url):
    logger.debug('Requesting geospatial info from Google API')
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logger.error(e)

    results = response.json()['results']
    if len(results) != 0:
        logger.debug('Found geospatial coordinates')
        return results[0]['geometry']['location']['lat'], results[0]['geometry']['location']['lng']
    else:
        logger.debug('Could not find geospatial coordinates')
        return np.nan, np.nan
