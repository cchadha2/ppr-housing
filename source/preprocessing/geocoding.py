import numpy as np
import requests
import logging


class Geocoding:
    geo_url = 'https://maps.googleapis.com/maps/api/geocode/json?address='

    def __init__(self, address, api_key):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        self.logger.addHandler(ch)

        add_list = address.split()
        add_list = [line + '+' for line in add_list[:-1]]
        add_list.append(address.split()[-1])
        add_url = "".join(add_list)
        self.url = self.geo_url + add_url + '&key=' + api_key

    def lat_lng(self):
        self.logger.debug('Requesting geospatial info from Google API')
        try:
            response = requests.get(self.url)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            self.logger.error(e)

        results = response.json()['results']
        if len(results) != 0:
            self.logger.debug('Found geospatial coordinates')
            return results[0]['geometry']['location']['lat'], results[0]['geometry']['location']['lng']
        else:
            self.logger.debug('Could not find geospatial coordinates')
            return np.nan, np.nan
