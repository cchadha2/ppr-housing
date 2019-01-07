import pandas as pd
import numpy as np
import requests
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

tqdm.pandas()


class Geocoding:
    geo_url = 'https://maps.googleapis.com/maps/api/geocode/json?address='

    def __init__(self, address, api_key):
        add_list = address.split()
        add_list = [line + '+' for line in add_list[:-1]]
        add_list.append(address.split()[-1])
        add_url = "".join(add_list)
        self.url = self.geo_url + add_url + '&key=' + api_key

    def lat_lng(self):
        logger.debug('Requesting geospatial info from Google API')
        try:
            response = requests.get(self.url)
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


def main():

    api_key = ''

    try:
        df = pd.read_csv('output/processed_ppr_cat_2018.csv', encoding='latin-1')
    except OSError as e:
        logger.error(e)
        raise

    logger.info('Requesting geocodes from Google API')
    df['coordinates'] = df['Address'].progress_apply(lambda x: Geocoding(x, api_key=api_key).lat_lng())

    logger.info('Saving dataframe to CSV')
    try:
        df.to_csv('output/geocodes_2018.csv', index=False)
    except OSError as e:
        logger.error(e)
        raise


if __name__ == '__main__':
    main()
