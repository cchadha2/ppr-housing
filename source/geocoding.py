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

# Google API
# url='https://maps.googleapis.com/maps/api/geocode/json?address='
# api_key='&key=AIzaSyDHNguaHa1brL6IKKr2EkDDnym32RfaX8E'

# OpenStreetMap API
url =  'https://nominatim.openstreetmap.org/search?q='
api_key = '&format=json'

def url_creator(address, url=url, api_key=api_key):
    logger.debug('Creating API request URL')
    add_list = address.split()
    add_list = [line + '+' for line in add_list[:-1]]
    add_list.append(address.split()[-1])
    add_url="".join(add_list)
    return url+add_url+api_key

def lat_lng(api_url):
    logger.debug('Requesting geocodes from Google API')
    try:
        response = requests.get(api_url)
    except requests.exceptions.RequestException as e:
        logger.error(e)
        raise
    response.raise_for_status()
    # Google API
    # results = response.json()['results']
    # OpenStreetMap API
    logger.info(response.status_code)
    if len(response.json()) != 0:
        logger.debug('Found geospatial coordinates')
        # Google API
        # return results[0]['geometry']['location']['lat'], results[0]['geometry']['location']['lng']
        # OpenStreetMap API
        return response.json()[0]['lat'], response.json()[0]['lon']
    else:
        logger.debug('Could not find geospatial coordinates')
        return np.nan, np.nan

def main():
    df = pd.read_csv('output/processed_ppr.csv', encoding='latin-1')

    logger.info('Creating API request URLs for property price registry')
    df['api_url']=df['Address'].apply(url_creator)
    logger.info('Requesting geocodes from API')
    df['lat'], df['lng'] = df['api_url'].progress_apply(lat_lng)

    logger.info('Saving dataframe to CSV')
    df.to_csv('geocodes.csv', index=False)

if __name__ == '__main__':
    main()