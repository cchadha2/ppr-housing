import logging
from tqdm import tqdm
import pandas as pd

from source.preprocessing.geocoding import lat_lng

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

tqdm.pandas()


class PreprocessingPPR:

    def __init__(self, df, api_key, geocoding=True):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.df = df
        self.df = self._remove_unnecessary_rows(self.df)
        self.df = self._time_subset(self.df)
        self.df = self._parse_price(self.df)

        self.geocoding = geocoding
        self.api_key = api_key


    @staticmethod
    def _remove_unnecessary_rows(df):
        # Remove properties not sold at full market price and remove small amount of Irish descriptions of properties.
        df = df[df['Not Full Market Price'] == 'No']
        df = df.drop('Not Full Market Price', axis=1)
        df = df[(df['Description of Property'] == 'Second-Hand Dwelling house /Apartment') | (
                 df['Description of Property'] == 'New Dwelling house /Apartment')]
        return df

    @staticmethod
    def _time_subset(df):
        # Look at data from 2016 onwards only.
        df['Date of Sale (dd/mm/yyyy)'] = pd.to_datetime(df['Date of Sale (dd/mm/yyyy)'],
                                                         dayfirst=True,
                                                         format='%d/%m/%Y')
        df = df.rename(columns={'Date of Sale (dd/mm/yyyy)': 'Date of Sale', 'Price ()': 'Price'})
        # df = df[(df['Date of Sale'] > '2018-01-01')].reset_index(drop=True)
        return df

    @staticmethod
    def _parse_price(df):
        # Parse 'Price' column into correct format.
        df['Price'] = df['Price'].apply(lambda x: x.lstrip('\x80'))
        df['Price'] = df['Price'].apply(lambda x: float(x.split()[0].replace(',', ''))).astype(float)
        return df

    @staticmethod
    def _no_geocoding(df):
        return df.drop('Address', axis=1)

    def _geocoding(self, df):
        address_list = df['Address'].tolist()
        coordinates = [lat_lng(address, api_key=self.api_key) for address in address_list]
        df['lat'] = [x[0] for x in coordinates]
        df['lng'] = [x[1] for x in coordinates]
        return df

    def num_processing(self):
        self.logger.info('One-hot encoding data')
        # One-hot encode categorical variables.
        self.df = pd.concat([self.df, pd.get_dummies(self.df[['Property Size Description', 'Description of Property',
                                                              'VAT Exclusive', 'County', 'Postal Code']])], axis=1)
        self.df = self.df.drop(['County', 'Date of Sale', 'Property Size Description', 'Description of Property',
                                'VAT Exclusive', 'Postal Code'], axis=1)

        if self.geocoding is True:
            self.logger.info('Geocoding addresses')
            self.df = self._geocoding(self.df)
        else:
            self.df = self._no_geocoding(self.df)

        return self.df

    def cat_processing(self):
        self.logger.info('Processing data for CatBoost')
        # Set index to 'Date of Sale'
        self.df = self.df.set_index('Date of Sale')

        # Convert NaN values to strings for CatBoost.
        self.df['Postal Code'][self.df['Postal Code'].isna()] = 'None'
        self.df['Property Size Description'][self.df['Property Size Description'].isna()] = 'None'

        if self.geocoding is True:
            self.logger.info('Geocoding addresses')
            self.df = self._geocoding(self.df)
        else:
            self.df = self._no_geocoding(self.df)

        return self.df


def main():
    logger.info('Pre-processing data')
    try:
        df = pd.read_csv('data/PPR-ALL.zip', encoding='latin-1', compression='zip', nrows=2)
    except OSError as e:
        logger.error(e)
        raise

    preprocessing_object = PreprocessingPPR(df, 'api-key')
    processed_df = preprocessing_object.cat_processing()

    logger.info('Saving DataFrame to CSV')
    try:
        processed_df.to_csv('output/processed_ppr.csv', index=False)
    except OSError as e:
        logger.error(e)
        raise


if __name__ == '__main__':
    main()
