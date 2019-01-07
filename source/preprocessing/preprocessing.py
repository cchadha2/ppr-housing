import pandas as pd 
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class PreprocessingPPR:

    def __init__(self, df, geocoding=True):
        self.df = df
        self.df = self._remove_unnecessary_rows(self.df)
        self.df = self._time_subset(self.df)
        self.df = self._parse_price(self.df)

        self.geocoding = geocoding


    @staticmethod
    def _remove_unnecessary_rows(df):
        # Remove properties not sold at full market price and remove small amount of Irish descriptions of properties.
        df = df[df['Not Full Market Price'] == 'No']
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
        df = df[df['Date of Sale'] > '2016-01-01'].reset_index(drop=True)
        return df

    @staticmethod
    def _parse_price(df):
        # Parse 'Price' column into correct format.
        df['Price'] = df['Price'].apply(lambda x: x.lstrip('\x80'))
        df['Price'] = df['Price'].apply(lambda x: float(x.split()[0].replace(',', ''))).astype(float)
        return df

    @staticmethod
    def _one_hot_encode(df):
        # One-hot encode categorical variables.
        df = pd.concat([df, pd.get_dummies(df[['Property Size Description',
                                               'Description of Property',
                                               'VAT Exclusive',
                                               'Not Full Market Price',
                                               'County',
                                               'Postal Code']])], axis=1)
        df = df.drop(['County',
                      'Date of Sale',
                      'Property Size Description',
                      'Description of Property',
                      'VAT Exclusive',
                      'Not Full Market Price',
                      'Postal Code'], axis=1)
        return df

    @staticmethod
    def _no_geocoding(df):
        return df.drop('Address', axis=1)

    def num_preprocessing(self):
        self.df = self._one_hot_encode(self.df)

        if self.geocoding is False:
            self.df = self._no_geocoding(self.df)

        return self.df

    def cat_preprocessing(self):
        # Set index to 'Date of Sale'
        self.df = self.df.set_index('Date of Sale')

        # Convert NaN values to strings for CatBoost.
        self.df['Postal Code'][self.df['Postal Code'].isna()] = 'None'
        self.df['Property Size Description'][self.df['Property Size Description'].isna()] = 'None'

        if self.geocoding is False:
            self.df = self._no_geocoding(self.df)

        return self.df


def main():
    logger.info('Preprocessing data')
    try:
        df = pd.read_csv('data/PPR-ALL.zip', encoding='latin-1', compression='zip')
    except OSError as e:
        logger.error(e)
        raise

    preprocess = PreprocessingPPR(df)
    processed_df = preprocess.cat_preprocessing()

    logger.info('Saving dataframe to CSV')
    try:
        processed_df.to_csv('output/processed_ppr_cat.csv', encoding='latin-1')
    except OSError as e:
        logger.error(e)
        raise


if __name__ == '__main__':
    main()
