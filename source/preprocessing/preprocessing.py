import pandas as pd 
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

def num_preprocessing(df):
    logger.debug('Preprocessing data for numerical model')

    # Remove properties not sold at full market price and remove small amount of Irish descriptions of properties.
    df = df[df['Not Full Market Price'] == 'No']
    df = df[(df['Description of Property'] == 'Second-Hand Dwelling house /Apartment') | (df['Description of Property'] == 'New Dwelling house /Apartment')]
    
    # Look at 2016-2018 data only.
    df['Date of Sale (dd/mm/yyyy)'] = pd.to_datetime(df['Date of Sale (dd/mm/yyyy)'], dayfirst=True, format='%d/%m/%Y')
    df = df.rename(columns={'Date of Sale (dd/mm/yyyy)': 'Date of Sale', 'Price ()': 'Price'})
    df = df[df['Date of Sale'] > '2016-01-01'].reset_index(drop=True)

    # Parse 'Price' column into correct format.
    df['Price'] = df['Price'].apply(lambda x: x.lstrip('\x80')).apply(lambda x: float(x.split()[0].replace(',', ''))).astype(float)

    # One-hot encode categorical variables.
    df = pd.concat([df, pd.get_dummies(df[['Property Size Description',
                                           'Description of Property',
                                           'VAT Exclusive',
                                           'Not Full Market Price',
                                           'County',
                                           'Postal Code']])], axis=1)
    df = df.drop(['County', 'Date of Sale', 'Property Size Description',
                  'Description of Property', 'VAT Exclusive',
                  'Not Full Market Price', 'Postal Code'], axis=1)
                  
    # Without geocoding:
    # df = df.drop('Address', axis=1)
    
    logger.debug('Processed dataframe')
    return df

def cat_preprocessing(df):
    logger.debug('Preprocessing data for CatBoost model')

    # Remove properties not sold at full market price and remove small amount of Irish descriptions of properties.
    df = df[df['Not Full Market Price'] == 'No']
    df = df[(df['Description of Property'] == 'Second-Hand Dwelling house /Apartment') | (df['Description of Property'] == 'New Dwelling house /Apartment')]
    
    # Look at 2016-2018 data only.
    df['Date of Sale (dd/mm/yyyy)'] = pd.to_datetime(df['Date of Sale (dd/mm/yyyy)'], dayfirst=True, format='%d/%m/%Y')
    df = df.rename(columns={'Date of Sale (dd/mm/yyyy)': 'Date of Sale', 'Price ()': 'Price'})
    df = df[df['Date of Sale'] > '2016-01-01'].reset_index(drop=True)

    # Parse 'Price' column into correct format.
    df['Price'] = df['Price'].apply(lambda x: x.lstrip('\x80')).apply(lambda x: float(x.split()[0].replace(',', ''))).astype(float)

    # Set index to 'Date of Sale'
    df = df.set_index('Date of Sale')

    # Convert NaN values to strings for CatBoost.
    df['Postal Code'][df['Postal Code'].isna()] = 'None'
    df['Property Size Description'][df['Property Size Description'].isna()] = 'None'
    
    # Without geocoding:
    # df = df.drop('Address', axis=1)
    
    logger.debug('Processed dataframe')
    return df

def main():
    logger.info('Preprocessing data')
    try:
        df = pd.read_csv('data/PPR-ALL.zip', encoding='latin-1', compression='zip')
    except OSError as e:
        logger.error(e)
        raise

    df = cat_preprocessing(df)

    logger.info('Saving dataframe to CSV')
    try:
        df.to_csv('output/processed_ppr_cat.csv', encoding='latin-1')
    except OSError as e:
        logger.error(e)
        raise
    
if __name__ == '__main__':
    main()