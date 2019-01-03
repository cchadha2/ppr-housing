import pandas as pd 
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

def lgbm_preprocessing(df):
    logger.debug('Preprocessing data for LightGBM model')
    df = df[df['Not Full Market Price'] == 'No']
    df = df[(df['Description of Property'] == 'Second-Hand Dwelling house /Apartment') | (df['Description of Property'] == 'New Dwelling house /Apartment')]
    df['Date of Sale (dd/mm/yyyy)'] = pd.to_datetime(df['Date of Sale (dd/mm/yyyy)'], dayfirst=True, format='%d/%m/%Y')
    df = df.rename(columns={'Date of Sale (dd/mm/yyyy)': 'Date of Sale', 'Price ()': 'Price'})
    df = df[df['Date of Sale'] > '2016-01-01'].reset_index(drop=True)
    df['Price'] = df['Price'].apply(lambda x: x.lstrip('\x80'))
    df['Price'] = df['Price'].convert_objects(convert_numeric=True)
    logger.debug('Processed dataframe')
    return df

def main():
    logger.info('Preprocessing data for LightGBM model')
    try:
        df = pd.read_csv('data/PPR-ALL.zip', encoding='latin-1', compression='zip')
    except OSError as e:
        logger.error(e)
        raise

    df = lgbm_preprocessing(df)

    logger.info('Saving dataframe to CSV')
    try:
        df.to_csv('output/processed_ppr.csv', encoding='latin-1', index=False)
    except OSError as e:
        logger.error(e)
        raise
    

if __name__ == '__main__':
    main()