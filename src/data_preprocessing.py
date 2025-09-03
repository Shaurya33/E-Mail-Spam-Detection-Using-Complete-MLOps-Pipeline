import pandas as pd
import os
import logging
import nltk
import string
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

#Make "log" directory for logging purposes
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

#setting console logger
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

#setting file logger
log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

#set logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text: str) -> str:
    """Performs text preprocessing like converting it to lowercase, tokenizing, removing stopwords, punctuation and stemming"""
    try:
        #convert to lower case
        text = text.lower()
        #tokenize the text
        text = nltk.word_tokenize(text)
        # Remove non-alphanumeric tokens
        text = [word for word in text if word.isalnum()]
        #remove stopwords and punctuation
        text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
        #stem the words
        ps = PorterStemmer()
        text = [ps.stem(word) for word in text]
        return " ".join(text)
    except Exception as e:
        logger.debug("Enexpected error occured: %s", e)
        raise


def preprocess_df(df: pd.DataFrame, text_column = 'text', target_column = 'target') -> pd.DataFrame:
    """Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column."""

    try:
        logger.debug("Label encoding started.")
        #Encode the Target Column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug("Label encoding ended.")
        #Remove duplicate rows
        df = df.drop_duplicates(keep= 'first')
        logger.debug("Duplicates removed")
        #Apply text transformation to the 'text' column
        df.loc[:,text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise


def main(text_column='text', target_column='target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        # Transform the data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()