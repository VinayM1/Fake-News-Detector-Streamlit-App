# data_loader.py
import pandas as pd

def load_dataset(file_path=None):
    """
    This function loads our news dataset from CSV files.
    It handles loading both 'Fake.csv' and 'True.csv' and combining them.

    Args:
        file_path (str, optional): Not used directly for loading Fake/True.csv,
                                   but kept for consistency.

    Returns:
        pandas.DataFrame: A table (DataFrame) containing the combined news data.
    """
    try:
        # Load Fake news
        df_fake = pd.read_csv('data/Fake.csv')
        df_fake['label'] = 'FAKE' # Add a 'label' column and set it to 'FAKE'
        print(f"Successfully loaded data from data/Fake.csv ({len(df_fake)} rows).")

        # Load True news
        df_true = pd.read_csv('data/True.csv')
        df_true['label'] = 'REAL' # Add a 'label' column and set it to 'REAL'
        print(f"Successfully loaded data from data/True.csv ({len(df_true)} rows).")

        # Combine both dataframes into a single one
        df = pd.concat([df_fake, df_true], ignore_index=True)
        print(f"Combined dataset has {len(df)} rows and {len(df.columns)} columns.")

        # Shuffle the combined dataframe to mix real and fake news
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        print("Dataset shuffled to mix real and fake news.")

        return df
    except FileNotFoundError as e:
        print(f"Error: One of the data files was not found: {e}")
        print("Please make sure 'Fake.csv' and 'True.csv' are in the 'data' folder.")
        return None
    except Exception as e:
        print(f"An error occurred while loading or combining the data: {e}")
        return None

def get_data_for_training(df):
    """
    This function prepares the data for our robot to learn from.
    It separates the news text from the 'real' or 'fake' label.

    Args:
        df (pandas.DataFrame): The loaded news data.

    Returns:
        tuple: (news_texts, labels) where news_texts are the stories
               and labels are 'REAL' or 'FAKE'.
    """
    if df is None:
        return None, None

    # We'll use the 'text' column for the news story content
    # And the 'label' column to know if it's real or fake
    # We also combine 'title' and 'text' to give our robot more clues!
    df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

    # 'x' will be our news stories (the input for the robot)
    X = df['full_text']
    # 'y' will be our labels (the correct answers for the robot to learn)
    y = df['label']

    print("Data prepared for training: X (full_text) and y (label).")
    return X, y

