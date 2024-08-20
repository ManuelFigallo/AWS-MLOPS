'''
STEP 1.0 - Data Management. Clean Text Data

test branch1

'''
"""
    Remove specified characters from a given column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the column.
    column (str): The name of the column from which to remove characters.
    chars_to_remove (list): A list of characters to remove from the column.

    Returns:
    pd.DataFrame: The DataFrame with the specified characters removed from the column.
"""
import pandas as pd

def remove_characters(df, column, chars_to_remove):
    # Create a regex pattern from the list of characters to remove
    pattern = '[' + ''.join(chars_to_remove) + ']'

    # Use str.replace to remove the characters
    df[column] = df[column].str.replace(pattern, '', regex=True)

    return df
