import hashlib
import numpy as np
import pandas as pd
import seaborn as sns


# Import practise dataframe

dta = sns.load_dataset("iris")

# Adding an id column that is == index
dta["id"] = dta.index

#print(hashlib.algorithms_available)

# Function to return the intiger representation of a 16 hexadecimal hash 

def return_hash_int(key):
    hashvalue = hashlib.md5(str(key).encode()).hexdigest()
    hash_int = int(hashvalue, 16)

    return hash_int

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.9

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    # Creating a new column as a unique identifier based on the intiger representation of hexadecimal hash keys
    df["unique_identifier"] = df[id_column].apply(return_hash_int)

    b = 10**8

    # Take modulus to make the number smaller but preserve the randomness
    df["unique_identifier"] = df["unique_identifier"].apply(lambda x: x%b)

    #as we have the property that 0<=(a%b)<b we can normalise the values, this will ensure a uniform distribution
    df["unique_identifier"] = df["unique_identifier"].apply(lambda x: x/b)

    train_df = df[df["unique_identifier"]<=training_frac]
    test_df = df[df["unique_identifier"]>training_frac]


    return train_df,test_df

test = create_sample_split(dta, "id")

print(f"length of training set is {len(test[0])}")
print(f"length of testing set is {len(test[1])}")
print(123/(123+27))