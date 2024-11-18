import os
import sys
from pathlib import Path
import seaborn as sns
import pytest

# Getting project root
project_root = Path(__file__).resolve().parent.parent.parent

# Add the project root directory to sys.path
sys.path.append(str(project_root))


from ps3.data._sample_split import create_sample_split

@pytest.mark.parametrize(
    "training_frac",[0.2, 0.4, 0.6, 0.8, 1]
)

def test__sample_split(training_frac):
    # Import practise dataframe
    dta = sns.load_dataset("iris")

    # Adding an id column that is == index
    dta["id"] = dta.index

    train_df = create_sample_split(dta, "id", training_frac=training_frac)[0]
    test_df = create_sample_split(dta, "id", training_frac=training_frac)[1]

    print(len(train_df) + len(test_df))

    assert len(train_df) + len(test_df) == len(dta), "The size of the split datasets does not add up to the size of total dataset"
    
    upper_bound = training_frac + 0.05
    lower_bound = training_frac - 0.05
    training_fraction = len(train_df)/ (len(train_df) + len(test_df))
    
    assert  lower_bound <= training_fraction <= upper_bound, f"length of training set is not within 5% of training_frac"

