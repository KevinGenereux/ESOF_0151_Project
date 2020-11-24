from Imports import *

# Split the dataset into training and testing sets
def test_train_split(df, test_ratio):
    # Use a seed so that the dataset is split consistently when training/testing each classifier
    seed = 123
    # Splitting into training and testing set using ratio
    train, test = train_test_split(df, test_size = test_ratio, random_state = test_ratio)
    return train, test