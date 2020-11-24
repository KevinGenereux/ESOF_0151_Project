from ReadDataset import readFiles
from DataPreprocessing import reduceMemUsage

# Read in the dataset containing the merged transaction and identity tables
train = readMergedDataset()

# Reduce the memory size of the dataset, this will also help improve computational speed
train = reduceMemUsage(train)

train = encodeNumericalColumns(train)

# Replace dataset missing values
train = replaceMissingValues(train)

# SE





test = replaceMissingValues()