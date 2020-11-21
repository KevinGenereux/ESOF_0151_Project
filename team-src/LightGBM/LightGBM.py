from ReadDataset import readFiles
from DataPreprocessing import reduceMemUsage

train, test = readFiles()

train = reduceMemUsage(train)
test = reduceMemUsage(train)

