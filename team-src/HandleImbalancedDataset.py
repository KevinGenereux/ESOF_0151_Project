from Imports import *

def handleImbalancedDataset(method, x ,y):
    X_resampled = []
    Y_resampled = []
    seed = 123

    if method.lower() == "smote":
        sm = SMOTE(sampling_strategy='auto', random_state=seed)
        X_resampled, Y_resampled = sm.fit_resample(x,y)

    if method.lower() == "adasyn":
        adas = ADASYN()
        X_resampled, Y_resampled = adas.fit_resample(x, y)

    if method.lower() == "enn":
        enn = EditedNearestNeighbours()
        X_resampled, Y_resampled = enn.fit_resample(x, y)

    if method.lower() == "cnn":
        cnn = CondensedNearestNeighbour()
        X_resampled, Y_resampled = cnn.fit_resample(x, y)

    if method.lower() == "oss":
        oss = OneSidedSelection()
        X_resampled, Y_resampled = oss.fit_resample(x, y)

    if method.lower() == "nm":
        nm = NearMiss(version=3,n_neighbors_ver3=3)
        X_resampled, Y_resampled = nm.fit_resample(x, y)

    if method.lower() == "smotetomek":
        smotetomek = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
        X_resampled, Y_resampled = smotetomek.fit_resample(x, y)

    return X_resampled, Y_resampled