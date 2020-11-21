from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

seed = 123
X_resampled = []
Y_resampled = []



def handleImbalancedDatset(method, x ,y):
    if method.lowercase == "smote":
        sm = SMOTE(sampling_strategy='auto', random_state=seed)
        X_resampled, Y_resampled = sm.fit_resample(x,y)

    if method.lowercase == "adasyn":
        adas = ADASYN()
        X_resampled, Y_resampled = adas.fit_resample(x, y)

    if method.lowercase == "enn":
        enn = EditedNearestNeighbours()
        X_resampled, Y_resampled = enn.fit_resample(x, y)

    if method.lowercase == "cnn":
        cnn = CondensedNearestNeighbour()
        X_resampled, Y_resampled = cnn.fit_resample(x, y)

    if method.lowercase == "oss":
        oss = OneSidedSelection()
        X_resampled, Y_resampled = oss.fit_resample(x, y)

    if method.lowercase == "nm":
        nm = NearMiss(version=3,n_neighbors_ver3=n)
        X_resampled, Y_resampled = nm.fit_resample(x, y)

    if method.lowercase == "smotetomek":
        smotetomek = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
        X_resampled, Y_resampled = smotetomek.fit_resample(x, y)

    return X_resampled, Y_resampled