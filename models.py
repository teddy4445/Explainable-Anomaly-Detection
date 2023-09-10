# Anomaly-detection models by PyOD
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM

# Classification models by sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

AD_MODELS = {
    "IForest": IForest,
    "OCSVM": OCSVM,
}

CLF_MODELS = {
    "SVM": SVC,
    "RandomForest": RandomForestClassifier,
    "GradientBoosting": GradientBoostingClassifier
}
