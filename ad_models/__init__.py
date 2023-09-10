from .isolation_forest import IsolationForest
from .one_class_svm import OneClassSVM

AD_MODELS = {
    "IForest": IsolationForest,
    "OCSVM": OneClassSVM,
}
