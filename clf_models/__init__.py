from .gradiant_boosting import GradBoost
from .svm import SVM
from .random_forest import RF

CLF_MODELS = {
    "SVM": SVM,
    "RF": RF,
    "GradBoost": GradBoost
}
