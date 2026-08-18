import numpy as np
import bentoml
from typing import Any, cast

iris_model = cast(Any, bentoml.sklearn.load_model("iris_clf:latest"))


@bentoml.service
class IrisClassifier:

    @bentoml.api
    def classify(self, features: np.ndarray) -> np.ndarray:
        prediction = iris_model.predict(features)
        return np.asarray(prediction)

# to run the service, use the following command in terminal:
# bentoml serve service.py:IrisClassifier --reload

# use bentoml build to build a BentoML bundle for deployment