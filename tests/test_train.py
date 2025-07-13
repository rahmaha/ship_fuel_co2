import os


def test_model_artifacts_exist():
    assert os.path.exists("models/model.pkl"), "Model file not found"
    assert os.path.exists("models/dv.pkl"), "DictVectorizer file not found"
