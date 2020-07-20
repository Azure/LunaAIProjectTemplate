from src.luna_publish.LunaPythonModel import LunaPythonModel

def init():
    global python_model
    python_model = LunaPythonModel()
    python_model.set_run_mode('azureml')

def run(userInput):
    result = python_model.predict(None, userInput)
    return result