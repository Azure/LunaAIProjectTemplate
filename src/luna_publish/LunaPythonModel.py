from azureml.core import model
from mlflow.pyfunc import PythonModel, PythonModelContext
from luna.lunaUtils import LunaUtils

from sklearn.linear_model import LinearRegression
from luna.numpyJsonEncoder import NumpyJSONEncoder
import pandas as pd
import json
import os
import pickle
import requests

from shutil import copyfile


class LunaPythonModel(PythonModel):
    def load_context(self, context):
        ## DO NOT CHANGE! Set mlflow as default run mode
        self._run_mode = 'mlflow'

        # This function will only be called if user deploy the model to a real time service endpoint
        # This function will be called every time when the container instance is started
        # DO NOT use context otherwise it won't work on Azure ML
        return

    def predict(self, context, model_input):
        ## DO NOT CHANGE! Get the model path
        model_path = LunaUtils.GetModelPath(run_mode = self._run_mode, context = context)
        
        # Add your scoring code here. You model is the following directory: model_path/<model_path you specified in train method>
        # DO NOT use context otherwise it won't work on Azure ML
        # Update the scoring_result with the real result
        
        model_file = os.path.join(model_path, 'models/model.pkl')
        model = pickle.load( open( model_file, "rb" ) )
        
        user_input = json.loads(model_input)

        scoring_result = model.predict(user_input["records"])
        
        scoring_result = json.dumps(scoring_result, cls=NumpyJSONEncoder)
        return scoring_result

    def train(self, args, user_input, logger):
        # train your model here
        # userInput is a dictionary, for example userInput['source']

        # Update the model_path if your model is saved in a different folder. 
        # All files under model_path will be saved and registered as a part of the model
        # Update the description for your model

        # Logging example:
        # 1. Log metric: logger.log_metric("accuracy, 0.89)
        # 2. Log metrics: logger.log_metrics({"accuracy": 0.89, "execution_time_in_sec": 100})
        # 3. Upload a file or artifact: logger.upload_artifacts(local_file_name, upload_file_name)
        # 4. Upload files or artifacts: logger.upload_artifacts(local_directory_name, upload_directory_name)

        train_data = pd.read_csv(user_input["trainingDataSource"])

        label_column_name = user_input['labelColumnName'] if 'labelColumnName' in user_input else train_data.columns[-1]
        timeout = user_input['timeout'] if 'timeout' in user_input else 10
        description = user_input['description'] if 'description' in user_input else 'this is my model description'

        X = train_data.drop([label_column_name], axis=1)

        Y = train_data[label_column_name]

        lin_reg = LinearRegression()
        lin_reg.fit(X, Y)

        model_path = 'models'
        model_file = os.path.join(model_path, "model.pkl")
        pickle.dump(lin_reg, open(model_file, 'wb'))

        return model_path, description

    def batch_inference(self, args, user_input, model_path, logger):
        # Do your batch inference here. You model is the following directory: model_path/<model_path you specified in train method>
        # userInput is a dictionary, for example userInput['source'] or userInput['hyper_parameters]['epocs']
        # The return value will be ignored. You should ask user to provide a output data source as user input and write the result

        # Logging example:
        # 1. Log metric: logger.log_metric("accuracy, 0.89)
        # 2. Log metrics: logger.log_metrics({"accuracy": 0.89, "execution_time_in_sec": 100})
        # 3. Upload a file or artifact: logger.upload_artifacts(local_file_name, upload_file_name)
        # 4. Upload files or artifacts: logger.upload_artifacts(local_directory_name, upload_directory_name)
        
        data = pd.read_csv(user_input["dataSource"])
        output_filename = user_input["output"]

        model_file = os.path.join(model_path, "models", "model.pkl")
        model = pickle.load(open(model_file, 'rb'))
        
        y_proba = model.predict(data)

        temp_filename = "imputation_result.csv"
        with open(temp_filename, "wt") as temp_file:
            pd.DataFrame(y_proba).to_csv(temp_file, header=False)

        with open(temp_filename , 'rb') as fh:
            response = requests.put(output_filename,
                                data=fh,
                                headers={
                                            'content-type': 'text/csv',
                                            'x-ms-blob-type': 'BlockBlob'
                                        }
                                )

        return

    ## DO NOT CHANGE
    def set_run_mode(self, run_mode):
        self._run_mode = run_mode
