from luna.lunaUtils import LunaUtils
import os
from LunaPythonModel import LunaPythonModel
import json
from datetime import datetime

utils = LunaUtils.Create()
args = utils.args
user_input = utils.user_input

python_model = LunaPythonModel()

model_path, description = python_model.train(args, user_input, utils.logger)


utils.RegisterModel(model_path = model_path,
                       description = description,
                       luna_python_model=LunaPythonModel())

local_directory_name = os.path.join(model_path, 'metadata')
output_file_name = os.path.join(local_directory_name, 'output.json')

data = {'model_id': args.operationId, 'description': description, 'registered_date': str(datetime.utcnow())}

if not os.path.exists(local_directory_name):
    os.makedirs(local_directory_name)
print(local_directory_name)
print(output_file_name)
print(data)
with open(output_file_name, 'w') as outfile:
    json.dump(data, outfile)

utils.logger.upload_artifacts(local_directory_name, 'outputs')