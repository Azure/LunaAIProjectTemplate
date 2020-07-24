from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline
from azureml.core import Workspace
from azureml.pipeline.core.graph import PipelineParameter
from azureml.core import Experiment
from azureml.core.webservice import AciWebservice, AksWebservice

from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from uuid import uuid4

from luna import utils
import os
from uuid import uuid4
import requests
import json
import pathlib

modelId = '00000000-0000-0000-0000-000000000000'
endpointId = '00000000-0000-0000-0000-000000000000'
operationId = '00000000-0000-0000-0000-000000000000'
subscriptionId = '00000000-0000-0000-0000-000000000000'

experimentName = 'xiwumlflowTest'
userId = 'xiwu@microsoft.com'
productName = 'eddi'
deploymentName = 'westus'
apiVersion = 'v1.0'
ws = None
dns_name_label = 'testlabel'

def init():
    global ws, modelId, endpointId, operationId, subscriptionId, dns_name_label, test_data
    ws = Workspace.from_config(path='.cloud/.azureml/', _file_name='test_workspace.json')
    modelId = str('a' + uuid4().hex[1:])
    endpointId = str('a' + uuid4().hex[1:])
    operationId = str('a' + uuid4().hex[1:])
    subscriptionId = str('a' + uuid4().hex)[1:]
    dns_name_label = str('a' + uuid4().hex[1:])
    print('modelId {}'.format(modelId))
    print('endpointId {}'.format(endpointId))
    print('operationId {}'.format(operationId))
    print('subscriptionId {}'.format(subscriptionId))
    print('dns_name_label {}'.format(dns_name_label))
    test_data_file = os.path.join(pathlib.Path(__file__).parent.absolute(), "test_data.json")
    with open(test_data_file) as f:
        test_data = json.load(f)

def trainModel(userInput='{}'):
    run_id = utils.RunProject(azureml_workspace = ws, 
                                    entry_point = 'training', 
                                    experiment_name = experimentName, 
                                    parameters={'modelId': modelId, 
                                                'userInput': userInput, 
                                                'operationId': operationId,
                                                'productName': productName,
                                                'deploymentName': deploymentName,
                                                'apiVersion': apiVersion}, 
                                    tags={'userId': userId, 
                                            'productName': productName, 
                                            'deploymentName': deploymentName, 
                                            'apiVersion': apiVersion,
                                            'modelId': modelId,
                                            'subscriptionId': subscriptionId})

    print(run_id)
    return run_id

def batchInference(userInput='{}'):
    run_id = utils.RunProject(azureml_workspace = ws, 
                                    entry_point = 'batchinference', 
                                    experiment_name = experimentName, 
                                    parameters={'modelId': modelId, 
                                                'userInput': userInput, 
                                                'operationId': operationId}, 
                                    tags={'userId': userId, 
                                            'productName': productName, 
                                            'deploymentName': deploymentName, 
                                            'apiVersion': apiVersion,
                                            'modelId': modelId,
                                            'subscriptionId': subscriptionId,
                                            'operationId': operationId})
    print(run_id)
    return run_id

def deploy(userInput='{"dns_name_label":"testlabel"}'):
    run_id = utils.RunProject(azureml_workspace = ws, 
                                    entry_point = 'deployment', 
                                    experiment_name = experimentName, 
                                    parameters={'modelId': modelId, 
                                                'userInput': userInput, 
                                                'endpointId': endpointId,
                                                'productName': productName,
                                                'deploymentName': deploymentName,
                                                'apiVersion': apiVersion}, 
                                    tags={'userId': userId, 
                                            'productName': productName, 
                                            'deploymentName': deploymentName, 
                                            'apiVersion': apiVersion,
                                            'modelId': modelId,
                                            'subscriptionId': subscriptionId,
                                            'endpointId': endpointId})
    print(run_id)
    return run_id

def get_run_by_tags(tags):
    exp = Experiment(ws, experimentName)
    runs = exp.get_runs(type='azureml.PipelineRun', tags=tags)
    run = next(runs)
    print(run.status)
    return run

def wait_for_run_completion(run):
    run.wait_for_completion(show_output=True)

def trace_run_by_tags(run_id, tags):
    run = get_run_by_tags(tags)
    if run.id != run_id:
        raise Exception('Found the wrong run. Expected run id: {}, actual run id: {}'.format(run_id, run.id))
    wait_for_run_completion(run)
    run = get_run_by_tags({'modelId': modelId, 'userId': userId, 'subscriptionId': subscriptionId})
    if run.status != 'Completed':
        raise Exception('Run completed with result:' + run.status)

def test_deployed_aci_service(data, expected_output):
    webservice = AciWebservice(ws, endpointId)
    headers = {'Content-Type': 'application/json'}
    headers['Authorization'] = 'Bearer '+webservice.get_keys()[0]

    test_sample = json.dumps(data)

    response = requests.post(
    webservice.scoring_uri, data=test_sample, headers=headers)
    if response.status_code != 200:
        raise Exception('The service return non-success status code: {}'.format(response.status_code))
    if response.json() != expected_output:
        raise Exception('The scoring result is incorrect: {}'.format(response.json()))

if __name__ == "__main__":
    init()
    
    run_id = trainModel(userInput=json.dumps(test_data['training_user_input']))
    trace_run_by_tags(run_id, tags={'modelId': modelId, 'userId': userId, 'subscriptionId': subscriptionId})
    run_id = batchInference(userInput=json.dumps(test_data['batch_inference_input']))
    trace_run_by_tags(run_id, tags={'operationId': operationId, 'userId': userId, 'subscriptionId': subscriptionId})

    userInput = '{{"dns_name_label":"{}"}}'.format(dns_name_label)
    run_id = deploy(userInput=userInput)
    trace_run_by_tags(run_id, tags={'endpointId': endpointId, 'userId': userId, 'subscriptionId': subscriptionId})

    test_deployed_aci_service(data=test_data['real_time_scoring_input'], expected_output=test_data['real_time_scoring_expected_output'])