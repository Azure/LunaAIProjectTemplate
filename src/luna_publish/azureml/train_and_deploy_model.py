from azureml.core import model
from azureml.core.webservice.aci import AciWebservice
from azureml.core.webservice.aks import AksWebservice
from luna import utils
from azureml.core import Workspace, Experiment, Run
from azureml.core.webservice import Webservice
from uuid import uuid4
import json
import os
import pathlib
import argparse

if __name__ == "__main__":

    parser=argparse.ArgumentParser(description="Train and publish model") 

    parser.add_argument('-experiment_name', 
                        '--experiment_name', 
                        help="The name of the experiment", 
                        default="train_and_deploy_model",
                        type=str)  
                        
    parser.add_argument('-model_id', 
                        '--model_id', 
                        help="The id of the model", 
                        default="default",
                        type=str)  
                        
    parser.add_argument('-endpoint_id', 
                        '--endpoint_id', 
                        help="The id of the service endpoint", 
                        default="default",
                        type=str)  

    parser.add_argument('-dns_name_label', 
                        '--dns_name_label', 
                        help="The name of DNS name label", 
                        default="default",
                        type=str)  

    parser.add_argument('-input_data_file_path', 
                        '--input_data_file_path', 
                        help="The data file for training input", 
                        default="default",
                        type=str) 

    args=parser.parse_args()

    experimentName = args.experiment_name
    modelId = args.model_id
    endpointId = args.endpoint_id
    serviceEndpointDnsNameLabel = args.dns_name_label
    input_data_file = args.input_data_file_path

    if modelId == "default":
        modelId = str('a' + uuid4().hex[1:])
    
    if endpointId == "default":
        endpointId = str('a' + uuid4().hex[1:])

    if serviceEndpointDnsNameLabel == "default":
        serviceEndpointDnsNameLabel = str('a' + uuid4().hex[1:])
    
    if input_data_file == "default":
        input_data_file = os.path.join(pathlib.Path(__file__).parent.absolute(), "training_input.json")

    with open(input_data_file) as f:
        trainingUserInput = f.read()

    print(trainingUserInput)
    
    deploymentUserInput = json.dumps({"dns_name_label": serviceEndpointDnsNameLabel})
    
    ws = Workspace.from_config(path='.cloud/.azureml/', _file_name='test_workspace.json')
    exp = Experiment(ws, experimentName)
    
    ws = Workspace.from_config(path='.cloud/.azureml/', _file_name='test_workspace.json')
    run_id = utils.RunProject(azureml_workspace = ws, 
                                    entry_point = 'training', 
                                    experiment_name = experimentName, 
                                    parameters={'modelId': modelId, 
                                                'userInput': trainingUserInput}, 
                                    tags={})
    
    run = Run(exp, run_id)
    run.wait_for_completion(show_output = False)
    
    run_id = utils.RunProject(azureml_workspace = ws, 
                                    entry_point = 'deployment', 
                                    experiment_name = experimentName, 
                                    parameters={'modelId': modelId, 
                                                'userInput': deploymentUserInput, 
                                                'endpointId': endpointId}, 
                                    tags={})
    
    run = Run(exp, run_id)
    run.wait_for_completion(show_output = False)
    
    webservice = Webservice(ws, endpointId)
    
    if (webservice.compute_type == 'ACI'):
        aciWebservice = AciWebservice(ws, endpointId)
        print("The model {} was deployed to a ACI service endpoint.".format(modelId))
        print("The scoring URL is: {}".format(aciWebservice.scoring_uri))
        print("The primary authentication key is: {}".format(aciWebservice.get_keys()[0]))
    elif (webservice.compute_type == 'AKS'):
        aksWebservice = AksWebservice(ws, endpointId)
        print("The model {} was deployed to a AKS service endpoint.".format(modelId))
        print("The scoring URL is: {}".format(aksWebservice.scoring_uri))
        print("The primary authentication key is: {}".format(aksWebservice.get_keys()[0]))