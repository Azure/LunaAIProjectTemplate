from azureml.core import model
from azureml.core.webservice.aci import AciWebservice
from azureml.core.webservice.aks import AksWebservice
from luna import utils
from azureml.core import Workspace, Experiment, Run
from azureml.core.webservice import Webservice
import json


experimentName = 'train_model'
modelId = 'sklearn-lin-reg-iris'
endpointId = 'sklearn-lin-reg-iris'
serviceEndpointDnsNameLabel = 'sklearn-lin-reg-iris'

trainingUserInput = json.dumps({
        "trainingDataSource": "https://xiwutestai.blob.core.windows.net/lunav2/iris/iris.csv?st=2020-07-22T17%3A19%3A10Z&se=2027-10-12T17%3A19%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=7c%2BaoI8QtdepDHKqJqjjljdBUyDyuL8wbKol2Kn7xaI%3D",
        "description": "Iris prediction"
    })

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


