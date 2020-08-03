from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline
from azureml.core.runconfig import RunConfiguration
from azureml.core import Workspace

from luna import utils

import argparse

import os

def publish(azureml_workspace, entry_point, name, description, parameters={}):
    luna_config = utils.Init()
    if azureml_workspace:
        run_config = RunConfiguration.load(luna_config['azureml']['run_config'])

        arguments = utils.GetPipelineArguments(luna_config['MLproject'], entry_point, parameters)

        trainStep = PythonScriptStep(
            script_name=luna_config['code'][entry_point],
            arguments=arguments,
            inputs=[],
            outputs=[],
            source_directory=os.getcwd(),
            runconfig=run_config
        )

        pipeline = Pipeline(workspace=azureml_workspace, steps=[trainStep])
        published_pipeline = pipeline.publish(name=name, description=description)
        return published_pipeline.endpoint

if __name__ == "__main__":

    parser=argparse.ArgumentParser(description="Publish AML pipelines") 

    parser.add_argument('-training_pipeline_name', 
                        '--training_pipeline_name', 
                        help="The name of training pipeline", 
                        default="mytrainingpipeline",
                        type=str)  
                        
    parser.add_argument('-batch_inference_pipeline_name', 
                        '--batch_inference_pipeline_name', 
                        help="The name of batch inference pipeline", 
                        default="mybatchinferencepipeline",
                        type=str) 
                        
    parser.add_argument('-deployment_pipeline_name', 
                        '--deployment_pipeline_name', 
                        help="The name of deployment pipeline", 
                        default="mydeploymentpipeline",
                        type=str)
                        
    parser.add_argument('-aml_workspace_name', 
                        '--aml_workspace_name', 
                        help="The name of the target aml workspace", 
                        default="default",
                        type=str) 

    parser.add_argument('-subscription_id', 
                        '--subscription_id', 
                        help="The subscription id of the target aml workspace", 
                        default="default",
                        type=str) 

    parser.add_argument('-resource_group', 
                        '--resource_group', 
                        help="The resource group name of the target aml workspace", 
                        default="default",
                        type=str) 

    args=parser.parse_args()

    if args.aml_workspace_name == "default":
        ws = Workspace.from_config(path='.cloud/.azureml/', _file_name='test_workspace.json')
    else:
        ws = Workspace.get(name=args.aml_workspace_name, subscription_id=args.subscription_id, resource_group=args.resource_group)
    
    training_pipeline_name = args.training_pipeline_name
    batch_inference_pipeline_name = args.batch_inference_pipeline_name
    deployment_pipeline_name = args.deployment_pipeline_name

    print('publishing training pipeline')
    endpoint = publish(ws, 'training', training_pipeline_name, 'The training pipeline')
    print('training pipeline published with endpoint {}'.format(endpoint))
    
    print('publishing batchinference pipeline')
    endpoint = publish(ws, 'batchinference', batch_inference_pipeline_name, 'The batchinference pipeline')
    print('batchinference pipeline published with endpoint {}'.format(endpoint))
    
    print('publishing deployment pipeline')
    endpoint = publish(ws, 'deployment', deployment_pipeline_name, 'The deployment pipeline')
    print('deployment pipeline published with endpoint {}'.format(endpoint))
