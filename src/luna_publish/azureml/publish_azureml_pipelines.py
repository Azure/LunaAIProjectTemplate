
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline
from azureml.core.runconfig import RunConfiguration
from azureml.core import Workspace

from luna import utils

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
    ws = Workspace.from_config(path='.cloud/.azureml/', _file_name='test_workspace.json')

    print('publishing training pipeline')
    endpoint = publish(ws, 'training', 'lunaaitrainingpipeline', 'The training pipeline')
    print('training pipeline published with endpoint {}'.format(endpoint))
    
    print('publishing batchinference pipeline')
    endpoint = publish(ws, 'batchinference', 'lunaaibatchinferencepipeline', 'The batchinference pipeline')
    print('batchinference pipeline published with endpoint {}'.format(endpoint))
    
    print('publishing deployment pipeline')
    endpoint = publish(ws, 'deployment', 'lunaaideploymentpipeline', 'The deployment pipeline')
    print('deployment pipeline published with endpoint {}'.format(endpoint))
