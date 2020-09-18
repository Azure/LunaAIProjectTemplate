import mlflow
import pyodbc


#mlflow.set_tracking_uri("mssql+pyodbc://cloudsa:Yukon900Yukon900@eddidemo.sql.azuresynapse.net:1433/eddidemosqlpool?driver=ODBC+Driver+17+for+SQL+Server")

mlflow.set_tracking_uri("mssql+pyodbc://cloudsa:Yukon900Yukon900@brazilmedia.database.windows.net:1433/facetsgraph?driver=ODBC+Driver+17+for+SQL+Server")

#mlflow.set_tracking_uri("mssql+pymssql://cloudsa:Yukon900Yukon900@eddidemo-ondemand.sql.azuresynapse.net:1433/customerSurvey")

mlflow.projects.run(uri="https://github.com/allenwux/ml", entry_point="training")

