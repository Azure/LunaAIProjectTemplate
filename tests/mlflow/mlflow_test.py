import mlflow
import pyodbc


#mlflow.set_tracking_uri("mssql+pyodbc://cloudsa:Yukon900Yukon900@eddidemo.sql.azuresynapse.net:1433/eddidemosqlpool?driver=ODBC+Driver+17+for+SQL+Server")

mlflow.set_tracking_uri("mssql+pyodbc://cloudsa:Yukon900Yukon900@brazilmedia.database.windows.net:1433/facetsgraph?driver=ODBC+Driver+17+for+SQL+Server")

#mlflow.set_tracking_uri("mssql+pymssql://cloudsa:Yukon900Yukon900@eddidemo-ondemand.sql.azuresynapse.net:1433/customerSurvey")

mlflow.projects.run(uri="https://github.com/allenwux/ml", entry_point="training")

"""
import pyodbc
server = 'brazilmedia.database.windows.net'
database = 'facetsgraph'
username = 'cloudsa'
password = 'Yukon900Yukon900'
driver= '{ODBC Driver 17 for SQL Server}'
cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()
cursor.execute("SELECT TOP 20 * FROM facetname")
row = cursor.fetchone()
while row:
    print (str(row[0]))
    row = cursor.fetchone()
"""