from google.cloud import bigquery
print([d.dataset_id for d in bigquery.Client().list_datasets()])
