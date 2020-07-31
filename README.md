fis
==============================

* [Begin to scope Fission dashboard](https://jira.mozilla.com/browse/DS-702)


Create docker container:
```
docker build -t ds_702_prod .
```

This will build a container based on `Dockerfile`. When you update a line in the
Dockerfile, it will rebuild the container from that line onward. Put time
consuming lines earlier in the file.


```
docker run -v=$HOME/.config/gcloud:/root/.config/gcloud -v
~/repos/fis/fission:/fission -it ds_702_prod /bin/bash
docker run -v ~/repos/fis/fission:/fission -it ds_702_prod /bin/bash


docker run -it -v ~/.R:/root/.R -v ~/.config:/root/.config /bin/bash
docker run -it /bin/bash
```
run 


- version

worthy goals
- python bq read/write
- gsutil read/write
- bq cmdline


# Resources
* [BQ creds]
(https://docs.telemetry.mozilla.org/cookbooks/bigquery/access.html#api-access)

# Bigquery IO

- `client.load_table_from_dataframe(df, table)`
    - no [{'key': 1, 'value': 1.2}]
    - ArrowTypeError: Unknown list item type: struct<key: int64, value: double>