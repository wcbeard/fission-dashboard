#!/usr/bin/env bash
set -o xtrace

DEST_PROJ="moz-fx-data-shared-prod"
DEST_TABLE="$DEST_PROJ:analysis.wbeard_fission_test_dirp"

# bq rm -f -t "moz-fx-data-shared-prod:analysis.wbeard_fission_test_dirp"

python -m fis.data.load_agg_hists \
    --sub_date_start='2020-06-01' \
    --sub_date_end='2020-07-01' \
    --sample=100 \
    --ret_sql=False 

# python -m fis.data.load_agg_hists \
#     --sub_date_start='2020-06-01' \
#     --sub_date_end='2020-07-01' \
#     --sample=100 \
#     --ret_sql=False 

    # | \
    # bq query --use_legacy_sql=false \
    # --destination_table="$DEST_TABLE" \
    # --replace=true \
    # --project_id=moz-fx-data-derived-datasets