fission dashboard
==============================

The significant components of this repo so far are

- query to summarize fission branches at the daily level
    - see `load_hist_aggs.sh` below
    - summed histograms saved to
    `moz-fx-data-shared-prod.analysis.wbeard_fission_test_dirp`
- jupyterlab dev environment for the dashboard (`notebooks/hist_plots.ipynb`)
    - Downloads summarized histograms
    - Creates credible intervals with Dirichlet model (`fis.models.hist.est_statistic` using geometric mean)
- script to convert jupyter notebook into a dashboard ()

Build conda environment, start jupyter kernel.

## Create env
```sh
conda env create --file env_fis.yaml
```

## Update histogram table summary
The file has the command:
```sh
python -m fis.data.load_agg_hists \
    --sub_date_start='2020-06-01' \
    --sub_date_end='2020-07-01' \
    --sample=100 \
    --ret_sql=False 
```

You'll need to update the `sub_date_start` and `sub_date_end` commands (or
better yet, figure out how to get this to update itself).

```sh
bash fis/data/load_hist_aggs.sh
```

## Launch jupyter dev environment
```
conda activate fission_db
jupyter lab
```

## Convert notebook into dashboard

```sh
cd reports
bash build.sh
```

* saves output to `reports/fisbook/_build/html/hist_plots.html`
* `_config.yml` is an important file


## Other resources

A starting point for a crash rates query can be found at
`fis/data/crash_rates.sql`.