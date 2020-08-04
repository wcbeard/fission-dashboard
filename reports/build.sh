set +x
cp ../notebooks/hist_plots.ipynb fisbook
python control_tags.py fisbook/hist_plots.ipynb
jupyter-book build fisbook