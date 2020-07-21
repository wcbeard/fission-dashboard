set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh

conda env create -f environment.yml

conda activate fis

jupyter labextension install @axlair/jupyterlab_vim @jupyterlab/toc @ryantam626/jupyterlab_code_formatter
jupyter lab build --name='fis'

cd notebooks
jupytext --to ipynb templ.py
cd ..

# jupyter labextension list
# pip install "jupyterlab_code_formatter==1.x.x"
# jupyter serverextension enable --py jupyterlab_code_formatter --sys-prefix