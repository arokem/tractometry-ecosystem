# hash:sha256:068af47628e285d3476e8d49a288056a7c4968a8a7cbb4ce9ba4015690a680b9
FROM registry.codeocean.com/codeocean/py-r:python3.12.4-R4.4.2-1.2204.0-JupyterLab4.3.0-RStudiorstudio-server-2024.09.0-375-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV TEMPLATEFLOW_HOME=/data
ENV DIPY_HOME=/data
ENV AFQ_HOME=/data

RUN conda install -y \
        jupyter==1.1.1 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    afqinsight==0.7.1 \
    cvxpy==1.6.0 \
    dash==2.18.2 \
    dipy==1.10 \
    ray==2.42.0 \
    tqdm \
    trx-python==0.3 \
    pyAFQ@git+https://github.com/tractometry/pyAFQ@main \
    fury 

RUN Rscript -e "install.packages(c('remotes', 'Hmisc'), repos='https://cloud.r-project.org'); remotes::install_github('tractometry/tractable')"
