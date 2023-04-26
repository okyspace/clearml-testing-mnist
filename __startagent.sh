#!/bin/bash
export HOME=/tmp ; 
export LOCAL_PYTHON=python3 ; 
$LOCAL_PYTHON -m pip install clearml-agent ; 
$LOCAL_PYTHON -m clearml_agent execute --full-monitoring --require-queue --id beebab6551f24a879887c5201ba11148