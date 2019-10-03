
#! /bin/bash
eval "$(/afs/cern.ch/work/j/jkiesele/public/conda_env/miniconda3/bin/conda shell.bash hook)"
THISDIR=`pwd`
cd /afs/cern.ch/work/j/jkiesele/public/conda_env/DeepJetCore
source env.sh
cd $THISDIR
export HGCALL1IMAGES=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P)
export DEEPJETCORE_SUBPACKAGE=$HGCALL1IMAGES

cd $HGCALL1IMAGES
export PYTHONPATH=$HGCALL1IMAGES/modules:$PYTHONPATH
export PYTHONPATH=$HGCALL1IMAGES/modules/datastructures:$PYTHONPATH
export PATH=$HGCALL1IMAGES/scripts:$PATH
