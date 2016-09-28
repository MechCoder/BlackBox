export PATH="$HOME/miniconda3/bin:$PATH"
source activate testenv
export SKOPT_HOME=$(pwd)

# Generating documentation
mkdir -p ${HOME}/doc/skopt/notebooks
for nb in examples/*ipynb; do
    jupyter nbconvert --ExecutePreprocessor.timeout=1024 --execute "$nb" --to markdown |& tee nb_to_md.txt
    traceback=$(grep "Traceback (most recent call last):" nb_to_md.txt)
    if [[ $traceback ]]; then
        exit 1
    fi
    support_files=$(grep -oP "Support files will be in \K.+$" nb_to_md.txt)
    if [[ $support_files ]]; then
        cp -r ${SKOPT_HOME}/examples/$support_files ${HOME}/doc/skopt/notebooks
    fi
done

cd ~
cp ${SKOPT_HOME}/examples/*md ${HOME}/doc/skopt/notebooks
python ${SKOPT_HOME}/build_tools/circle/make_doc.py --overwrite --html --html-dir ./doc --template-dir ${SKOPT_HOME}/build_tools/circle/templates --notebook-dir ./doc/skopt/notebooks skopt
cp -r ./doc ${CIRCLE_ARTIFACTS}
