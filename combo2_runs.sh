#!/usr/bin/env bash


# >>> conda init >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$(CONDA_REPORT_ERRORS=false '/home/ivo/.opt/Miniconda3/bin/conda' shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then
    \eval "$__conda_setup"
else
    if [ -f "/home/ivo/.opt/Miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/ivo/.opt/Miniconda3/etc/profile.d/conda.sh"
        CONDA_CHANGEPS1=false conda activate base
    else
        \export PATH="/home/ivo/.opt/Miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda init <<<

conda activate cs985


#./main_mnist.py 2 0.001 5 32 12345
#./main_mnist.py 2 0.001 10 60 12345
./main_mnist.py 2 0.001 20 60 12345
