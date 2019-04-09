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

## [ Final Runs - Combo 2 ]
# [ f ]
#~./fashion.py 2 0.001 60 60 12345
# [ g ]
#~./fashion.py 2 0.001 45 50 12345
# [ h ]
./fashion.py 2 0.001 30 40 12345
# [ i ]
#~./fashion.py 2 0.01 20 30 12345
# [ j ]
#~./fashion.py 2 0.1 10 30 12345


## [ Combo 1 ]

#~python main.py


## EOF
