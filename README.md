## Quick running guide

### Windows

0. Download the source code and navigate, using Powershell, into the parent directory 'MLP', inside you should see folders: 'MLP', 'venv', ...

        cd $HOME$/MLP

1. Activate the virtual environment (Windows Powershell)

        .\venv\Scripts\Activate.ps1

2. In the root directory ($HOME$/MLP/), install dependencies

        pip install -e . 

3. Run an experiment

        python MLP/experiments/monk1.py

### Linux

0. Download the source code and navigate, using bash, into the parent directory 'MLP', inside you should see folders: 'MLP', 'venv', ...

        cd $HOME$/MLP

1. Activate the virtual environment (Bash)

        ./venv/Scripts/activate

2. In the root directory ($HOME$/MLP/), install dependencies

        pip install -e . 

3. Run an experiment

        python MLP/experiments/monk1.py

### MacOS

Follow the same steps as for linux