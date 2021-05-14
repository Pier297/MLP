Quick running guide:

0. Download the source code and navigate from your favorite terminal into the parent directory 'MLP', inside you should see folders: 'MLP', 'venv', ...

        cd $HOME$/MLP

1. Activate the virtual environment (Windows Powershell)

        .\venv\Scripts\Activate.ps1

2. In the root directory ($HOME$/MLP/)

        pip install -e . 

3. Run an experiment

        python MLP/experiments/monk1.py