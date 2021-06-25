## Quick running guide

### Windows

0. Download the source code and navigate, using Powershell, into the parent directory 'MLP', inside you should see folders: 'MLP', 'venv', ...

        cd $HOME$/MLP

1. Create a new python environment

        python -m venv ./venv/

2. Activate the virtual environment (Windows Powershell)

        .\venv\Scripts\Activate.ps1

3. Install requirements

        pip install -r requirements.txt

4. Build package

        pip install -e .

5. Run an experiment

        python MLP/monk/monk1.py

### Linux

0. Download the source code and navigate, using bash, into the parent directory 'MLP', inside you should see folders: 'MLP', 'venv', ...

        cd $HOME$/MLP

1. Create a new python environment

        python -m venv ./venv/

2. Activate the virtual environment (bash)

        ./venv/Scripts/activate

3. Install requirements

        pip install -r requirements.txt

4. Build package

        pip install -e .

5. Run an experiment

        python MLP/monk/monk1.py

### MacOS

Follow the same steps as for linux