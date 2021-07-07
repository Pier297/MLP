# Machine Learning 2020/21 Project
Andrea Laretto (ID: 619624) laretto@studenti.unipi.it<br>
Pier Paolo Tarasco (ID: 619622) p.tarasco@studenti.unipi.it <br>Master Degree in Computer Science<br>
ML course (654AA), Academic Year: 2020/2021<br>
Date: 09/07/2021<br>
Type of project: A

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

        python MLP/monk/monk.py

### Linux / MacOS

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

        python MLP/monk/monk.py

---

## Directory Structure

```
MLP
│   README.md
│   requirements.txt
│   setup.py
│   
├───MLP
│   │   ActivationFunctions.py
│   │   Adam.py
│   │   Gradient.py
│   │   GradientDescent.py
│   │   GridSearch.py
│   │   Layers.py
│   │   LossFunctions.py
│   │   Nesterov.py
│   │   Network.py
│   │   Plotting.py
│   │   RandomSearch.py
│   │   Utils.py
│   │   Validations.py
│   │   
│   ├───cup
│   │   │   cup.py
│   │   │   cup_hyperparameters.py
│   │   │   double_grid_search.py
│   │   │   load_cup.py
│   │   │   
│   │   ├───data
│   │   │       ML-CUP-test.csv
│   │   │       ML-CUP-training.csv
│   │   │       ML-CUP20-TR.csv
│   │   │       ML-CUP20-TS.csv
│   │   │       
│   │   ├───plots
│   │   │       
│   │   ├───results
│   │           
│   ├───monk
│   │   │   load_monk.py
│   │   │   monk.py
│   │   │   monk_hyperparameters.py
│   │   │   
│   │   ├───data
│   │   │       monks-1.test
│   │   │       monks-1.train
│   │   │       monks-2.test
│   │   │       monks-2.train
│   │   │       monks-3.test
│   │   │       monks-3.train
│   │   │       
│   │   ├───plots
│   │   │   ├───monk1
│   │   │   ├───monk2
│   │   │   └───monk3
│   │   ├───results
├───MLP.egg-info  
└───venv
```