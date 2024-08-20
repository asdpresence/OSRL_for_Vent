# OSRL_for_Vent

An offline safe reinforcement learning framework for Mechanical ventilation.
Official code for "Offline Safe Reinforcement Learning for Mechanical Ventilation via Dynamic Safety Violation Penalties"

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Installation](#installation)
* [Processing Data](#processing-data)
* [Training policies](#training-policies)
* [Evaluating policies](#evaluation)

Note: M1 chip is still not supported in many packages so there might be compatibility issues if you are running using M1 chip.

<!-- ABOUT THE PROJECT -->
## About the Project 
Mechanical ventilation is a key form of life support for patients with pulmonary impairment. Healthcare workers are required to continuously adjust ventilator settings for each patient, a challenging and time consuming task. Hence, it would be beneficial to develop an automated decision support tool to optimize ventilation treatment. We present DeepVent, a Conservative Q-Learning (CQL) based offline Deep Reinforcement Learning (DRL) agent that learns to predict the optimal ventilator parameters for a patient to promote 90 day survival. We design a clinically relevant intermediate reward that encourages continuous improvement of the patient vitals as well as addresses the challenge of sparse reward in RL. We find that DeepVent recommends ventilation parameters within safe ranges, as outlined in recent clinical trials. The CQL algorithm offers additional safety by mitigating the overestimation of the value estimates of out-of-distribution states/actions. We evaluate our agent using Fitted Q Evaluation (FQE) and demonstrate that it outperforms physicians from the MIMIC-III dataset.

<!-- INSTALLATION -->
## Installation
1. Go to the parent directory 
```
cd OSRL_for_Vent
```
2. Create and activate virtual environment 

Linux:
```sh
python -m venv env
source env/bin/activate
```
Windows (run this from the command prompt NOT powershell):
```
python -m venv env
.\env\Scripts\activate.bat
```
3. install the required libraries
```
pip install -r requirements.txt 
```
4. install the root package (Run this from the ROOT directory of the repository, i.e. the directory which contains "examples", "evaluation", etc.)
```
pip install -e .
```
5. install pytorch with CUDA capabilities

(This step is not mandatory; you can also train policies using the CPU.)
Go to https://pytorch.org/get-started/locally/ and follow the instructions to install PyTorch with CUDA capabilities (using **pip**) on your OS and with your CUDA version.
<!-- PREPROCESSING DATA -->
## Processing Data
1. Data Imputation
2. Data Cleaning(remove outlier, normalization)
3. Build Dataset
4. Build Trajectories

To run the data preprocessing: 
```
cd preprocessing
python impute.py
python preprocess_mimic3_data.py
python processed_mimic3_episodes.py
```
Note: The data dictionary and patient trajectory datasets are both essential.


<!-- TRAINING POLICIES -->
## Training
1. For example, to train the bcql method, simply run by overriding the default parameters: 
```
python examples/vent/train_vent_bcql.py --task mimic3 --device cuda:0 --name bcql ...
```
the config file and the logs during training will be written to logs\ folder and the training plots can be viewed online using Wandb.
<!-- EVALUATING POLICIES -->
## Evaluation

To evaluate a policy, for example, a BCQ-L policy, simply do:
```
cd evaluation
python BCQL_predicted_actions.py
python FQE_train.py
python FQE_bcql.py
```
After having all the trained models, you can observe the predicted action distribution using
```
python action_distribution_plot.py
```
