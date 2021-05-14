# Setup
The setup to find the code of this project goes as follows:
The folder named FootballML contains three folders, classifiers, dataset, and individual results. The dataset folder contains the python files to clean all of the data. The classifier folder contains two folders, ensemble, where the ensemble code is, and individual were all of the developers code of their own algorithm is. Finally in the individual results folder is where the jupyter notebook files are where you can see the indiviual algorithms' results.

# FootballML
Ensemble Learning classifier to predict the winner of future NFL games using historical game data.

# Running The Ensemble Classifier
Run the notebook ensemble_classifier.ipynb and execute all cells to run the classifier and view the results.

# Dataset
Sportsipy https://sportsipy.readthedocs.io/en/stable/

# Development
| Developer          | Model                          |
|--------------------|--------------------------------|
| Caleb Williams     | Random Forest classifier       |
| Alexander Townsend | Neural Network classifier      |
| George Fafard      | SVM classifier                 |
| Alex Farrell       | Logistic Regression classifier |

# Invidividual Classifier Results
Run the .ipynb notebook files Found in FootballML/Individual Results to see individual results

# Reproduction
The classifiers are defined in FootballML/Classifiers/Individual or for ensemble, in FootballML/Classifiers/Ensemble.
There, the code is stored in .py files and the necessary HyperParams are available for reproduction. Data was split and tuned using
the file found in FootballML/Dataset/cleaned_data.py.

