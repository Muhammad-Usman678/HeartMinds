# HeartMinds: Machine Learning for ECG Analysis

HeartMinds is a machine learning project designed to classify Electrocardiogram (ECG) signals using a Random Forest classifier. The project employs GridSearchCV for hyperparameter tuning to optimize the performance of the Random Forest model.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [ROC Curve](#roc-curve)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Electrocardiograms (ECGs) are vital tools in diagnosing and monitoring heart conditions. HeartMinds uses a Random Forest classifier to analyze and classify ECG signals, enhancing the accuracy and reliability of diagnoses.

## Dataset

The dataset used in this project is sourced from Kaggle. It contains ECG signals and their corresponding classifications. You can find more details about the dataset and download it from [Kaggle ECG Dataset](https://www.kaggle.com/).

## Installation
o get started with HeartMinds, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Muhammad-Usman678/HeartMinds.git
    cd HeartMinds
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the necessary dependencies:
    ```bash
    pip install numpy pandas scikit-learn matplotlib jupyter
    ```

4. Download the ECG dataset from Kaggle

## Usage

To train and evaluate the model, open the provided Jupyter Notebook `ECG.ipynb` and follow the instructions. The notebook performs the following steps:
- Loads the dataset
- Splits the data into training and test sets
- Performs GridSearchCV to find the best hyperparameters
- Trains the Random Forest model with the best parameters
- Evaluates the model on the test set
- Outputs the classification report and confusion matrix
- Plots the ROC curve

## Results

After running the cells in the notebook, you will see the best parameters and test accuracy. A sample output is shown below:

Best Parameters: {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 2}
Best Score: 0.92
Test Accuracy: 0.89

## ROC Curve

The ROC curve illustrates the performance of the classifier. To generate the ROC curve, run the corresponding cell in the Jupyter Notebook.

## Contributing

Contributions are welcome! 

## License

This project is licensed under the MIT License.
