# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
 The prediction task was to determine whether a person makes over $50,000 a year. The GradientBoostingClassifier with the optimized hyperparameters in scikit-learn, and the tuning was realized using the GridSearchCV. The optimal parameters uses are:
    - learning_rate: 1.0
    - max_depth: 10
    - min_samples_split: 100
    - n_estimators: 30 Model is saved as pickle file in the model folder. All the training steps and metrics are logged in the journal.log file.

## Intended Use
The model is intended to be used to predict the salary of an individual based on a some attributes. This prediction will be used for students, academics and research.

## Training Data
The Census Income Dataset was obtained as a csv from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income) 

## Evaluation Data
The proportion of the dataset that was set aside for model evaluation was 20%. 
The categorical features and the target label had transformation applied to them using the One Hot Encoder and label binarizer fitted on the train set.

## Metrics
The classification performance is evaluated using precision, recall and fbeta metrics. 
The model achieves below scores using the test set: 
    - precision: 0.7590
    - recall: 0.6429
    - fbeta: 0.6961
   
## Ethical Considerations
The dataset should not be considered as a fair representation of the salary distribution and should not be used to assume salary level of certain population categories.

## Caveats and Recommendations
Extraction was done from the 1994 Census database. The dataset is a outdated sample and cannot adequately be used as a statistical representation of the population. It is recommended to use the dataset for training purpose on ML classification or related problems.