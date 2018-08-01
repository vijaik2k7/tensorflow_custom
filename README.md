
## Summary
This program builts a custom Tensorflow Estimator API which is trained and tested on a retail banking application.
## Program
Q2_Estimator_API.py (for training and running locally without serving)

trainer_custom/model.py (file that contains the tf model and function to serve the model on the Cloud ML Engine)

trainer_custom/task.py (experiment file that trains, evaluates and serves the model on the Cloud ML Engine)

## Python version
3.5, for compatibility with TensorFlow
## Packages
- os
- numpy
- tensorflow
- scikit-learn
## Input 
- Training set: bank-full-v4-train.csv
- Test set: bank-full-v4-test.csv
## Output
- Creates a custom Tensorflow estimator model each time the program is executed.
- Accuracy results (overall accuracy on the test set, and results for the new cases listed in the code) are sent to the console via Python 'print'.
## Instructions

#### For running files locally without serving:

1. Ensure that the input programs are in the current directory
2. Ensure that the Python packages listed above are installed (use requirements.txt file)
3. Run the Python program
4. Informational messages will appear while the program runs
5. The program will print the overall accuracy as a percentage, followed by the predicted results of the two new test case

#### For Training on Cloud ML Engine:

Git clone the repository (git clone https://github.com/vijaik2k7/deloitte_ml_spec.git). Navigate to Q2 - Estimator API Model. Follow steps below:

`export GCS_TRAIN_FILE=gs://ml_spec/data/bank-full-v4-train.csv`

`export GCS_EVAL_FILE=gs://ml_spec/data/bank-full-v4-test.csv`

`export SCALE_TIER=BASIC`

`export JOB_NAME=bank_custom_estimator_<timestamp>`

`export GCS_JOB_DIR=gs://ml_spec/bank_custom_estimator_<timestamp>`

`gcloud ml-engine jobs submit training $JOB_NAME --scale-tier $SCALE_TIER \
    --runtime-version 1.4 --job-dir $GCS_JOB_DIR \
    --module-name trainer_custom.task --package-path trainer_custom/ \
    --region us-central1 \
    -- --train-steps 10000 --train-files $GCS_TRAIN_FILE --eval-files $GCS_EVAL_FILE --eval-steps 100`


#### For Online Prediction (using currently hosted model):

`gcloud ml-engine predict --model bank_custom_predict --version v1 --json-instances test.json`
