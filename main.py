import pandas as pd
import numpy as np
import argparse
import pickle
from task1 import task1_train, task1_test
from task2 import task2_train, task2_test
from task3 import task3_train, task3_test
from merge_tasks import merge_results

parser = argparse.ArgumentParser(description = 'parse this')
parser.add_argument('--retrain_model', action='store_true', default=False, dest='retrain_model', help='To retrain the models set this parameter')
args = parser.parse_args()

# retrain models
if args.retrain_model == True:
    print('You have choosen to make predictions on newly trained models!')
    print('This will take longer and may potentially lead to different results!')
    print("If you want to use the pretrained models don't use the optional parameter --retrain_model!")
    print('')

    #task 1
    print('training for task 1')
    model_1 = task1_train()

    with open('saved_models/new_model_task1.pkl', 'wb') as file:
        pickle.dump(model_1, file)

    # task 2
    print('training for task 2')
    model_2 = task2_train()

    with open('saved_models/new_model_task2.pkl', 'wb') as file:
        pickle.dump(model_2, file)

    # task 3
    print('training for task 3')
    model_3 = task3_train()

    with open('saved_models/new_model_task3.pkl', 'wb') as file:
        pickle.dump(model_3, file)


    #here model loading starts
    with open('saved_models/new_model_task1.pkl', 'rb') as file:
        model_1_loaded = pickle.load(file)

    with open('saved_models/new_model_task2.pkl', 'rb') as file:
        model_2_loaded = pickle.load(file)

    with open('saved_models/new_model_task3.pkl', 'rb') as file:
        model_3_loaded = pickle.load(file)


# use pretrained models
else:
    print('You have choosen to make predictions with the pretrained models!')
    print('If you want to train from scratch use the optional parameter --retrain_model')
    print('')

    # here model loading starts
    with open('saved_models/pretrained_model_task1.pkl', 'rb') as file:
        model_1_loaded = pickle.load(file)

    with open('saved_models/pretrained_model_task2.pkl', 'rb') as file:
        model_2_loaded = pickle.load(file)

    with open('saved_models/pretrained_model_task3.pkl', 'rb') as file:
        model_3_loaded = pickle.load(file)


print("prediction is on it's way...")
task1_test(model_1_loaded)
task2_test(model_2_loaded)
task3_test(model_3_loaded)

merge_results()

print('Prediction finished! Results can be found in "submission.csv" and "submission.zip')