#!/usr/bin/env python3
'''
Created on May 11, 2024

@author: jrade
'''
import sys  # import the sys module for command-line arguments
import preprocesor  # import the preprocesor module for data preprocessing
import dataset  # import the dataset module for loading and manipulating datasets
import train  # import the train module for training machine learning models
# import traceback  # (commented out) import the traceback module for error handling and debugging

if __name__ == '__main__':  # ensure the code inside is only executed when the script is run directly
    if len(sys.argv) != 2:  # check if the number of command-line arguments is not exactly 2
        print("Usage: python (link unavailable) <db_file>")  # print an error message indicating the correct usage of the script
        sys.exit(1)  # exit the script with a non-zero exit code (indicating an error)

    db_file = sys.argv[1]  # retrieve the first command-line argument (the database file)
    print('Loading Dataset')  # print a message indicating that the dataset is being loaded
    train_data = dataset.load_dataset(db_file)  # load the dataset using the load_dataset function from the dataset module

    print('Running Preprocessor')  # print a message indicating that the data is being preprocessed
    train_data = preprocesor.preprocess_data(train_data)  # preprocess the data using the preprocess_data function from the preprocesor module

    print('Training Model')  # print a message indicating that the model is being trained
    train.train_model(train_data)  # train the model using the train_model function from the train module


