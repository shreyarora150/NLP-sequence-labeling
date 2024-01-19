# Objectives

The learning objectives of this assignment are to:

1. obtain the numeric representation of neural network input and output
2. create, train and run a recurrent neural network 
3. initialize an embedding layer with pre-trained word embeddings

# Setup your environment and prepare data

First, carefully follow the *General Instructions for Programming Assignments*.

To install the libraries required for this assignment run:

    pip install -r requirements.txt
    
Clone the MeasEval repository:

    git clone https://github.com/harperco/MeasEval.git
    
Download GloVe embeddings and prepare the dataset by running:

    python prepare_data.py 

This script will take several minutes to run.
As a result, the `glove.6B.300d.txt` file should be unzipped into the `glove` folder. 
The `data/train.tsv`, `data/trial.tsv` and `data/eval.tsv` files should also be created.

# Grading

The grading distribution for this assignment is listed below:
- test_get_vocabulary = 16%
- test_format_examples = 16%
- test_create_model = 16%
- test_train_model = 16%
- test_make_predictions = 16%
- test_create_embedding_matrix = 10%
- test_create_model_with_embeddings = 10%