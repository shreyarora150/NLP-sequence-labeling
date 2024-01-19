import pytest
import nbimporter
import json
import pandas as pd
import pickle as pkl
import numpy as np
from pandas import DataFrame
from tensorflow.keras.models import load_model, Sequential
from tensorflow.random import set_seed
from tensorflow.keras.utils import set_random_seed
from numpy.testing import assert_array_equal, assert_allclose
from dataclasses import dataclass


from sequencelabeling import (get_vocabulary, integrate_sentences, format_examples,
                              create_model, train_model, make_predictions,
                              create_embedding_matrix, create_model_with_embeddings)


set_seed(42)
set_random_seed(42)


@dataclass
class Shared:
    data: DataFrame
    target_vocab: list
    target_word2idx: dict
    target_word2idx_incomplete: dict
    target_labels: list
    target_label2idx: dict
    target_x: np.array
    target_y: np.array
    target_predictions: np.array
    embedding_index: dict
    target_embedding_matrix: np.array
    target_model: Sequential
    target_model_trained: Sequential
    target_model_with_embs: Sequential
    maxlen: int
    epochs: int
    batch_size: int
    embedding_dim: int
    rnn_units: int
    vocab_size: int
    label_size: int


@pytest.fixture(scope="session")
def shared():
    data = pd.read_csv("test_utils/data.tsv", sep="\t", encoding="utf8").dropna()
    with open("test_utils/vocab.json", encoding="utf8") as jfile:
        target_vocab = json.load(jfile)
    with open("test_utils/word2idx.json", encoding="utf8") as jfile:
        target_word2idx = json.load(jfile)
    with open("test_utils/word2idx_incomplete.json", encoding="utf8") as jfile:
        target_word2idx_incomplete = json.load(jfile)
    with open("test_utils/labels.json", encoding="utf8") as jfile:
        target_labels = json.load(jfile)
    with open("test_utils/label2idx.json", encoding="utf8") as jfile:
        target_label2idx = json.load(jfile)

    with open("test_utils/x.pkl", "rb") as pklfile:
        target_x = pkl.load(pklfile)
    with open("test_utils/y.pkl", "rb") as pklfile:
        target_y = pkl.load(pklfile)
    with open("test_utils/predictions.pkl", "rb") as pklfile:
        target_predictions = pkl.load(pklfile)
    with open("test_utils/embedding_index.pkl", "rb") as pklfile:
        embedding_index = pkl.load(pklfile)
    with open("test_utils/embedding_matrix.pkl", "rb") as pklfile:
        embedding_matrix = pkl.load(pklfile)

    target_model = load_model("test_utils/model.h5")
    target_model_trained = load_model("test_utils/model_trained.h5")
    target_model_with_embs = load_model("test_utils/model_with_embs.h5")

    maxlen = 130
    epochs = 1
    batch_size = 64
    embedding_dim = 10
    rnn_units = 10
    vocab_size = len(target_vocab)
    label_size = len(target_labels)

    return Shared(data,
                  target_vocab, target_word2idx, target_word2idx_incomplete,
                  target_labels, target_label2idx,
                  target_x, target_y, target_predictions,
                  embedding_index, embedding_matrix,
                  target_model, target_model_trained, target_model_with_embs,
                  maxlen, epochs, batch_size, embedding_dim, rnn_units,
                  vocab_size, label_size)


def assert_equal_compilers(model_1, model_2):
    assert model_1.compiled_loss._losses == model_2.compiled_loss._losses[0].name
    assert model_1.compiled_metrics._metrics == [model_2.compiled_metrics._metrics[0][0].name]
    assert type(model_1.optimizer) == type(model_2.optimizer)


def assert_equal_layers(layer_1, layer_2):
    assert layer_1['class_name'] == layer_2['class_name']
    assert set(layer_1['config']) == set(layer_2['config'])
    for config in layer_2['config']:
        if config == "name" or config == "layer":
            continue
        if config == "embeddings_initializer":
            assert "embeddings_initializer" in layer_1['config']
            assert layer_1['config']["embeddings_initializer"]["class_name"] == layer_2['config'][config]["class_name"]
        else:
            assert layer_1['config'].get(config, None) == layer_2['config'][config]
    if 'layer' in layer_2['config']:
        assert 'layer' in layer_1['config']
        layer_1_layer = layer_1['config']['layer']
        layer_2_layer = layer_2['config']['layer']
        assert_equal_layers(layer_1_layer, layer_2_layer)

    
def assert_equal_models(model_1, model_2):
    assert type(model_1) == type(model_2)
    assert_equal_compilers(model_1, model_2)
    for layer_1, layer_2 in zip(model_1.get_config()['layers'], model_2.get_config()['layers']):
        assert_equal_layers(layer_1, layer_2)


def test_get_vocabulary(shared):
    test_vocab, test_word2idx, test_labels, test_label2idx = get_vocabulary(shared.data)
    assert test_vocab[:2] == shared.target_vocab[:2]
    assert set(test_vocab) == set(shared.target_vocab)
    target_word2idx = {(v, i) for i, v in enumerate(test_vocab)}
    assert test_word2idx.items() == target_word2idx
    assert test_labels[0] == shared.target_labels[0]
    assert set(test_labels) == set(shared.target_labels)
    target_label2idx = {(v, i) for i, v in enumerate(test_labels)}
    assert test_label2idx.items() == target_label2idx

   
def test_format_examples(shared):
    examples = integrate_sentences(shared.data)
    test_x, test_y = format_examples(examples, shared.target_word2idx_incomplete,
                                     shared.target_label2idx, shared.maxlen)
    assert_array_equal(test_x, shared.target_x)
    assert_array_equal(test_y, shared.target_y)

   
def test_create_model(shared):
    test_model = create_model(shared.vocab_size, shared.label_size,
                             shared.maxlen, shared.embedding_dim, shared.rnn_units)
    assert_equal_models(test_model, shared.target_model)


def test_train_model(shared):
    test_model_trained = load_model("test_utils/model.h5")
    train_model(test_model_trained, shared.target_x, shared.target_y,
                shared.target_x, shared.target_y,
                shared.batch_size, shared.epochs)
    assert_allclose(test_model_trained.weights[-1].numpy(), shared.target_model_trained.weights[-1].numpy())
   

def test_make_predictions(shared):
    test_predictions = make_predictions(shared.target_model_trained, shared.target_x, shared.batch_size)
    assert_array_equal(test_predictions, shared.target_predictions)
   

def test_create_embedding_matrix(shared):
    test_embedding_matrix = create_embedding_matrix(shared.embedding_index, shared.target_word2idx,
                                                    shared.vocab_size, shared.embedding_dim)
    assert_array_equal(test_embedding_matrix, shared.target_embedding_matrix)


def test_create_model_with_embeddings(shared):
    test_model_with_embs = create_model_with_embeddings(shared.vocab_size, shared.label_size,
                                                        shared.maxlen, shared.embedding_dim, shared.rnn_units,
                                                        shared.target_embedding_matrix)
    assert_equal_models(test_model_with_embs, shared.target_model_with_embs)
    assert_allclose(test_model_with_embs.get_weights()[0], shared.target_model_with_embs.get_weights()[0])
