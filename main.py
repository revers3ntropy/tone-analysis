import tensorflow as tf
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random as jrandom
from jax.nn import selu
import time
import csv
import matplotlib.pyplot as plt

TRAINING_DATA_PATH = 'data/training.csv'
TEST_DATA_PATH = 'data/validation.csv'

SENTENCE_LEN = 128
LAYER_SIZES = [SENTENCE_LEN, 512, 512, 128, 16, 1]
STEP_SIZE = 0.00001
NUM_EPOCHS = 1000
BATCH_SIZE = 256


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
    keys = jrandom.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def log(message: str):
    print(f'[LOG] {message}')


def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = jrandom.split(key)
    return scale * jrandom.normal(w_key, (n, m)), scale * jrandom.normal(b_key, (n,))


def predict(params, sentences):
    # per-example predictions
    activations = sentences
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = selu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    # take the first element as the output, as there is one node in the output layer
    return logits[0]


batched_predict = vmap(predict, in_axes=(None, 0))


@jit
def accuracy(params, sentences, targets):
    # only checks if it correctly determines positive or negative
    target_labels = targets > 0.5
    predicted_labels = batched_predict(params, sentences) > 0.5
    return jnp.mean(predicted_labels == target_labels)


@jit
def loss(params, sentences, targets):
    preds = batched_predict(params, sentences)
    diff = preds - targets
    return jnp.sum(diff * diff) / preds.shape[0]


@jit
def update(params, x, y):
    grads = grad(loss)(params, x, y)
    return [(w - STEP_SIZE * dw, b - STEP_SIZE * db)
            for (w, b), (dw, db) in zip(params, grads)]


words = {}


def add_word(word: str):
    if word not in words:
        # +1 so 0 is null
        words[word] = len(words) + 1
    return words[word]


def tokenise(sentence: str) -> jnp.ndarray:
    tokens: list[int] = []
    current_word = ''
    for char in sentence:
        if char.isalnum():
            current_word += char
        elif char == ' ' and current_word:
            tokens.append(add_word(current_word))
            current_word = ''

    if current_word:
        tokens.append(add_word(current_word))

    if SENTENCE_LEN - len(tokens) < 0:
        return jnp.array(tokens[:SENTENCE_LEN], dtype=jnp.int32)

    return jnp.pad(jnp.array(tokens, dtype=jnp.int32), (0, SENTENCE_LEN - len(tokens)), mode='constant')


def read_dataset(path: str) -> list[list[str, float]]:
    with open(path) as f:
        reader = csv.reader(f)
        return list(map(lambda row: [row[0], float(row[1])], reader))


def add_mutated_data(data: list[list[jnp.ndarray, float]]) -> list[list[jnp.ndarray, float]]:
    return data


def tokenise_data(data: list[list[str, float]]) -> list[list[jnp.ndarray, float]]:
    return list(map(lambda x: [tokenise(x[0]), x[1]], data))


def write_results(params, training_accs: list[float], test_accs: list[float]):
    with open('params.txt', 'w') as f:
        f.write(params.__str__())

    with open('training_accs.txt', 'w') as f:
        f.write(training_accs.__str__())

    with open('test_accs.txt', 'w') as f:
        f.write(test_accs.__str__())


def plot_results(training_accs: list[float], test_accs: list[float]):
    plt.plot(training_accs, label='Training Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def main():
    total_start = time.time()

    # Ensure TF does not see GPU and grab all GPU memory
    tf.config.set_visible_devices([], device_type='GPU')

    params = init_network_params(LAYER_SIZES, jrandom.key(0))

    train_data = tokenise_data(read_dataset(TRAINING_DATA_PATH))
    test_data = tokenise_data(read_dataset(TEST_DATA_PATH))

    train_data_sentences = jnp.array(list(map(lambda x: x[0], train_data)), dtype=jnp.int32)
    train_labels = jnp.array(list(map(lambda x: x[1], train_data)), dtype=jnp.float32)
    train_sentences = jnp.reshape(train_data_sentences, (len(train_labels), SENTENCE_LEN))

    test_data_sentences = jnp.array(list(map(lambda x: x[0], test_data)), dtype=jnp.int32)
    test_labels = jnp.array(list(map(lambda x: x[1], test_data)), dtype=jnp.float32)
    test_sentences = jnp.reshape(test_data_sentences, (len(test_labels), SENTENCE_LEN))

    log(f'Loaded data in {time.time() - total_start:0.3f} seconds')
    log(f'Train: {train_sentences.shape=}, {train_labels.shape=}')
    log(f'Test: {test_sentences.shape=}, {test_labels.shape=}')

    training_accs = []
    test_accs = []

    training_start = time.time()

    try:
        for epoch in range(NUM_EPOCHS):
            start_time = time.time()
            for i in range(0, len(train_sentences), BATCH_SIZE):
                x, y = train_sentences[i:i + BATCH_SIZE], train_labels[i:i + BATCH_SIZE]
                x = jnp.reshape(x, (len(x), SENTENCE_LEN))
                params = update(params, x, y)
            epoch_time = time.time() - start_time

            train_acc = accuracy(params, train_sentences, train_labels)
            test_acc = accuracy(params, test_sentences, test_labels)

            if epoch % 1 == 0:
                log("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
                log(f"Training set accuracy {train_acc * 100:0.3f}%")
                log(f"Test set accuracy {test_acc * 100:0.3f}%")
            training_accs.append(float(train_acc))
            test_accs.append(float(test_acc))
    except KeyboardInterrupt:
        log('Training interrupted')

    log(f'Finished training in {time.time() - training_start:0.3f} seconds')

    write_results(params, training_accs, test_accs)
    plot_results(training_accs, test_accs)

    log('Results written to files')
    log(f'Total time: {time.time() - total_start:0.3f} seconds')


if __name__ == '__main__':
    main()
