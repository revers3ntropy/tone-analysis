import tensorflow as tf
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random as jrandom
from jax.scipy.special import logsumexp
import time
import random
import matplotlib.pyplot as plt

DATA_PATH = './tone_v1.txt'

TONES = {
    'appreciative': 0,
    'cautionary': 1,
    'diplomatic': 2,
    'direct': 3,
    'informative': 4,
    'inspirational': 5,
    'thoughtful': 6,
    'witty': 7,
    'absurd': 8,
    'accusatory': 9,
    'acerbic': 10,
    'admiring': 11,
    'aggressive': 12,
    'aggrieved': 13,
    'altruistic': 14,
    'ambivalent': 15,
    'amused': 16,
    'angry': 17,
    'animated': 18,
    'apathetic': 19,
    'apologetic': 20,
    'ardent': 21,
    'arrogant': 22,
    'assertive': 23,
    'belligerent': 24,
    'benevolent': 25,
    'bitter': 26,
    'callous': 27,
    'candid': 28,
    'caustic': 29,
}

TRAIN_RATIO = 0.9
NUM_LABELS = len(TONES)

SENTENCE_LEN = 32
LAYER_SIZES = [SENTENCE_LEN, 256, 32, 30]
STEP_SIZE = 0.002
NUM_EPOCHS = 2000
BATCH_SIZE = 64


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
    keys = jrandom.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def log(message: str):
    print(f'[LOG] {message}')


def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = jrandom.split(key)
    return scale * jrandom.normal(w_key, (n, m)), scale * jrandom.normal(b_key, (n,))


def relu(x):
    return jnp.maximum(0, x)


def predict(params, image):
    # per-example predictions
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)


batched_predict = vmap(predict, in_axes=(None, 0))


def one_hot(x, k, dtype=jnp.int32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)


def loss(params, images, targets):
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)


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
        elif char == ' ':
            tokens.append(add_word(current_word))
            current_word = ''

    if current_word:
        tokens.append(add_word(current_word))

    val = jnp.pad(jnp.array(tokens, dtype=jnp.int32), (0, SENTENCE_LEN - len(tokens)), mode='constant')
    return val


def read_dataset():
    with open(DATA_PATH) as f:
        lines = list(map(lambda line: line.split(' || '), f.read().splitlines()))
        log(f'loaded tone data ({len(lines)=} lines)')
        return list(map(lambda x: [x[0].rstrip('.').lower(), TONES[x[1].rstrip('.').lower()]], lines))


def add_mutated_data(data):
    new_data = []
    for sentence, tone in data:
        new_data.append([sentence, tone])
        # remove each character from sentence in turn
        # assumed that this does not change the meaning of the sentence too much
        for i in range(1, len(sentence)):
            new_data.append([sentence[:i] + sentence[i + 1:], tone])
    return new_data


def tokenise_data(data):
    return list(map(lambda x: [tokenise(x[0]), x[1]], data))


def write_results(params, training_accs, test_accs):
    with open('params.txt', 'w') as f:
        f.write(params.__str__())

    with open('training_accs.txt', 'w') as f:
        f.write(training_accs.__str__())

    with open('test_accs.txt', 'w') as f:
        f.write(test_accs.__str__())


def plot_results(training_accs, test_accs):
    plt.plot(training_accs, label='Training Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def main():
    # Ensure TF does not see GPU and grab all GPU memory
    tf.config.set_visible_devices([], device_type='GPU')

    params = init_network_params(LAYER_SIZES, jrandom.key(0))

    tone_data = add_mutated_data(read_dataset())
    random.shuffle(tone_data)
    tone_data = tokenise_data(tone_data)

    train_data, test_data = tone_data[:int(len(tone_data) * TRAIN_RATIO)], tone_data[int(len(tone_data) * TRAIN_RATIO):]

    train_data_sentences = jnp.array(list(map(lambda x: x[0], train_data)), dtype=jnp.int32)
    train_labels = one_hot(jnp.array(list(map(lambda x: x[1], train_data)), dtype=jnp.int32), NUM_LABELS)
    train_sentences = jnp.reshape(train_data_sentences, (len(train_labels), SENTENCE_LEN))

    test_data_sentences = jnp.array(list(map(lambda x: x[0], test_data)), dtype=jnp.int32)
    test_labels = one_hot(jnp.array(list(map(lambda x: x[1], test_data)), dtype=jnp.int32), NUM_LABELS)
    test_sentences = jnp.reshape(test_data_sentences, (len(test_labels), SENTENCE_LEN))

    log(f'Train: {train_sentences.shape=}, {train_labels.shape=}')
    log(f'Test: {test_sentences.shape=}, {test_labels.shape=}')

    training_accs = []
    test_accs = []

    training_start = time.time()

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        for i in range(0, len(train_sentences), BATCH_SIZE):
            x, y = train_sentences[i:i + BATCH_SIZE], train_labels[i:i + BATCH_SIZE]
            x = jnp.reshape(x, (len(x), SENTENCE_LEN))
            params = update(params, x, y)
        epoch_time = time.time() - start_time

        train_acc = accuracy(params, train_sentences, train_labels)
        test_acc = accuracy(params, test_sentences, test_labels)

        if epoch % 10 == 0:
            log("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
            log("Training set accuracy {:0.3f}%".format(train_acc * 100))
            log("Test set accuracy {:0.3f}%".format(test_acc * 100))
        training_accs.append(float(train_acc))
        test_accs.append(float(test_acc))

    log(f'Finished training in {time.time() - training_start:0.3f} seconds')

    write_results(params, training_accs, test_accs)
    plot_results(training_accs, test_accs)

    log('Results written to files')


if __name__ == '__main__':
    main()
