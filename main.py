import tensorflow as tf
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp
import time

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

TRAIN_RATIO = 0.8
NUM_LABELS = len(TONES)

SENTENCE_LEN = 64
LAYER_SIZES = [SENTENCE_LEN, 32, 30]
STEP_SIZE = 0.002
NUM_EPOCHS = 5_000
BATCH_SIZE = 64


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def log(message: str):
    print(f'[LOG] {message}')


def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


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
        print(f'Adding word: {word}')
        words[word] = len(words)
    return words[word]


def tokenise(sentence: str) -> jnp.ndarray:
    tokens: list[int] = []
    current_word = ''
    for char in sentence:
        if char.isalnum():
            current_word += char
            continue
        elif current_word:
            tokens.append(add_word(current_word))
            current_word = ''

        if char != ' ':
            tokens.append(add_word(char))

    if current_word:
        tokens.append(add_word(current_word))

    val = jnp.pad(jnp.array(tokens, dtype=jnp.int32), (0, SENTENCE_LEN - len(tokens)), mode='constant')
    return val


# Ensure TF does not see GPU and grab all GPU memory
tf.config.set_visible_devices([], device_type='GPU')

with open('./tone_v1.txt') as f:
    params = init_network_params(LAYER_SIZES, random.key(0))

    lines = list(map(lambda line: line.split(' || '), f.read().splitlines()))
    log(f'loaded tone data ({len(lines)=} lines)')
    tone_data = list(map(lambda x: [tokenise(x[0].rstrip('.').lower()), TONES[x[1].rstrip('.').lower()]], lines))

    train_data, test_data = tone_data[:int(len(tone_data) * TRAIN_RATIO)], tone_data[int(len(tone_data) * TRAIN_RATIO):]
    log(f'{type(train_data)=}, {train_data[0]=}, {len(train_data)=}')

    train_data_sentences = jnp.array(list(map(lambda x: x[0], train_data)), dtype=jnp.int32)
    train_labels = one_hot(jnp.array(list(map(lambda x: x[1], train_data)), dtype=jnp.int32), NUM_LABELS)
    log(f'{type(train_data_sentences)=}, {train_data_sentences[0]=}, {train_labels=}')
    train_sentences = jnp.reshape(train_data_sentences, (len(train_labels), SENTENCE_LEN))
    log(f'loaded train data ({len(train_data)} lines)')

    test_data_sentences = jnp.array(list(map(lambda x: x[0], test_data)), dtype=jnp.int32)
    test_labels = one_hot(jnp.array(list(map(lambda x: x[1], test_data)), dtype=jnp.int32), NUM_LABELS)
    test_sentences = jnp.reshape(test_data_sentences, (len(test_labels), SENTENCE_LEN))
    log(f'loaded test data ({len(test_data)} lines)')

    log(f'Train: {train_sentences.shape=}, {train_labels.shape=}')
    log(f'Test: {test_sentences.shape=}, {test_labels.shape=}')


    def get_train_batches():
        for i in range(0, len(train_sentences), BATCH_SIZE):
            yield train_sentences[i:i + BATCH_SIZE], train_labels[i:i + BATCH_SIZE]


    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        for x, y in get_train_batches():
            x = jnp.reshape(x, (len(x), SENTENCE_LEN))
            params = update(params, x, y)
        epoch_time = time.time() - start_time

        train_acc = accuracy(params, train_sentences, train_labels)
        test_acc = accuracy(params, test_sentences, test_labels)
        log("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        log("Training set accuracy {:0.3f}%".format(train_acc * 100))
        log("Test set accuracy {:0.3f}%".format(test_acc))
