import csv
import random

Row = tuple[str, float]
Dataset = tuple[list[Row], list[Row]]


def sentiment_text_to_values(text: str):
    text = text.lower().strip()
    if text == 'positive':
        return 1.0
    elif text == 'negative':
        return 0.0
    return 0.5


def twitter() -> Dataset:
    with open('data/twitter_training.csv', newline='') as f:
        twitter_training = list(map(lambda row: (row[3], sentiment_text_to_values(row[2])), csv.reader(f)))

    with open('data/twitter_validation.csv', newline='') as f:
        twitter_test = list(map(lambda row: (row[3], sentiment_text_to_values(row[2])), csv.reader(f)))

    return twitter_training, twitter_test


def write_data(training: list[Row], test: list[Row], training_path: str, test_path: str):
    with open(training_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(training)

    with open(test_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(test)


def main():
    twitter_training, twitter_test = twitter()

    training = [*twitter_training]
    test = [*twitter_test]

    random.shuffle(training)
    random.shuffle(test)
    write_data(training, test, 'data/training.csv', 'data/validation.csv')


if __name__ == '__main__':
    main()
