import argparse
import logging
import os
import time

import gensim
import pandas as pd
import spacy
from tqdm import tqdm

tqdm.pandas()
import swifter

nlp = spacy.load("en_core_sci_sm")
tokenizer = nlp.Defaults.create_tokenizer(nlp)


# global parameters
class callback(gensim.models.callbacks.CallbackAny2Vec):
    """Callback to print loss after each epoch."""

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print("Loss after epoch {}: {}".format(self.epoch, loss_now))
        self.epoch += 1
        print(time.strftime("%H:%M:%S", time.gmtime()))


def train_word_embeddings(args: argparse.Namespace):
    """Train a word embedding model based on the given information.
    @param use_skipgram is whether we train a skipgram model or a cbow model.
    @param f_text is where we read the text from to train the model.
    @param wv_size is the size of the word embeddings to use.
    @param iter is the number of epochs to train for.
    @param window_size is the window size of the model.
    @param model_save_name is where to save the model.
    """
    notes = pd.read_csv(args.input_file).TEXT[:10000]

    print("Loaded Text")
    sentences = notes.progress_apply(
        lambda note: [[token.text.lower() for token in tokenizer(sentence)] for sentence in note.split("\n")]
    )

    print("Tokenized")

    sentences = [sent for note in sentences for sent in note]

    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    model = gensim.models.Word2Vec(
        sentences,
        min_count=1,
        size=args.embedding_size,
        window=args.window_size,
        workers=5,
        negative=10,
        iter=args.epochs,
        sg=True if args.embedding_type == "skipgram" else False,
        callbacks=[callback()],
    )

    output_file = os.path.join(args.output_dir, f"{args.embedding_type}.vectors")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    model.wv.save(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", help="Location of the text file to train on.", type=str)
    parser.add_argument(
        "--embedding-type", help="Are we using SkipGram or CBoW?", choices=["cbow", "skipgram"]
    )
    parser.add_argument("--embedding-size", help="How large are the word vectors?", default=200, type=int)
    parser.add_argument("--epochs", help="The number of epochs to train for.", default=10, type=int)
    parser.add_argument("--window-size", help="What window size to use.", default=6, type=int)
    args = parser.parse_args()

    train_word_embeddings(args)
