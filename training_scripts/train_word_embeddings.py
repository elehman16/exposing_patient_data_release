import argparse
import logging
import os
import time

import gensim
import pandas as pd
import spacy
import glob

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])


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


from joblib import Parallel, delayed

def tokenizer(chunk) :
    return [" ".join([token.text.lower() for token in sentence]) for sentence in nlp.pipe(chunk)]

def preprocess_parallel(texts, chunksize=100):
    chunker = (texts[i:i + chunksize] for i in range(0, len(texts), chunksize))
    executor = Parallel(n_jobs=16, backend='multiprocessing', prefer="processes", verbose=20)
    do = delayed(tokenizer)
    tasks = (do(chunk) for chunk in chunker)
    result = executor(tasks)
    return [text for chunk in result for text in chunk]


def train_word_embeddings(args: argparse.Namespace):
    """Train a word embedding model based on the given information.
    @param use_skipgram is whether we train a skipgram model or a cbow model.
    @param f_text is where we read the text from to train the model.
    @param wv_size is the size of the word embeddings to use.
    @param iter is the number of epochs to train for.
    @param window_size is the window size of the model.
    @param model_save_name is where to save the model.
    """
    all_notes = glob.glob(args.input_file)
    all_sentences = []
    for note_f in all_notes:
        notes = pd.read_csv(note_f).TEXT

        print("Loaded Text")
        sentences = [sentence for note in notes for sentence in note.split("\n")]

        print(f"Num Sentences : {len(sentences)}")

        tokenized_sentences = preprocess_parallel(sentences, chunksize=100000)
        print(f"Num Sentence : {len(tokenized_sentences)}")
        sentences = [sent.split() for sent in tokenized_sentences]
        print("Tokenized")
        all_sentences.extend(sentences)
    
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    model = gensim.models.Word2Vec(
        sentences,
        min_count=1,
        size=args.embedding_size,
        window=args.window_size,
        workers=16,
        negative=10,
        iter=args.epochs,
        sg=True if args.embedding_type == "skipgram" else False,
        callbacks=[callback()],
    )

    output_file = os.path.join(args.output_dir, f"{args.embedding_type}.vectors")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    model.save(output_file)


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
