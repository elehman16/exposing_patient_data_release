import re
import string


def string_cleanup(x):
    y = re.sub("\\[(.*?)\\]", "", x)  # remove de-identified brackets
    y = re.sub("[0-9]+\.", "", y)  # remove 1.2. since the segmenter segments based on this
    y = re.sub("dr\.", "doctor", y)
    y = re.sub("m\.d\.", "md", y)
    y = re.sub("ms\.", "ms", y)
    y = re.sub("mr\.", "mr", y)
    y = re.sub("mrs\.", "mrs", y)
    y = re.sub("admission date:", "", y)
    y = re.sub("discharge date:", "", y)
    y = re.sub("--|__|==", "", y)

    # remove, digits, spaces
    y = y.translate(str.maketrans("", "", string.digits))
    y = " ".join(y.split())
    return y


from spacy.lang.en import English

sentence_nlp = English()  # just the language with no model
sentence_nlp.add_pipe(sentence_nlp.create_pipe("sentencizer"))
sentence_nlp.max_length = 2000000


def convert_to_sentence(text: str) -> str:
    doc = sentence_nlp(text)
    text = []
    try:
        for sent in doc.sents:
            st = str(sent).strip()
            if len(st) < 20:
                # a lot of abbreviation is segmented as one line. But these are all describing the previous things
                # so I attached it to the sentence before
                if len(text) != 0:
                    text[-1] = " ".join((text[-1], st))
                else:
                    text = [st]
            else:
                text.append(st)
    except:
        print(doc)

    return "\n".join([re.sub(r"\s+", " ", sent).strip() for sent in text if len(sent) > 0])


def preprocess_text(text: str, sentencize: bool = True):
    if text is None:
        text = " "
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.strip()
    text = text.lower()
    text = string_cleanup(text)

    if sentencize:
        text = convert_to_sentence(text)
    return text