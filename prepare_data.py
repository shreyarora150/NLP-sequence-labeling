import os
import spacy
from spacy.tokens import Span
import pandas as pd
from glob import glob
import zipfile
from urllib.request import urlretrieve
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def process_set(dset, nlp):
    data_rows = []
    text_path = (f"MeasEval/data/{dset}/text/"
                 if os.path.exists(f"MeasEval/data/{dset}/text/")
                 else f"MeasEval/data/{dset}/txt/")
    for text_file in sorted(glob(os.path.join(text_path, "*.txt"))):
        docname = os.path.splitext(os.path.basename(text_file))[0]
        text = open(text_file, encoding='utf-8').read()
        doc = nlp(text)
        for s, sent in enumerate(doc.sents):
            for t, token in enumerate(sent):
                token_id = sent.start + t
                span = Span(doc, token_id, token_id + 1)
                data_rows.append([docname, s, token.text, token.lemma_,
                                  span.start_char, span.end_char, "O"])

    data_df = pd.DataFrame(data_rows, columns=["docId", "sentId", "word", "lemma",
                                               "startOffset", "endOffset", "label"])

    for tsv_file in sorted(glob(os.path.join(f"MeasEval/data/{dset}/tsv/", "*.tsv"))):
        tsv = pd.read_csv(tsv_file, sep="\t", encoding="utf8")
        tsv = tsv[["docId",  "annotType", "startOffset", "endOffset"]].drop_duplicates()
        tsv = tsv[tsv["annotType"] == "Quantity"]
        for _, ann in tsv.iterrows():
            tokens = data_df[(data_df["docId"] == ann["docId"]) &
                             (data_df["startOffset"] >= ann["startOffset"]) &
                             (data_df["startOffset"] < ann["endOffset"])].index
            if (data_df.loc[tokens]["label"] != "O").any():
                continue
            data_df.loc[tokens[:1], "label"] = f"B-{ann['annotType']}"
            data_df.loc[tokens[1:], "label"] = f"I-{ann['annotType']}"
    data_df.drop(columns=["startOffset", "endOffset"]).to_csv(f"data/{dset}.tsv", sep="\t", index=False, encoding="utf8")

    
def prepare_data():
    if not os.path.exists("glove.6B.zip"):
        #  wget http://nlp.stanford.edu/data/glove.6B.zip
        print("Downloading glove.6B.zip...", end=" ", flush=True)
        urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", "glove.6B.zip")
        print("Done.")
    if not os.path.exists("glove/glove.6B.300d.txt"):
        # unzip -d glove glove.6B.zip
        print("Unzipping...", end=" ", flush=True)
        with zipfile.ZipFile('glove.6B.zip', 'r') as zip_ref:
            zip_ref.extractall("glove")
        print("Done.")
    print("Processing...", end=" ", flush=True)
    os.makedirs("data", exist_ok=True)
    nlp = spacy.load("en_core_web_sm")
    process_set("train", nlp)
    process_set("trial", nlp)
    process_set("eval", nlp)
    print("Done.")


if __name__ == "__main__":
    #  git clone https://github.com/harperco/MeasEval.git
    nlp = spacy.load("en_core_web_sm")
    process_set("train", nlp)
    process_set("trial", nlp)
    process_set("eval", nlp)
    print("Done.")
