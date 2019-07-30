import os
import zipfile
import gzip
import shutil
# import logging
import torch
import six
import tarfile

from six.moves.urllib.request import urlretrieve
from tqdm import tqdm
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText

def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optional
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
        return inner

def _infer_shape(f):
    num_lines, vector_dim = 0, None
    for line in f:
        if vector_dim is None:
            row = line.rstrip().split(b" ")
            vector = row[1:]
            # Assuming word, [vector] format
            if len(vector) > 2:
                # The header present in some (w2v) formats contains two elements.
                vector_dim = len(vector)
                num_lines += 1  # First element read
        else:
            num_lines += 1
    f.seek(0)
    return num_lines, vector_dim

def get_all_vectors(pretrained_model):
    emb_vectors = []
    
    if pretrained_model == "":
        return emb_vectors

    emb_vector_names = pretrained_model.split(",")
    for emb_vector_name in emb_vector_names:
        emb_info = emb_vector_name.split("_")
        if len(emb_info) == 3:
            emb_name, emb_set, emb_size = emb_info[0], emb_info[1], emb_info[2]
        else:
            emb_name, emb_set = emb_info[0], emb_info[1]

        if emb_name == "glove":  # glove_640B_300
            print("glove")
            emb_vectors.append(GloVe(name=emb_set, dim=emb_size))
        elif emb_name == "fasttext":
            if emb_set == "subwordcc":  # fasttext_subwordcc
                print("fasttext_subwordcc")
                emb_vectors.append(FastTextSubwordCC())
            elif emb_set == "wiki":  # fasttext_wiki_en
                print("fasttext_wiki")
                emb_vectors.append(FastText(language=emb_size))
            elif emb_set == "cc":  # fasttext_cc_en
                print("fasttext_cc")
                emb_vectors.append(FastTextCC(language=emb_size))
        elif emb_name == "char": # char_ngram
            if emb_set == "ngram":
                print("char_ngram")
                emb_vectors.append(CharNGram())
    return emb_vectors


class CustomVectors(Vectors):

    def cache(self, name, cache, url=None, max_vectors=None):
        if os.path.isfile(name):
            path = name
            if max_vectors:
                file_suffix = '_{}.pt'.format(max_vectors)
            else:
                file_suffix = '.pt'
            path_pt = os.path.join(cache, os.path.basename(name)) + file_suffix
        else:
            path = os.path.join(cache, name)
            if max_vectors:
                file_suffix = '_{}.pt'.format(max_vectors)
            else:
                file_suffix = '.pt'
            path_pt = path + file_suffix

        if not os.path.isfile(path_pt):
            if not os.path.isfile(path) and url:
                print('Downloading vectors from {}'.format(url))
                # logger.info('Downloading vectors from {}'.format(url))
                if not os.path.exists(cache):
                    os.makedirs(cache)
                dest = os.path.join(cache, os.path.basename(url))
                if not os.path.isfile(dest):
                    with tqdm(unit='B', unit_scale=True, miniters=1, desc=dest) as t:
                        try:
                            # urlretrieve(url, dest, reporthook=reporthook(t))
                            urlretrieve(url, dest)
                        except KeyboardInterrupt as e:  # remove the partial zip file
                            os.remove(dest)
                            raise e
                print('Extracting vectors into {}'.format(cache))
                # logger.info('Extracting vectors into {}'.format(cache))
                ext = os.path.splitext(dest)[1][1:]
                if ext == 'zip':
                    with zipfile.ZipFile(dest, "r") as zf:
                        zf.extractall(cache)
                elif ext == 'gz':
                    if dest.endswith('.tar.gz'):
                        with tarfile.open(dest, 'r:gz') as tar:
                            tar.extractall(path=cache)
                    elif dest.endswith('.gz'):
                        with gzip.open(dest, 'rb') as f_in:
                            with open(dest.replace(".gz", ""), 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                                path = dest.replace(".gz", "")
            if not os.path.isfile(path):
                raise RuntimeError('no vectors found at {}'.format(path))

            print("Loading vectors from {}".format(path))
            # logger.info("Loading vectors from {}".format(path))
            ext = os.path.splitext(path)[1][1:]
            if ext == 'gz':
                open_file = gzip.open
            else:
                open_file = open

            vectors_loaded = 0
            with open_file(path, 'rb') as f:
                num_lines, dim = _infer_shape(f)
                if not max_vectors or max_vectors > num_lines:
                    max_vectors = num_lines

                itos, vectors, dim = [], torch.zeros((max_vectors, dim)), None

                for line in tqdm(f, total=num_lines):
                    # Explicitly splitting on " " is important, so we don't
                    # get rid of Unicode non-breaking spaces in the vectors.
                    entries = line.rstrip().split(b" ")

                    word, entries = entries[0], entries[1:]
                    if dim is None and len(entries) > 1:
                        dim = len(entries)
                    elif len(entries) == 1:
                        print("Skipping token {} with 1-dimensional "
                                       "vector {}; likely a header".format(word, entries))
                        # logger.warning("Skipping token {} with 1-dimensional "
                                    #    "vector {}; likely a header".format(word, entries))
                        continue
                    elif dim != len(entries):
                        raise RuntimeError(
                            "Vector for token {} has {} dimensions, but previously "
                            "read vectors have {} dimensions. All vectors must have "
                            "the same number of dimensions.".format(word, len(entries),
                                                                    dim))

                    try:
                        if isinstance(word, six.binary_type):
                            word = word.decode('utf-8')
                    except UnicodeDecodeError:
                        print("Skipping non-UTF8 token {}".format(repr(word)))
                        # logger.info("Skipping non-UTF8 token {}".format(repr(word)))
                        continue

                    vectors[vectors_loaded] = torch.tensor([float(x) for x in entries])
                    vectors_loaded += 1
                    itos.append(word)

                    if vectors_loaded == max_vectors:
                        break

            self.itos = itos
            self.stoi = {word: i for i, word in enumerate(itos)}
            self.vectors = torch.Tensor(vectors).view(-1, dim)
            self.dim = dim
            print('Saving vectors to {}'.format(path_pt))
            # logger.info('Saving vectors to {}'.format(path_pt))
            if not os.path.exists(cache):
                os.makedirs(cache)
            torch.save((self.itos, self.stoi, self.vectors, self.dim), path_pt)
        else:
            print('Loading vectors from {}'.format(path_pt))
            # logger.info('Loading vectors from {}'.format(path_pt))
            self.itos, self.stoi, self.vectors, self.dim = torch.load(path_pt)

class FastTextSubwordCC(Vectors):
    def __init__(self, **kwargs):
        url = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M-subword.zip"
        super(FastTextSubwordCC, self).__init__(
            'fasttext-crawl-300d-2M-subword', url=url, **kwargs)


class FastTextCC(CustomVectors):
    def __init__(self, language, **kwargs):
        if language == "en":
            url = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/cc.{}.300.vec.gz".format(
            language)
        else:
            url = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.{}.300.vec.gz".format(
            language)
        print(url)
        super(FastTextCC, self).__init__(
            'fasttext-cc' + "-" + language, url=url, **kwargs)
