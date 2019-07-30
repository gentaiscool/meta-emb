########################################
# modified torchtext.data.field.py
########################################

import six
from collections import Counter, OrderedDict

import torch
from torchtext.data import Dataset
from torchtext.data import RawField
from torchtext.vocab import Vocab
# from torchtext.data.utils import get_tokenizer, dtype_to_attr, is_tokenizer_serializable

# from torchtext.data.utils
def _split_tokenizer(x):
    return x.split()

def _spacy_tokenize(x, spacy):
    return [tok.text for tok in spacy.tokenizer(x)]

def get_tokenizer(tokenizer, language='en'):
    # default tokenizer is string.split(), added as a module function for serialization
    if tokenizer is None:
        return _split_tokenizer

    # simply return if a function is passed
    if callable(tokenizer):
        return tokenizer

    if tokenizer == "spacy":
        try:
            import spacy
            spacy = spacy.load(language)
            return partial(_spacy_tokenize, spacy=spacy)
        except ImportError:
            print("Please install SpaCy. "
                  "See the docs at https://spacy.io for more information.")
            raise
        except AttributeError:
            print("Please install SpaCy and the SpaCy {} tokenizer. "
                  "See the docs at https://spacy.io for more "
                  "information.".format(language))
            raise
    elif tokenizer == "moses":
        try:
            from sacremoses import MosesTokenizer
            moses_tokenizer = MosesTokenizer()
            return moses_tokenizer.tokenize
        except ImportError:
            print("Please install SacreMoses. "
                  "See the docs at https://github.com/alvations/sacremoses "
                  "for more information.")
            raise
    elif tokenizer == "toktok":
        try:
            from nltk.tokenize.toktok import ToktokTokenizer
            toktok = ToktokTokenizer()
            return toktok.tokenize
        except ImportError:
            print("Please install NLTK. "
                  "See the docs at https://nltk.org  for more information.")
            raise
    elif tokenizer == 'revtok':
        try:
            import revtok
            return revtok.tokenize
        except ImportError:
            print("Please install revtok.")
            raise
    elif tokenizer == 'subword':
        try:
            import revtok
            return partial(revtok.tokenize, decap=True)
        except ImportError:
            print("Please install revtok.")
            raise
    raise ValueError("Requested tokenizer {}, valid choices are a "
                     "callable that takes a single string as input, "
                     "\"revtok\" for the revtok reversible tokenizer, "
                     "\"subword\" for the revtok caps-aware tokenizer, "
                     "\"spacy\" for the SpaCy English tokenizer, or "
                     "\"moses\" for the NLTK port of the Moses tokenization script.".format(tokenizer))

def dtype_to_attr(dtype):
    # convert torch.dtype to dtype string id
    # e.g. torch.int32 -> "int32"
    # used for serialization
    _, dtype = str(dtype).split('.')
    return dtype

def is_tokenizer_serializable(tokenizer, language):
    """Extend with other tokenizers which are found to not be serializable
    """
    if tokenizer == 'spacy':
        return False
    return True

class Field(RawField):
    """Defines a datatype together with instructions for converting to Tensor.
    Field class models common text processing datatypes that can be represented
    by tensors.  It holds a Vocab object that defines the set of possible values
    for elements of the field and their corresponding numerical representations.
    The Field object also holds other parameters relating to how a datatype
    should be numericalized, such as a tokenization method and the kind of
    Tensor that should be produced.
    If a Field is shared between two columns in a dataset (e.g., question and
    answer in a QA dataset), then they will have a shared vocabulary.
    Attributes:
        sequential: Whether the datatype represents sequential data. If False,
            no tokenization is applied. Default: True.
        use_vocab: Whether to use a Vocab object. If False, the data in this
            field should already be numerical. Default: True.
        init_token: A token that will be prepended to every example using this
            field, or None for no initial token. Default: None.
        eos_token: A token that will be appended to every example using this
            field, or None for no end-of-sentence token. Default: None.
        fix_length: A fixed length that all examples using this field will be
            padded to, or None for flexible sequence lengths. Default: None.
        dtype: The torch.dtype class that represents a batch of examples
            of this kind of data. Default: torch.long.
        preprocessing: The Pipeline that will be applied to examples
            using this field after tokenizing but before numericalizing. Many
            Datasets replace this attribute with a custom preprocessor.
            Default: None.
        postprocessing: A Pipeline that will be applied to examples using
            this field after numericalizing but before the numbers are turned
            into a Tensor. The pipeline function takes the batch as a list, and
            the field's Vocab.
            Default: None.
        lower: Whether to lowercase the text in this field. Default: False.
        tokenize: The function used to tokenize strings using this field into
            sequential examples. If "spacy", the SpaCy tokenizer is
            used. If a non-serializable function is passed as an argument,
            the field will not be able to be serialized. Default: string.split.
        tokenizer_language: The language of the tokenizer to be constructed.
            Various languages currently supported only in SpaCy.
        include_lengths: Whether to return a tuple of a padded minibatch and
            a list containing the lengths of each examples, or just a padded
            minibatch. Default: False.
        batch_first: Whether to produce tensors with the batch dimension first.
            Default: False.
        pad_token: The string token used as padding. Default: "<pad>".
        unk_token: The string token used to represent OOV words. Default: "<unk>".
        pad_first: Do the padding of the sequence at the beginning. Default: False.
        truncate_first: Do the truncating of the sequence at the beginning. Default: False
        stop_words: Tokens to discard during the preprocessing step. Default: None
        is_target: Whether this field is a target variable.
            Affects iteration over batches. Default: False
    """

    vocab_cls = Vocab
    # Dictionary mapping PyTorch tensor dtypes to the appropriate Python
    # numeric type.
    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,

        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
    }

    ignore = ['dtype', 'tokenize']

    def __init__(self, sequential=True, use_vocab=True, init_token=None,
                 eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False,
                 tokenize=None, tokenizer_language='en', include_lengths=False,
                 batch_first=False, pad_token="<pad>", unk_token="<unk>",
                 pad_first=False, truncate_first=False, stop_words=None,
                 is_target=False):
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.lower = lower
        # store params to construct tokenizer for serialization
        # in case the tokenizer isn't picklable (e.g. spacy)
        self.tokenizer_args = (tokenize, tokenizer_language)
        self.tokenize = get_tokenizer(tokenize, tokenizer_language)
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token if self.sequential else None
        self.pad_first = pad_first
        self.truncate_first = truncate_first
        try:
            self.stop_words = set(stop_words) if stop_words is not None else None
        except TypeError:
            raise ValueError("Stop words must be convertible to a set")
        self.is_target = is_target

    def __getstate__(self):
        str_type = dtype_to_attr(self.dtype)
        if is_tokenizer_serializable(*self.tokenizer_args):
            tokenize = self.tokenize
        else:
            # signal to restore in `__setstate__`
            tokenize = None
        attrs = {k: v for k, v in self.__dict__.items() if k not in self.ignore}
        attrs['dtype'] = str_type
        attrs['tokenize'] = tokenize

        return attrs

    def __setstate__(self, state):
        state['dtype'] = getattr(torch, state['dtype'])
        if not state['tokenize']:
            state['tokenize'] = get_tokenizer(*state['tokenizer_args'])
        self.__dict__.update(state)

    def __hash__(self):
        # we don't expect this to be called often
        return 42

    def __eq__(self, other):
        if not isinstance(other, RawField):
            return False

        return self.__dict__ == other.__dict__

    def preprocess(self, x):
        """Load a single example using this field, tokenizing if necessary.
        If the input is a Python 2 `str`, it will be converted to Unicode
        first. If `sequential=True`, it will be tokenized. Then the input
        will be optionally lowercased and passed to the user-provided
        `preprocessing` Pipeline."""
        if (six.PY2 and isinstance(x, six.string_types) and
                not isinstance(x, six.text_type)):
            x = Pipeline(lambda s: six.text_type(s, encoding='utf-8'))(x)
        if self.sequential and isinstance(x, six.text_type):
            x = self.tokenize(x.rstrip('\n'))
        if self.lower:
            x = Pipeline(six.text_type.lower)(x)
        if self.sequential and self.use_vocab and self.stop_words is not None:
            x = [w for w in x if w not in self.stop_words]
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, device=None):
        """ Process a list of examples to create a torch.Tensor.
        Pad, numericalize, and postprocess a batch and create a tensor.
        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            torch.autograd.Variable: Processed object given the input
            and custom postprocessing Pipeline.
        """
        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device)
        return tensor

    def pad(self, minibatch):
        """Pad a batch of examples using this field.
        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True` and `self.sequential` is `True`, else just
        returns the padded list. If `self.sequential` is `False`, no padding is applied.
        """
        minibatch = list(minibatch)
        if not self.sequential:
            return minibatch
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x)) +
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]))
            else:
                padded.append(
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]) +
                    [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return (padded, lengths)
        return padded

    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.
        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))
        def_specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))
        specials = kwargs.pop("specials", def_specials) # custom
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a Variable.
        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.
        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

        if self.use_vocab:
            if self.sequential:
                arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]
            else:
                arr = [self.vocab.stoi[x] for x in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab)
        else:
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype))
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explicitly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            if not self.sequential:
                arr = [numericalization_func(x) if isinstance(x, six.string_types)
                       else x for x in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

        var = torch.tensor(arr, dtype=self.dtype, device=device)

        if self.sequential and not self.batch_first:
            var.t_()
        if self.sequential:
            var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var