import re
import os
import string
import collections


class MakeDoc2Vec(object):
    def __init__(self, list_of_words, size=100, window=5, min_count=5):
        """Create Doc2Vec model

        Args:
            list_of_words (list of list of words): Description
            size (int, optional): doc2vec vector size
            window (int, optional): doc2vec window size
            min_count (int, optional):
        """
        from gensim.models import doc2vec
        from gensim.models.doc2vec import LabeledSentence

        index = 1
        list_of_docs = []
        for i in list_of_words:
            list_of_docs.append(LabeledSentence(i, ['doc_{}'.format(index)]))
            index += 1
        self.model = doc2vec.Doc2Vec(list_of_docs, size=size, window=window, min_count=min_count, workers=4)
        self.doc_len = len(list_of_words)

    def to_array(self):
        """Convert model into data features

        Returns:
            np.array: array of vectors (as feature)
        """
        import numpy as np

        matrix = [self.model.docvecs[i] for i in range(self.doc_len)]
        return np.array(matrix)


def preprocess(sentence, language='english', stopword=False, nonascii=True, punctuation=None):
    """Preprocess String, remove punctuation and delete stopwords

    Args:
        sentence (TYPE): Description
        language (str, optional): indonesia or english use to get stopwords
        stopword (bool, optional): Remove stopwords or not
        nonascii (bool, optional): Remove nonascii character or not
        punctuation (None, optional): punctuation (
                'all' --> remove all punctuation,
                '!;,' --> keep some punctuation,
                None --> punctuation not removed
                )
    Returns:
        str: cleaned sentence
    """
    import re
    import string
    from nltk.corpus import stopwords

    if nonascii == True:
        sentence = re.sub("[^\x00-\x7F]+\ *(?:[^\x00-\x7F]| )*", "", sentence, flags=re.UNICODE)

    if punctuation is not None:
        table = string.maketrans("", "")
        remove = string.punctuation  # delete all punctuation

        if punctuation != 'all':
            for i in punctuation:  # list all punctuation that don't want to deleted
                remove = remove.replace(i, '')

        # delete punctuation
        sentence = sentence.translate(table, remove)

    words = re.split(r'\s', sentence)  # delete empty char from list

    # stopword
    if stopword == True:
        if language == 'indonesia':
            basepath = os.path.dirname(__file__)
            rel_path = "stopword.txt"  # get stopwords
            filepath = os.path.abspath(os.path.join(basepath, rel_path))
            f = open(filepath, "r")
            stopwords = [line.rstrip('\n') for line in f]
            words = filter(lambda x: x not in stopwords, words)

        elif language == 'english':
            stopwords = stopwords.words('english')
            words = filter(lambda x: x not in stopwords, words)

    sentence = ' '.join(words).lower()
    return sentence
