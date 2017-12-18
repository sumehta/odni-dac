#!/usr/bin/env python
"""
Version 0.1 - Dependency based subsequence similarity using convolution kernel
"""
__author__ = 'Rupen'
__email__ = 'rupen@cs.vt.edu'


import re
import os
import math
import numpy as np
from nltk import word_tokenize, sent_tokenize

from scipy.linalg.blas import dscal, dnrm2

import functools
import ujson as json
from collections import defaultdict
from utils import get_key_terms

import spacy
import gensim
import subprocess


REAL       = np.float32
NPDOT      = np.dot
NPARRAY    = np.array
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


nlp = spacy.load('en')
def _token_info(token):
    return {'idx': token.i,
            'ent': "-".join([token.ent_iob_, token.ent_type_]) if token.ent_type_ else token.ent_iob_,
            'start': token.idx,
            'lemma': token.lemma_,
            'text' : token.text,
            'clean_text' : remove_determiners(token.text),
            'pos'  : token.pos_,
            'tag'  : token.tag_,
            'dep'  : token.dep_,
            'head' : token.head.i,
            'ws'   : token.whitespace_
           }

def _tokens(doc):
    return list(map(_token_info, doc))

def remove_determiners(text):
    _text = re.sub(r'(\ba\b|\ban\b|\band\b|\bthe|\bis\b|\bof\b)', "", text, re.UNICODE|re.IGNORECASE)
    return "_".join(filter(None,_text.split()))

class SpacyEnricher(object):
    @staticmethod
    def getTokens(text, collapse_phrases=True):
        """
        collapse phrases will convert noun-chunks and named entities like:
        artificial intelligence ==> aritificial_intelligence
        the United States ==> United_States (minus the determiners)
        """
        doc = nlp(text)
        if collapse_phrases:
            for ent in doc.ents:
                if ent.label_ != "O":
                    ent.merge(ent.root.tag_, ent.text, ent.label_)
            for ent in list(doc.noun_chunks):
                ent.merge(ent.root.tag_, ent.text, ent.label_)
        return _tokens(doc)


class Embeddings:
    def __init__(self, w2vfile, phrase_similarity=0.2, word_similarity=0.3):
        self._embeddings = gensim.models.KeyedVectors.load_word2vec_format(w2vfile, binary=True)
        self._cache = {}
        self._wsim = word_similarity
        self._psim = phrase_similarity
        self._ignore_pos_tags = set([u"PUNCT", u"SPACE", u"SYM", u"NUM", u"EOL", u"DET", u"PART"])

    def info(self):
        return "embeddings (vocab size %d) loaded w/ thresholds: Phrase=%.2f Word=%.2f" % (len(self._embeddings.vocab), self._psim, self._wsim)

    def _unitvec_l2(self, vec):
        return dscal(1.0 / dnrm2(vec), vec)

    def set_phrase_similarity_threshold(self, val):
        self._psim = val

    def set_word_similarity_threshold(self, val):
        self._wsim = val

    def parse_doc(self, tokens, text_field="clean_text"):
        N = len(tokens)

        for index in range(0,N):
            tokens[index]['deps'] = []

        for index in range(0,N):
            if tokens[index]['head'] != index:
                tokens[tokens[index]['head']]['deps'].append((index, tokens[index]['dep']))

        _tokens = N*[0]

        for index in range(0,N):
            if tokens[index]['head'] == index:
                head = -1
            else:
                head = tokens[index]['head']

            _tokens[index] = tuple([tokens[index][text_field], tokens[index]['pos'],
                                    head, tuple(tokens[index]['deps'])])

        return tuple(_tokens)

    def _dbget(self, w):
        if w in self._embeddings:
            return self._embeddings[w]
        elif '_' in w:
            arr = [self._embeddings[v] for v in w.split("_") if v in self._embeddings]
            if arr:
                return NPARRAY(arr).mean(axis=0)
        raise ValueError("Empty Array: %s" % w)

    def _in_vocab(self, w):
        if w in self._embeddings:
            return 1
        elif '_' in w and any([True for v in w.split("_") if v in self._embeddings]):
            return 1
        else:
            return 0

    def _n_similarity(self, w1, w2, u, k):
        wv1, wv2, uv = self._dbget(w1), self._dbget(w2), self._dbget(u)
        self._cache[k] = NPDOT(self._unitvec_l2(NPARRAY([wv1, uv]).mean(axis=0)),
                               self._unitvec_l2(NPARRAY([wv2, uv]).mean(axis=0))) >= self._psim
        return self._cache[k]

    def _similarity(self, w1, w2, k):
        wv1, wv2 = self._dbget(w1), self._dbget(w2)
        self._cache[k] = NPDOT(self._unitvec_l2(wv1), self._unitvec_l2(wv2)) >= self._wsim
        return self._cache[k]

    def _get_key(self, n1, n2):
        if n1 > n2:
            return (n2, n1)
        else:
            return (n1, n2)

    def has_similar_production(self, n1, n2, hd=u""):
        #if n1 == n2:
        #    return 1

        if not self._in_vocab(n1) or not self._in_vocab(n2):
            return 0

        if n1 > n2:
            k = (n2, n1)
        else:
            k = (n1, n2)
        k2 = k + (hd,)

        if k in self._cache:
            s1 = self._cache[k]
        else:
            s1 = self._similarity(n1, n2, k)

        if k2 in self._cache:
            s2 = self._cache[k2]
        elif hd != u"":
            if not self._in_vocab(hd):
                self._cache[k2] = 0
                s2 = 0
            else:
                s2 = self._n_similarity(n1, n2, hd, k2)
        else:
            s2 = 0

        return (s1 | s2) & 1

    def similarity(self, w1, w2):
        if self._in_vocab(w1) and self._in_vocab(w2):
            k = self._get_key(w1, w2)
            wv1, wv2 = self._dbget(w1), self._dbget(w2)
            return NPDOT(self._unitvec_l2(wv1), self._unitvec_l2(wv2))
        else:
            return 0.0

    def multiword_similarity(self, words1, words2):
        wv1 = [self._dbget(w) for w in words1 if self._in_vocab(w)]
        wv2 = [self._dbget(w) for w in words2 if self._in_vocab(w)]
        if len(wv1) > 0 and len(wv2) > 0:
            return NPDOT(self._unitvec_l2(NPARRAY(wv1).mean(axis=0)),
                               self._unitvec_l2(NPARRAY(wv2).mean(axis=0)))
        else:
            0.0

def _get_dep_label_of_node(deplist, n):
    for c, dep_label in deplist:
        if c == n:
            return dep_label
    return u""

def _has_same_production(n1, n2, t1, t2, embd, debug=False):

    w1 = t1[n1][0]
    w2 = t2[n2][0]

    """
    To switch off use of embeddings for word or phrase similarity comment
    out rest of function body and replace with below commented if clause
    """
    #if w1 == w2:
    #    return 1

    p1 = t1[n1][1]
    p2 = t2[n2][1]

    hd1 = t1[n1][2]
    hd2 = t2[n2][2]
    val = 0
    if hd1 != -1 and hd2 != -1:
        h1 = t1[hd1][0]
        h2 = t2[hd2][0]
        if h1 == h2:
            d1 = _get_dep_label_of_node(t1[hd1][3], n1)
            d2 = _get_dep_label_of_node(t2[hd2][3], n2)
            if d1 == d2 and d1 in set([u"compound", u"amod"]):
                return embd.has_similar_production(w1, w2, h1)

    return ( embd.has_similar_production(w1, w2) | (w1 == w2) ) & 1

def _child_combinations(deplist):
    # implementation of itertools.combinations(iterable, 2)
    result = []
    N = len(deplist) - 1
    for x in range(N):
        ls = deplist[x+1:]
        for y in ls:
            result.append((deplist[x][0],y[0]))
    return result

def _common_peak_path(n1, n2, t1, t2, embd, lm, lmsq):
    cdp_score = _common_downward_path(n1, n2, t1, t2, embd, lm, lmsq)
    cpp_score = 0.0
    ch1 = _child_combinations(t1[n1][3])
    ch2 = _child_combinations(t2[n2][3])

    for c1, c_1 in ch1:
        for c2, c_2 in ch2:
            if (_has_same_production(c1, c2, t1, t2, embd) == 1 and
                _has_same_production(c_1, c_2, t1, t2, embd) == 1):

                _cdp1 = _common_downward_path(c1, c2, t1, t2, embd, lm, lmsq)
                _cdp2 = _common_downward_path(c_1, c_2, t1, t2, embd, lm, lmsq)
                cpp_score += lmsq + (lm * _cdp1) + (lm * _cdp2) + (lmsq * _cdp1 * _cdp2)
    return cdp_score + cpp_score

def _common_downward_path(n1, n2, t1, t2, embd, lm, lmsq):
    cdp_score = 0.0
    ch1 = t1[n1][3]
    ch2 = t2[n2][3]

    for c1, s1 in ch1:
        for c2, s2 in ch2:
            if _has_same_production(c1, c2, t1, t2, embd) == 1:
                _cdp = _common_downward_path(c1, c2, t1, t2, embd, lm, lmsq)
                cdp_score += lm + (lm * _cdp)
    return cdp_score

def _dependencyWordSubsequenceKernel(t1, t2, embd, lambda_=0.5, debug=False):
    _lambda_sq = lambda_ ** 2
    _tot_score = 0
    skip_pos_tags = set(['ADV', 'PUNCT', 'ADP', 'DET'])

    N1 = len(t1)
    N2 = len(t2)
    #calculate dot product
    for idx in range(0, N1):
        for jdx in range(0, N2):
            if _has_same_production(idx, jdx, t1, t2, embd, debug) == 1:
                _peak_score = _common_peak_path(idx, jdx, t1, t2, embd, lambda_, _lambda_sq)
                if t1[idx][1] not in skip_pos_tags and t2[jdx][1] not in skip_pos_tags:
                    if '_' in t2[jdx][0]:
                        _tot_score += len(t2[jdx][0].split("_")) + _peak_score
                    else:
                        _tot_score += 1 + _peak_score
                    if debug:
                        print(t1[idx][:2], t2[jdx][:2], _peak_score, _tot_score)

    return _tot_score

class ConvolutionKernel(object):
    def __init__(self, embeddings_obj):
        self._embd = embeddings_obj
        self._lambda = 0.5

    def similarity(self, primary_text , candidate_text, collapse_phrases=False, debug=False):
        # extract tokens
        t1 = SpacyEnricher.getTokens(primary_text, collapse_phrases=collapse_phrases)
        t2 = SpacyEnricher.getTokens(candidate_text, collapse_phrases=collapse_phrases)
        # extract dependency trees
        d1, d2 = self._embd.parse_doc(t1), self._embd.parse_doc(t2)

        sim_score = _dependencyWordSubsequenceKernel(d1, d2, self._embd,
                                                     lambda_=self._lambda, debug=debug)
        #calculate normalized score
        sim_scr_11 = _dependencyWordSubsequenceKernel(d1, d1, self._embd, lambda_=self._lambda)
        sim_scr_22 = _dependencyWordSubsequenceKernel(d2, d2, self._embd,lambda_=self._lambda)

        # print(candidate_text)
        # if (sim_scr_11 != 0) and (sim_scr_22 != 0):
        norm_score = (sim_score / math.sqrt(sim_scr_11 * sim_scr_22))
        # else:
        #     norm_score = -1
        #return both raw & normalized similarity score
        return sim_score, norm_score


def article_score(article, key_words=[]):
    """ Given a signal article object and a list of key_words, compute a score on how
    relevant that article is matching the given key words.
    Args:
        article:
        key_words:
    Returns:
        score:
    """
    ti_score = 0
    ta_score = 0
    at_score = 0
    te_score = 0
    ab_score = 0

    # title score
    if article['title']:
        title_tokens = set([t.lower() for t in word_tokenize(article['title'])])
        for word in key_words:
            if set(word.lower().split()).issubset(title_tokens):
                ti_score+=1

    # tag score:
    for tag in article['tags']:
        tag_tokens = set([t.lower() for t in word_tokenize(tag)])
        for word in key_words:
            if set(word.lower().split()).issubset(tag_tokens):
                ta_score+=1
    # imgalttext
    for alt_text in article['imgalttext']:
        alt_text_tokens = set([t.lower() for t in word_tokenize(alt_text)])
        for word in key_words:
           if set(word.lower().split()).issubset(alt_text_tokens):
                at_score+=1

    # text
    if article['text']:
        text_tokens = set(word_tokenize(article['text']))

        # key words can occur separately in different articles
        for word in key_words:
            if set(word.lower().split()).issubset(set([t.lower() for t in text_tokens])):
                te_score+=1
    # abstract
    if article['abstract']:
        article_tokens = set([t.lower() for t in word_tokenize(article['abstract'])])
        for word in key_words:
            if set(word.lower().split()).issubset(article_tokens):
                ab_score+=1

    return 2*ti_score + 2*ta_score + at_score + te_score + ab_score


def qa_pair_generator(key_terms, articles, threshold_score=3):
    for article in articles:
        if article_score(article, key_terms) >= threshold_score:
            for i, sentence in enumerate(article['abstract_sentences']):
                question = question_generator(sentence)
                if question:
                    yield question, sentence, article["id"], i

def question_generator(sentence=""):
    """ Generate questions for the given sentence.
    """
    os.chdir("/Users/sneha/Documents/dev/odni/QuestionGeneration/")
    try:
        p = subprocess.Popen(['bash', 'run.sh'], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = p.communicate(input=bytearray(sentence, 'utf-8'), timeout=1000)[0]
        output =  output.decode()
    except TimeoutExpired:
        # p.kill()
        print('Question could not be generated!')
        return

    try:
        return output.split('\n')[3].split('\t')[0]
    except:
        print("Error generating question")


if __name__ == '__main__':
    import logging
    import argparse
    """
    Caution: collapse_phrases is experiental. If selected pls test performance.
    """

    ap = argparse.ArgumentParser()
    ap.add_argument('--debug', action='store_true', help='show basic debug statements')
    ap.add_argument('--collapse-phrases', action='store_true', help='collapse phrases is experimental (use with caution)')
    ap.add_argument('--question', required=True, help='Input reference question')
    ap.add_argument('--infile', required=True, help='Path to the input corpus JSON')
    ap.add_argument('--w2vfile', required=True, help='Path to the word2vec file')
    ap.add_argument('--outfile', required=True, help='Path to output JSON')
    ap.add_argument('--ws', type=float, default=0.3, help='word pair similarity threshold')
    ap.add_argument('--ps', type=float, default=0.2, help='phrase pair similarity threshold')
    args = ap.parse_args()


    W2VFILE = args.w2vfile

    logger = logging.getLogger("Convolution-Kernel SubseqSimilarity")

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("loading embeddings from file: %s" % W2VFILE)
    logger.info("loading embeddings is a time-consuming step, do this only once")
    ## loading embeddings is a time-consuming step, do this only once
    #embeddings = Embeddings(W2VFILE, phrase_similarity=0.2, word_similarity=0.3)
    embeddings = Embeddings(W2VFILE, phrase_similarity=args.ps, word_similarity=args.ws)
    logger.info(embeddings.info())
    logger.info("creating kernel")
    kernel = ConvolutionKernel(embeddings)

    logger.info("calculating similarity scores from candidates...")
    scores = []

    corpus = []
    for article in open(args.infile):
        corpus.append(json.loads(article))

    corpus = {a['id']:a for a in corpus}

    key_terms = get_key_terms(args.question, kernel, corpus)
    logger.info("Retrieving relevant documents accorind to key_terms: %s" % ','.join(key_terms))

    candidates = qa_pair_generator(key_terms, corpus.values(), threshold_score=3)

    limit = 200 # limit items

    logger.info("Inspecting candidates")
    for c in candidates:
        score, norm_score = kernel.similarity(args.question, c[0], collapse_phrases=args.collapse_phrases, debug=args.debug)
        # use this defulat setting for production
        # score, norm_score = kernel.similarity(q0, c)
        scores.append((c, score, norm_score))
        limit-=1
        if limit<=0:
            break;
        if args.debug:
            #use when debug is true
            print("\n".join(["",c]), score, norm_score)
            print()
    logger.info("ranking candidates based on similarity...")
    RANKING_FUNC = lambda x: (x[1],x[2])
    top_ranked_scores = sorted(scores, key=RANKING_FUNC, reverse=True)

    # create a json dump for clustering
    dump = [{'question':t[0][0],'sentence': t[0][1], "article_id":t[0][2], "sentence_id":t[0][3], 'rank': i+1, 'score': t[1], 'norm_score': t[2]} for i, t in enumerate(top_ranked_scores)]

    filename = os.path.join(DIR_PATH, args.outfile)

    # if os.path.exists(filename):
    #     append_write = 'a' # append if already exists
    # else:
    #     append_write = 'w' # make a new file if not
    # with open(filename, append_write) as f:
    #     json.dump(dump,f)

    with open(filename, 'w') as f:
        json.dump(dump, f)
