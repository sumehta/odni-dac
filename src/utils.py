#!/usr/bin/env python

import re
import os
import itertools
#import ujson as json
import json
import collections

import spacy
import textacy

from sklearn.cluster import AffinityPropagation
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize

NLP_PARSER = None

COMMON_ABBREV = {u'artificial intelligence': u'AI'}
QUESTION_TEMPLATE = u"What developments related to something are most "+\
           u"impactful to the national security of the United States?"

def get_nlp_parser():
    global NLP_PARSER
    if NLP_PARSER is None:
        #NLP_PARSER = spacy.en.English(path="/home/rupen/nlp/nlp_data/spacy/en-1.1.0/")
        NLP_PARSER = spacy.load('en')
    return NLP_PARSER

dir_path = os.path.dirname(os.path.realpath(__file__))
STOPWORDS_FILEPATH = os.path.join(dir_path, "../data/en_es_pt_stopwords_v1.3.txt")
STOPWORDS = set(map(lambda x:x.strip().lower(), open(STOPWORDS_FILEPATH)))
STOPWORDS |= set([u'\u2019s'])


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
        nlp_parser = get_nlp_parser()
        doc = nlp_parser(text)
        if collapse_phrases:
            for ent in doc.ents:
                if ent.label_ != "O":
                    ent.merge(ent.root.tag_, ent.text, ent.label_)
            for ent in list(doc.noun_chunks):
                ent.merge(ent.root.tag_, ent.text, ent.label_)
        return _tokens(doc)


def is_stopword(token):
    if token['text'].lower() in STOPWORDS and token['lemma'].lower() in STOPWORDS:
        return True
    else:
        return False


def preprocess_text(doc, collapse_phrases=False):
    tokens = SpacyEnricher.getTokens(doc, collapse_phrases=collapse_phrases)
    _ignore_pos_tags = set([u"X", u"PUNCT", u"SPACE", u"SYM", u"NUM", u"EOL", u"DET", u"PART"])
    tokens = filter(lambda x: x['pos'] not in _ignore_pos_tags and not is_stopword(x) and len(x['lemma']) > 1, tokens)
    return list(map(lambda x:x['lemma'].lower(), tokens))


def preprocess_doc(doc, fieldname='text'):
    tokens = SpacyEnricher.getTokens(doc[fieldname], collapse_phrases=True)
    _ignore_pos_tags = set([u"X", u"PUNCT", u"SPACE", u"SYM", u"NUM", u"EOL", u"DET", u"PART"])
    tokens = filter(lambda x: x['pos'] not in _ignore_pos_tags and not is_stopword(x), tokens)
    return " ".join(list(map(lambda x:x['lemma'].lower(), tokens)))

def tfidf_affinity_propagation_clustering(docs, max_ngram=3, use_idf=True):
    tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocess_doc, max_df=0.9, min_df=0.1,
                                      use_idf=use_idf, ngram_range=(1,max_ngram))
    #builds a tf-idf matrix for the sentences
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    ap = AffinityPropagation()
    ap.fit(tfidf_matrix)
    clusters = collections.defaultdict(list)
    for i, label in enumerate(ap.labels_):
            clusters[label].append(docs[i]['id'])
    return dict(clusters)

def narratives_clustering(article_ids, corpus):
    selected_articles = [corpus[x] for x in article_ids]
    clusters = tfidf_affinity_propagation_clustering(selected_articles)
    # remove clusters with only 1 members
    clusters = {k:vs for k, vs in clusters.items() if len(vs) > 1}
    return clusters

def get_corpus_tags(corpus):
    tags = set()
    for article_id, article in corpus.items():
        tags.update(article['tags'])
    tags = {k: preprocess_text(k) for k in tags}
    tags = {k:v for k,v in tags.items() if v}
    return tags

def get_key_terms(target_question, kernel, corpus):
    nlp_parser = get_nlp_parser()
    target_question_keyterms = textacy.keyterms.sgrank(nlp_parser(target_question))
    target_question_keyterms = {k:v for k, v in target_question_keyterms}

    question_template_keyterms = textacy.keyterms.sgrank(nlp_parser(QUESTION_TEMPLATE))
    question_template_keyterms = {k:v for k, v in question_template_keyterms}

    topk = 1

    target_question_top_keyterms = {k:v for k, v in target_question_keyterms.items()\
                                    if k not in question_template_keyterms}
    target_question_top_keyterms = sorted(target_question_top_keyterms.items(), key=lambda x:x[1], reverse=True)[:topk]
    target_question_top_keyterms = {term: [] for term, score in target_question_top_keyterms}


    #print(target_question_top_keyterms_new)
    corpus_tags = get_corpus_tags(corpus)
    #target_question_top_keyterms = target_question_top_keyterms_new.copy()
    #print(target_question_top_keyterms)

    for top_term in target_question_top_keyterms.copy().keys():
        if len(top_term.split(" ")) > 1:
            top_term_tokens = preprocess_text(top_term)
        else:
            top_term_tokens = [top_term]
        scored_tags = {}
        for tag_key, tag_tokens in corpus_tags.items():
            scored_tags[tag_key] = kernel._embd.multiword_similarity(top_term_tokens, tag_tokens)
        scored_tags = filter(lambda x: x[1] is not None, scored_tags.items())
        scored_tags = sorted(scored_tags, key=lambda x:x[1], reverse=True)

        target_question_top_keyterms[top_term] = scored_tags[:topk]

    for term in target_question_top_keyterms.copy().keys():
        if term in COMMON_ABBREV:
            target_question_top_keyterms[COMMON_ABBREV[term].lower()] = []
    top_keyterms = set()
    for k, vs in target_question_top_keyterms.items():
        top_keyterms.add(k)
        top_keyterms.update(list(map(lambda x:x[0], vs)))
    return top_keyterms
