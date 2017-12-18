"""
__description__: Consider top ranked sentences from the abstracts of the articles and make them the main point. For each of those top ranked sentences, get similarity score for sentences from the corresponding texts. Make the highest scored articles as as supporting points
"""

import json
import argparse
from itertools import groupby
from nltk import word_tokenize
from convolutionkernel import *
from collections import defaultdict
from utils import narratives_clustering


parser = argparse.ArgumentParser(description="Cluster text using k-means")
parser.add_argument('-i', '--input', type=str, required=True, help="Input JSON file containing the corpus")
parser.add_argument('-a', '--abstract', type=str, required=True, help="Input json file containing sentence from abstracts")
parser.add_argument('-o', '--output', type=str, required=True, help="File path to write the output to. Output contains of clusters of input text.")
parser.add_argument('-q', '--question', required=True, help='Input reference question')
parser.add_argument('-l', '--limit', type=int, required=False, default=1, help="Number of main points to print. ")
parser.add_argument('-v', '--w2vfile', type=str, required=True, help='word vectors file' )
parser.add_argument('-w', '--window', type=int, required=False, default=2, help="The size of the window to be used to print the context")

if __name__ == '__main__':

    args = parser.parse_args()

    W2VFILE = args.w2vfile

    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    abstractsfile = os.path.join(DIR_PATH, args.abstract)
    with open(abstractsfile, 'r') as f:
        data = json.load(f)

    if len(data) < 15:
        paragraph_limit = len(data)
    else:
        paragraph_limit = 15

    data = data[:paragraph_limit]
    data.sort(key=lambda x: x['rank'])

    articles = []
    for article in open(args.input):
        articles.append(json.loads(article))

    corpus = {}
    for art in open(args.input):
        article = json.loads(art)
        corpus[article['id']] = article

    print('Loading embeddings')
    embeddings = Embeddings(W2VFILE, phrase_similarity=0.3, word_similarity=0.2)
    print('Loading kernel')
    kernel = ConvolutionKernel(embeddings)

    article_ids = [abstract["article_id"] for abstract in data]
    article_ids_set = set(article_ids)
    print(article_ids_set)
    op = narratives_clustering(article_ids_set, corpus)

    # sort as per length of clusters
    aligned = sorted(op.values(), key=lambda x: len(x), reverse=True)

    narrative_cluster = aligned[0]
    narrative = [article_ids[idx] for idx in sorted([article_ids.index(x) for x in narrative_cluster])]


    article_titles = list(map(lambda x:(x, corpus[x]['title']), narrative_cluster))
    primary_titles = []
    for article_id, title in article_titles:
        kernel_score, norm_score = kernel.similarity(args.question, title)
        primary_titles.append((title, norm_score, kernel_score))
    print(primary_titles)
    if primary_titles:
        primary_title = sorted(primary_titles, key=lambda x:x[1], reverse=True)[0][0]


    alt_narrative_cluster = []
    alt_narrative = []
    # check if there are more than 2 clusters
    if len(aligned) >= 2:
        alt_narrative_cluster = aligned[1]
        alt_narrative = [article_ids[idx] for idx in sorted([article_ids.index(x) for x in alt_narrative_cluster])]

    if len(alt_narrative) < 3:
        num_alt_pts = len(alt_narrative)
    else:
        num_alt_pts = 3

    article_titles = list(map(lambda x:(x, corpus[x]['title']), alt_narrative[:num_alt_pts]))
    alt_primary_titles = []
    for article_id, title in article_titles:
        kernel_score, norm_score = kernel.similarity(args.question, title)
        alt_primary_titles.append((title, norm_score, kernel_score))
    print(alt_primary_titles)
    if alt_primary_titles:
        alt_primary_title = sorted(alt_primary_titles, key=lambda x:x[1], reverse=True)[0][0]

    # for each highest ranked sentence from the abstracts fetch the most similar sentences from the text
    # and write it to output
    # Write Narrative
    with open(args.output, 'w') as f:
        if narrative:
            f.write('{}\n\n'.format(primary_title))
        for idx in narrative:
            abstract = list(filter(lambda x: idx==x['article_id'], data))[0]
            assert(abstract['article_id'] == idx)
            scores_sents = []

            for obj in list(filter(lambda x: x["id"] == abstract["article_id"], articles)):
                # compute similarity between the sentence from the abstract and each sentence in the article text
                for i, sentence in enumerate(obj['text_sentences']):
                    try:
                        score, norm_score =  kernel.similarity(abstract["sentence"], sentence, collapse_phrases=True, debug=False)
                        scores_sents.append((i, sentence, score, norm_score))
                    except:
                        continue;

            scores_sents.sort(key=lambda x: (x[2], x[3]), reverse=True)
            f.write("{} \t {}".format(abstract['sentence'].strip('\n'), abstract['article_id']))
            f.write('\n')

            context_start_offset = 0
            if abstract['question'].startswith('Who'+' '):
                context_start_offset = -1

            # args.limit indicates number of main supporting points.
            for i in range(args.limit):
                  if scores_sents[i][1] != abstract["sentence"]:
                    for j in range(scores_sents[i][0]+context_start_offset, scores_sents[i][0]+args.window+1):
                                for obj in list(filter(lambda x: x["id"] == abstract["article_id"], articles)):
                                    if j>=0 and j<=len(obj['text_sentences'])-1:
                                        if j==scores_sents[i][0]:
                                            f.write("{} \t {}-{}, {}-{}".format(obj['text_sentences'][j].strip('\n'), obj["id"], j,scores_sents[i][2], scores_sents[i][3]))
                                        else:
                                            f.write("{} \t {}-{}".format(obj['text_sentences'][j].strip('\n'), obj["id"], j))
                                        f.write('\n')
                    break;
            # paragraph break
            f.write('\n\n')

    # Alternative narrative
    with open(args.output, 'a') as f:
        if alt_narrative:
            f.write('{}\n\n'.format(alt_primary_title))

        for idx in alt_narrative:
            abstract = list(filter(lambda x: idx==x['article_id'], data))[0]
            assert(abstract['article_id'] == idx)
            scores_sents = []
            for obj in list(filter(lambda x: x["id"] == abstract["article_id"], articles)):
                for i, sentence in enumerate(obj['text_sentences']):
                    try:
                        score, norm_score =  kernel.similarity(abstract["sentence"], sentence, collapse_phrases=True, debug=False)
                        scores_sents.append((i, sentence, score, norm_score))
                    except:
                        continue;

            scores_sents.sort(key=lambda x: (x[2], x[3]), reverse=True)
            f.write("{} \t {}".format(abstract['sentence'].strip('\n'), abstract['article_id']))
            f.write('\n')

            context_start_offset = 0
            if abstract['question'].startswith('Who'+' '):
                context_start_offset = -1

            # args.limit indicates number of main points.
            for i in range(args.limit):
                if scores_sents[i][1] != abstract["sentence"]:
                    for j in range(scores_sents[i][0]+context_start_offset, scores_sents[i][0]+args.window+1):
                                for obj in list(filter(lambda x: x["id"] == abstract["article_id"], articles)):
                                    if j>=0 and j<=len(obj['text_sentences'])-1:
                                        if j==scores_sents[i][0]:
                                            f.write("{} \t {}-{}, {}-{}".format(obj['text_sentences'][j].strip('\n'), obj["id"], j,scores_sents[i][2], scores_sents[i][3]))
                                        else:
                                            f.write("{} \t {}-{}".format(obj['text_sentences'][j].strip('\n'), obj["id"], j))
                                        f.write('\n')
                    break;
            # paragraph break
            f.write('\n\n')
