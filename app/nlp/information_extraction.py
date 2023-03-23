import logging
import random
import spacy
from spacy import displacy 
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
from spacy.lang.en import English
import networkx as nx
import matplotlib.pyplot as plt

from ..utility.pre_processing import cleanhtml
class NlpAlgos:

    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")


    def POS_tagging(self, text):
        # create spacy doc
        doc = self.nlp(text)
        
        pos_tags_dict = {}
        # applying POS to each token
        for token in doc:
            pos_tags_dict[token.text] = token.pos_
        # filtering out tokens based on POS
        return pos_tags_dict

    def dependency_graph(self, text):
        doc = self.nlp(text)
        html = displacy.render(doc, style="dep", page=True, minify=True)
        html_final = cleanhtml(html)
        return {"raw_html" : html}

    def summarize(self, long_rev):
        # summ = spacy.load('en')
        long_rev = self.nlp(long_rev)

        keyword = []

        summary_response = {}

        stopwords = list(STOP_WORDS)
        pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
        for token in long_rev:
            if(token.text in stopwords or token.text in punctuation):
                continue
            if(token.pos_ in pos_tag):
                keyword.append(token.text)
        freq_word = Counter(keyword)
        summary_response["word_freq"] = freq_word.most_common(5)
        # Normalization
        # Each sentence is weighed based on the 
        # frequency of the token present in each sentence
        max_freq = Counter(keyword).most_common(1)[0][1]
        for word in freq_word.keys():  
                freq_word[word] = (freq_word[word]/max_freq)
        freq_word.most_common(5)

        # Strength of sentences

        sent_strength={}
        for sent in long_rev.sents:
            for word in sent:
                if word.text in freq_word.keys():
                    if sent in sent_strength.keys():
                        sent_strength[sent]+=freq_word[word.text]
                    else:
                        sent_strength[sent]=freq_word[word.text]
        # the nlargest function returns a list containing the top 3 sentences which are stored as summarized_sentences
        summarized_sentences = nlargest(3, sent_strength, key=sent_strength.get)
        final_sentences = [ w.text for w in summarized_sentences ]
        summary = ' '.join(final_sentences)
        summary_response["summarized"] = summary
        return summary_response


    def apply_ner(self, sentence):
        doc = self.nlp(sentence)
        ner = {}
        for ent in doc.ents:
            ner[ent.text] = ent.label_
        return ner
            

class KnowledgeGraph(NlpAlgos):

    def getSentences(self, text):
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        document = self.nlp(text)
        return [sent.string.strip() for sent in document.sents]


    def printToken(self, token):
        print(token.text, "->", token.dep_)


    def appendChunk(self, original, chunk):
        return original + ' ' + chunk


    def isRelationCandidate(self, token):
        deps = ["ROOT", "adj", "attr", "agent", "amod"]
        return any(subs in token.dep_ for subs in deps)


    def isConstructionCandidate(self, token):
        deps = ["compound", "prep", "conj", "mod"]
        return any(subs in token.dep_ for subs in deps)


    def processSubjectObjectPairs(self, tokens):
        subject = ''
        object = ''
        relation = ''
        subjectConstruction = ''
        objectConstruction = ''
        for token in tokens:
            self.printToken(token)
            if "punct" in token.dep_:
                continue
            if self.isRelationCandidate(token):
                relation = self.appendChunk(relation, token.lemma_)
            if self.isConstructionCandidate(token):
                if subjectConstruction:
                    subjectConstruction = self.appendChunk(subjectConstruction, token.text)
                if objectConstruction:
                    objectConstruction = self.appendChunk(objectConstruction, token.text)
            if "subj" in token.dep_:
                subject = self.appendChunk(subject, token.text)
                subject = self.appendChunk(subjectConstruction, subject)
                subjectConstruction = ''
            if "obj" in token.dep_:
                object = self.appendChunk(object, token.text)
                object = self.appendChunk(objectConstruction, object)
                objectConstruction = ''

        print (subject.strip(), ",", relation.strip(), ",", object.strip())
        return (subject.strip(), relation.strip(), object.strip())

    def processSentence(self, sentence):
        tokens = self.nlp(sentence)
        return self.processSubjectObjectPairs(tokens)

    def printGraph(self, triples):
        request_id = random.randint(0, 20)
        G = nx.Graph()
        for triple in triples:
            G.add_node(triple[0])
            G.add_node(triple[1])
            G.add_node(triple[2])
            G.add_edge(triple[0], triple[1])
            G.add_edge(triple[1], triple[2])

        pos = nx.spring_layout(G)
        plt.figure()
        nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
                node_size=500, node_color='seagreen', alpha=0.9,
                labels={node: node for node in G.nodes()})
        plt.axis('off')
        plt.savefig(f'graphs/knowledge_graph_{request_id}')
        return plt.show()


    def knowledge_graph(self, text):
        sentences = self.getSentences(text)
        triples = []
        print (text)
        for sentence in sentences:
            triples.append(self.processSentence(sentence))

        self.printGraph(triples)