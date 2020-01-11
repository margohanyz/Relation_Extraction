import functions as f
import os, nltk, re
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('universal_tagset')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
#nltk.download('ace')

sentence_list = []
basepath = 'training/'
for entry in os.listdir(basepath):
    if os.path.isfile(os.path.join(basepath, entry)):
        sent, sentences = f.parse_task_1_2(basepath+entry, only_sentences=True)
        sentence_list.append(sent)

tokens = f.ie_preprocess(sentence_list)

#grammar = "NP: {<DET|PRON\$>?<ADJ>*<NOUN>+}"
grammar2 = r"""
  PERSON: {<PERSON><PERSON>+}
  NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
  PP: {<IN><NP>}               # Chunk prepositions followed by NP
  VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
  CLAUSE: {<NP><VP>}           # Chunk NP, VP
  """
cp = nltk.RegexpParser(grammar2)

chunks = []
for t in tokens:
    chunk = nltk.ne_chunk(t)
    chunk2 = cp.parse(chunk)
    chunks.append(chunk2)

#print(type(chunks[4]))

IN = re.compile(r'.*\bin\b(?!\b.+ing)')
IN2 = re.compile(r'.*\bin\b.*')
X = re.compile(r'\b')

#for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
#    for rel in nltk.sem.extract_rels('ORG', 'LOC', doc, corpus='ieer', pattern = IN):
#        print(nltk.sem.rtuple(rel))

for rel in nltk.sem.extract_rels('PER', 'ORG', chunks, pattern = IN):
    print(nltk.sem.rtuple(rel))

x = 1
