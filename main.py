import functions as f
import os, nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('universal_tagset')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')

sentence_list = []
basepath = 'training/'
for entry in os.listdir(basepath):
    if os.path.isfile(os.path.join(basepath, entry)):
        sent, sentences = f.parse_task_1_2(basepath+entry, only_sentences=True)
        sentence_list.append(sent)

tokens = f.ie_preprocess(sentence_list)

#grammar = "NP: {<DET|PRON\$>?<ADJ>*<NOUN>+}"
grammar2 = r"""
  NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
  PP: {<IN><NP>}               # Chunk prepositions followed by NP
  VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
  CLAUSE: {<NP><VP>}           # Chunk NP, VP
  """
cp = nltk.RegexpParser(grammar2)

chunks = []
for t in tokens:
    chunk = cp.parse(t)
    chunk2 = nltk.ne_chunk(chunk)
    chunks.append(chunk2)


x = 1