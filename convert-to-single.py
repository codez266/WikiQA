import numpy as np
import pdb
def convert(files):
    for f in files:
        scores = np.loadtxt(f+'.ref', dtype=np.int32)
        scores=np.append(scores, [[0,0,0,0]], axis=0)
        scores_compact = np.zeros((1,4), dtype=np.int32)
        prev = scores[0][0]
        prevlabel = 0
        for j in scores:
            if prev != j[0]:
                scores_compact=np.append(scores_compact, [[prev, 0, 0, prevlabel]], axis=0)
                prevlabel = 0
            prevlabel = 1 if prevlabel == 1 else j[3]
            #pdb.set_trace()
            prev = j[0]
        np.savetxt(f+'2.ref', scores_compact[1:], fmt='%d')

convert(['WikiQA-train', 'WikiQA-dev', 'WikiQA-test'])
