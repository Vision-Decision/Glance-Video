import json
T=list(range(0,480,4))
R_yes={48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,128,132,136,140,156,160,172,176,180,192,208,228,236,240,244,248,252,256,260,264,268,272,280,284,312,316,320,360,364,368,372,376,408,428,456,460,472,476}
R_unc={36,40,44,124,144,148,468}
F_yes={120,132,136,140,144,152,164,168,184,188,196,200,204,216,220,308,324,328,380,384,392,396,400,404,408,412,432,440,444,448,452,456,460}
F_unc={72,100,232,292,304}
B_yes={28,76,80,84,88,92,96,108,116,128,136,140,188,204,212,232,476}
B_unc={12,160}
def lab(t,y,u): return 1 if t in y else (None if t in u else 0)
gt={"rabbit":[lab(t,R_yes,R_unc) for t in T],"squirrel":[lab(t,F_yes,F_unc) for t in T],"butterfly":[lab(t,B_yes,B_unc) for t in T],"car":[0 for t in T]}
json.dump({"t":T,"gt":gt,"how":"hand-labelled frame by frame by the author of the post; null = cannot tell from the frame"},open('bbb_gt.json','w'))
d=json.load(open('bbb4.json'))
P={k:[r['results'][i]['present_prob'] for r in d['rows']] for k,i in [('rabbit',0),('squirrel',1),('butterfly',2),('car',3)]}
from math import *
for k in P:
    ok=n=0;fp=fn=0
    for p,g in zip(P[k],gt[k]):
        if g is None: continue
        n+=1; ok+= (p>=.5)==(g==1); fp+= p>=.5 and g==0; fn+= p<.5 and g==1
    print(k,n,ok,'%.1f'%(100*ok/n),'fp',fp,'fn',fn,'pos',sum(1 for g in gt[k] if g==1))
