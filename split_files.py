import pdb
def gen_qa_pairs(names, stoplist):
	for n in names:
		fi = open(n+'.txt')
		fo = open(n+'.rf', 'w')
		prev, labellist = "", []
		i = 0
		alist = []
		for line in fi:
			sent = line.strip().split("\t")
			if prev != sent[0]:
				if i != 0:
					qlist = [i for i in prev.split() if i not in stoplist]
					flatalist = [j for i in alist for j in i]
					label = 1 if 1 in labellist else 0
					lineout = " ".join(qlist) + "\t" + " ".join(flatalist) + "\t" + str(label)
					fo.write(lineout + "\n")
					alist = []
					labellist = []
				else:
					i = i + 1
			prev = sent[0]
			alist.append([i for i in sent[1].split() if i not in stoplist])
			labellist.append(int(sent[2]))
		qlist = [i for i in prev.split() if i not in stoplist]
		flatalist = [j for i in alist for j in i]
		label = 1 if 1 in labellist else 0
		lineout = " ".join(qlist) + "\t" + " ".join(flatalist) + "\t" + str(label)
		fo.write(lineout + "\n")

		fi.close()
		fo.close()

def gen_qa_pairs_bal(names, stoplist):
	for n in names:
		fi = open(n+'.txt')
		fo = open(n+'.bal', 'w')
		prev, labellist = "", []
		samples = {0:'', 1:'','q':''}
		i = 0
		alist = []
		for line in fi:
			sent = line.strip().split("\t")
			if prev != sent[0]:
				if i!=0:
					qlist = [i for i in prev.split() if i not in stoplist]
					lineout = ""
					if samples[1]:
						alist = [i for i in samples[1].split() if i not in stoplist]
						lineout = lineout + " ".join(qlist) + "\t" + " ".join(alist) + "\t" + "1" + "\n"
					elif samples[0]:
						alist = [i for i in samples[0].split() if i not in stoplist]
						lineout = lineout + " ".join(qlist) + "\t" + " ".join(alist) + "\t" + "0" +"\n"
					fo.write(lineout)
					samples[0], samples[1] = '', ''
				else:
					i = i+1
			prev = sent[0]
			label = int(sent[2])
			if not samples[label]:
				samples[label] = sent[1]
		fi.close()
		fo.close()
def genstoplist(stopfile):
	stop = []
	with open(stopfile) as sf:
		for line in sf:
			stop.append(line.strip())
	return set(stop)

def preproc():
	slist = genstoplist('data/short-stopwords.txt')
	gen_qa_pairs_bal(['data/old/WikiQASent-train', 'data/old/WikiQASent-dev', 'data/old/WikiQASent-test'],
			slist)

preproc()
