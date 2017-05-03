import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Flatten
from keras.layers import Convolution1D, MaxPooling1D, Merge
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from gensim.models.keyedvectors import KeyedVectors
import pdb

TRAIN_PART = 0.7
DEV_PART = 0.2
nb_filter = 3
filter_length = 3
hidden_dims=300
batch_size=32
epochs=4

class TextCNN:
	def __init__(self, data_dir, files):
		self.MAX_NB_WORDS = 50000
		texts = np.empty(shape=(0,2))
		labels = np.array([], dtype=np.int32)
		self.texts, self.labels = [[],[],[]], [[], [], []]
		self.texts[0], self.labels[0] = TextCNN.generateData(data_dir, files[0])
		self.texts[1], self.labels[1] = TextCNN.generateData(data_dir, files[1])
		self.texts[2], self.labels[2] = TextCNN.generateData(data_dir, files[2])
		self.MAX_LENA, self.MAX_LENQ = 0, 0
		for t in self.texts:
			maxa = len(max(t[:,1], key=lambda x:len(x)))
			maxq = len(max(t[:,0], key=lambda x:len(x)))
			self.MAX_LENA = max(self.MAX_LENA, maxa)
			self.MAX_LENQ = max(self.MAX_LENQ, maxq)

	@staticmethod
	def generateData(data_dir, filename):
		texts = np.empty(shape=(0,2))
		labels = np.array([], dtype=np.int32)
		text, label = TextCNN.read_from_txt(data_dir+filename)
		texts = np.append(texts, text, axis=0)
		labels = np.append(labels, label)
		return texts, labels

	@staticmethod
	def read_from_txt(filename):
		texts,labels = [],[]
		with open(filename) as fin:
			for l in fin:
				line = l.strip().split('\t')
				texts.append(line[0:2])
				labels.append(int(line[2]))
		return texts,labels

	def fit(self):
		self.tokenizer = Tokenizer(nb_words=self.MAX_NB_WORDS)
		#pdb.set_trace()
		texts = []
		for text in self.texts:
			texts += text.ravel().tolist()
		self.tokenizer.fit_on_texts(texts)
		self.word_index = self.tokenizer.word_index
		print('Found %s unique tokens.' % len(self.word_index))
		ros = RandomOverSampler(random_state=42)
		self.texts[0], self.labels[0] = ros.fit_sample(self.texts[0], self.labels[0])
		self.texts[1], self.labels[1] = ros.fit_sample(self.texts[1], self.labels[1])
		self.x = [[],[],[]]
		for i, t in enumerate(self.texts):
			train_q = pad_sequences(self.tokenizer.texts_to_sequences(t[:,0]), self.MAX_LENQ)
			train_a = pad_sequences(self.tokenizer.texts_to_sequences(t[:,1]), self.MAX_LENA)
			self.x[i] = [train_a, train_q]
		pdb.set_trace()

	def load_emb(self, embpath):
		try:
			self.w2v = KeyedVectors.load_word2vec_format(embpath, binary=True)
			#self.w2v = Word2Vec.load(embpath)
		except Exception as e:
			pdb.set_trace()
			self.w2v = None
			print("Failed to load word2vec")

	def get_embedding_matrix(self):
		embedding_matrix = np.zeros((len(self.word_index) + 1, 300))
		for word, i in self.word_index.items():
			try:
				embedding_vector = self.w2v[word]
				embedding_matrix[i] = embedding_vector
			except Exception as e:
				pass
		return embedding_matrix

	def CNN(self, embpath):
		self.fit()
		self.load_emb(embpath)
		embedding_matrix = self.get_embedding_matrix()
		# answer model
		branches = []
		train_xs = []
		dev_xs = []
		test_xs = []
		for filter_len in [2,3,4]:
			branch = Sequential()
			branch.add(Embedding(
				len(self.word_index) + 1,
				300,
				weights=[embedding_matrix],
				input_length = self.MAX_LENA,
				trainable=False
				))
			branch.add(Convolution1D(nb_filter=nb_filter,
				filter_length = filter_len,
				border_mode = 'valid',
				activation = 'relu',
				subsample_length = 1
				))
			branch.add(MaxPooling1D(pool_length=2))
			branch.add(Flatten())
			branches.append(branch)
			train_xs.append(self.x[0][0])
			dev_xs.append(self.x[1][0])
			test_xs.append(self.x[2][0])

		for filter_len in [2,3,4]:
			branch = Sequential()
			branch.add(Embedding(
				len(self.word_index) + 1,
				300,
				weights=[embedding_matrix],
				input_length = self.MAX_LENQ,
				trainable=False
				))
			branch.add(Convolution1D(nb_filter=nb_filter,
				filter_length = filter_len,
				border_mode = 'valid',
				activation = 'relu',
				subsample_length = 1
				))
			branch.add(MaxPooling1D(pool_length=2))
			branch.add(Flatten())
			branches.append(branch)
			train_xs.append(self.x[0][1])
			dev_xs.append(self.x[1][1])
			test_xs.append(self.x[2][1])
			

			model = Sequential()
			model.add(Merge(branches, mode='concat'))
			model.add(Dense(100))
			#model.add(Dropout(0.2))
			model.add(Activation('relu'))
			model.add(Dense(1))
			model.add(Activation('sigmoid'))
			model.compile(loss='binary_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])
			#pdb.set_trace()
			model.fit(train_xs, self.labels[0],
				batch_size=batch_size,
				nb_epoch=epochs,
				validation_data=(dev_xs, self.labels[1])
				)
			labels = model.predict(test_xs)
			clf=None
			score=0
			for i, C in enumerate((100, 1, 0.01)):
			# turn down tolerance for short training time
				clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01)
				clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01)
				clf_l1_LR.fit(labels, self.labels[2])
				clf_l2_LR.fit(labels, self.labels[2])
				l1=clf_l1_LR.score(labels,self.labels[2])
				l2=clf_l2_LR.score(labels,self.labels[2])
				if l1 > score:
					clf = clf_l1_LR
					score = l1
				if l2 > score:
					clf = clf_l2_LR
					score = l2
				print(l1, " ", l2)
			preds = clf.predict(labels).astype(int)
			y_test = self.labels[2].astype(int)
			print(classification_report(y_test, preds))
			pdb.set_trace()

s=TextCNN('data/old/', ['WikiQASent-train.txt', 'WikiQASent-dev.txt', 'WikiQASent-test.txt'])
s.CNN('/home/Btech13/sumit.cs13/GoogleNews-vectors-negative300.bin')
