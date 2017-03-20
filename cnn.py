import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D, Merge
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pdb

TRAIN_PART = 0.7
DEV_PART = 0.2
nb_filter = 300
filter_length = 3
hidden_dims=300
batch_size=32
epochs=1
class TextCNN:
	def __init__(self, data_dir, files):
		self.MAX_NB_WORDS = 50000
		#texts, labels = [], []
		texts = np.empty(shape=(0,2))
		labels = np.array([], dtype=np.int32)
		#bytetounicode = lambda x: x.decode('utf-8')
		for name in files:
			#text = np.genfromtxt(data_dir+name, delimiter='\t', dtype=None, invalid_raise=False)
			#text = np.genfromtxt(data_dir+name, delimiter='\t', converters={0: lambda x:x.decode('utf8'), 1:bytetounicode}, dtype=None, invalid_raise=False)
			#text = np.asarray([list(x) for x in text])
			#texts = np.append(texts, text[:,0:2], axis=0)
			#labels = np.append(labels, text[:,2])
			text, label = TextCNN.read_from_txt(data_dir+name)
			texts = np.append(texts, text, axis=0)
			labels = np.append(labels, label)
		self.texts = texts
		self.labels = labels
		self.MAX_LENA = len(max(self.texts[:,1], key=lambda x:len(x)))
		self.MAX_LENQ = len(max(self.texts[:,0], key=lambda x:len(x)))
		print("Max len ques:%s\nMAX len ans:%s", self.MAX_LENQ, self.MAX_LENA)

	@staticmethod
	def read_from_txt(filename):
		texts,labels = [],[]
		with open(filename) as fin:
			for l in fin:
				line = l.strip().split('\t')
				texts.append(line[0:2])
				labels.append(line[2])
		return texts,labels
	
	def fit(self):
		self.tokenizer = Tokenizer(nb_words=self.MAX_NB_WORDS)
		#pdb.set_trace()
		texts = self.texts.ravel().tolist()
		self.tokenizer.fit_on_texts(texts)
		self.sequencesq = self.tokenizer.texts_to_sequences(self.texts[:,0])
		self.sequencesa = self.tokenizer.texts_to_sequences(self.texts[:,1])
		self.word_index = self.tokenizer.word_index
		print('Found %s unique tokens.' % len(self.word_index))
		dataq = pad_sequences(self.sequencesq, maxlen=self.MAX_LENQ)
		dataa = pad_sequences(self.sequencesa, maxlen=self.MAX_LENA)
		train = int(self.texts.shape[0] * TRAIN_PART)
		dev = int(self.texts.shape[0] * DEV_PART)
		print("Train & test: %s and %s" %( train, dev))
		self.x_train = [dataa[0:train], dataq[0:train]]
		#num_samples = self.labels.shape[0]
		#self.labels = self.labels.reshape((-1, num_samples, 1))
		self.y_train = self.labels[0:train]
		self.x_dev = [dataa[train:train+dev], dataq[train:train+dev]]
		self.y_dev = self.labels[train:train+dev]
		self.x_test = [dataa[train:], dataq[train:]]
		self.y_test = self.labels[train:]

	def load_emb(self, embpath):
		try:
			self.w2v = Word2Vec.load_word2vec_format(embpath, binary=True)
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
		modela = Sequential()
		modela.add(Embedding(
			len(self.word_index) + 1,
			300,
			weights=[embedding_matrix],
			input_length = self.MAX_LENA,
			dropout=0.2,
			trainable=False
			))
		modela.add(Convolution1D(nb_filter=nb_filter,
			filter_length=filter_length,
			border_mode='valid',
			activation='relu',
			subsample_length=1
			))
		modela.add(GlobalMaxPooling1D())
		modela.add(Dense(hidden_dims, bias=False))
		#modela.add(Dropout(0.2))
		#modela.add(Activation('relu'))

		# question model

		modelq = Sequential()
		modelq.add(Embedding(
			len(self.word_index) + 1,
			300,
			weights=[embedding_matrix],
			input_length = self.MAX_LENQ,
			dropout=0.2,
			trainable=False
			))
		modelq.add(Convolution1D(nb_filter=nb_filter,
			filter_length=filter_length,
			border_mode='valid',
			activation='relu',
			subsample_length=1
			))
		modelq.add(GlobalMaxPooling1D())
		#modelq.add(Dense(hidden_dims))
		#modelq.add(Dropout(0.2))
		#modelq.add(Activation('relu'))
		# merge question and answer model, using dot product,
		# and take sigmoid

		model = Sequential()
		model.add(Merge([modela, modelq], mode='dot'))
		#model.add(Dense(1))
		model.add(Activation('sigmoid'))
		model.compile(loss='binary_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])
		#pdb.set_trace()
		model.fit(self.x_train, self.y_train,
				batch_size=batch_size,
				nb_epoch=epochs,
				validation_data=(self.x_dev, self.y_dev)
				)
		labels = model.predict(self.x_test)
		clf=None
		score=0
		for i, C in enumerate((100, 1, 0.01)):
			# turn down tolerance for short training time
			clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01)
			clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01)
			clf_l1_LR.fit(labels, self.y_test)
			clf_l2_LR.fit(labels, self.y_test)
			l1=clf_l1_LR.score(labels,self.y_test)
			l2=clf_l2_LR.score(labels,self.y_test)
			if l1 > score:
				clf = clf_l1_LR
				score = l1
			if l2 > score:
				clf = clf_l2_LR
				score = l2
			print(l1, " ", l2)
		preds = clf.predict(labels).astype(int)
		y_test = self.y_test.astype(int)
		print(classification_report(y_test, preds))
		pdb.set_trace()

s=TextCNN('data/', ['WikiQASent-train.bal', 'WikiQASent-dev.bal', 'WikiQASent-test.bal'])
s.CNN('/home/Btech13/sumit.cs13/GoogleNews-vectors-negative300.bin')
