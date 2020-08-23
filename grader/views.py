from django.shortcuts import get_object_or_404, render, redirect
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse

from .models import Essay
from .forms import AnswerForm

from .utils.model import *
from .utils.helpers import *

import language_tool_python
from collections import Counter
import nltk
from nltk.corpus import stopwords
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')

import os
current_path = os.path.abspath(os.path.dirname(__file__))

# Create your views here.

def semantic_similarity(sentence1, sentence2):
	X_list = nltk.word_tokenize(sentence1)  # turn sentences into lists of words
	Y_list = nltk.word_tokenize(sentence2)

	sw = stopwords.words('english')  # create a list of non-keywords

	X_set = {w for w in X_list if not w in sw}  # turn lists of words into sets, and only include keywords
	Y_set = {w for w in Y_list if not w in sw} 
	
	# form a set containing keywords of both strings
	l1 = []
	l2 = []  
	rvector = X_set.union(Y_set)  
	for w in rvector:
		if w in X_set: l1.append(1) # create a vector 
		else: l1.append(0) 
		if w in Y_set: l2.append(1) 
		else: l2.append(0) 
	c = 0.0

	for i in range(len(rvector)): 
		c+= l1[i]*l2[i]
	geometric_mean = (sum(l1)*sum(l2))**0.5
	similarity = c / float(geometric_mean) # return number of keywords that are shared between the two sentences divided by the geometric mean number of	  keywords in each sentence
												 # in other words, number of shared keywords over the average number of total keywords
	print(similarity)
	return similarity


def essay(request, essay_id):
	essay = get_object_or_404(Essay, pk=essay_id)
	context = {
		"essay": essay,
	}
	return render(request, 'grader/essay.html', context)

def question(request):
	if request.method == 'POST':
		# create a form instance and populate it with data from the request:
		form = AnswerForm(request.POST)
		if form.is_valid():
			content = form.cleaned_data.get('answer')
			
			tool = language_tool_python.LanguageTool('en-US')
			matches = tool.check(content)
			incorrect = len(matches)
			
			word_count = len(nltk.word_tokenize(content))
			
			wordpercentage = ((word_count-incorrect)/word_count)*100
			
			tagged = nltk.pos_tag(content.split())
			print(tagged)
			counts = Counter(tag for word,tag in tagged)
			print(counts)
			past = counts["VBD"] + counts["VBN"]
			present = counts["VB"] + counts["VBZ"] + counts["VBP"]
			future = counts["MD"]

			totverbs = past + present + future
			totverbs = (max(max(present, future),past)/totverbs)*100
			
			scores = []
			tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
			sentences_essay = tokenizer.tokenize(content)
			print(sentences_essay)
			mean = 0
			for i in range(len(sentences_essay)):
				for j in range(i+1, len(sentences_essay)):
					mean += semantic_similarity(sentences_essay[i], sentences_essay[j])

			mean = (mean/(len(sentences_essay)*(len(sentences_essay)-1)/2))*100
			if len(content) > 20:
				num_features = 300
				model = word2vec.KeyedVectors.load_word2vec_format(os.path.join(current_path, "deep_learning_files/word2vec.bin"), binary=True)
				clean_test_essays = []
				clean_test_essays.append(essay_to_wordlist( content, remove_stopwords=True ))
				testDataVecs = getAvgFeatureVecs( clean_test_essays, model, num_features )
				testDataVecs = np.array(testDataVecs)
				testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

				lstm_model = get_model()
				lstm_model.load_weights(os.path.join(current_path, "deep_learning_files/final_lstm.h5"))
				preds = lstm_model.predict(testDataVecs)

				if math.isnan(preds):
					preds = 0
				else:
					preds = np.around(preds)

				if preds < 0:
					preds = 0
			else:
				preds = 0

			K.clear_session()
			essay = Essay.objects.create(
				content=content,
				score=preds,
				semantic=mean,
				tense=totverbs,
				accuracy = wordpercentage,
				wordcount = word_count
			)
		return redirect('essay', essay_id=essay.id)
	else:
		form = AnswerForm()

	context = {
		"form": form,
	}
	return render(request, 'grader/index.html', context)