# sklearn mlcomp data set classification learning
# description: http://scikit-learn.org/stable/auto_examples/text/mlcomp_sparse_document_classification.html

from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn import metrics
from pprint import pprint

##---------1.  extract the 20 news_group dataset to /scikit_learn_data all categories  
newsgroup_train = fetch_20newsgroups(subset='train')  
#part categories  
categories = ['comp.graphics',  
 'comp.os.ms-windows.misc',  
 'comp.sys.ibm.pc.hardware',  
 'comp.sys.mac.hardware',  
 'comp.windows.x'];  
newsgroup_train = fetch_20newsgroups(subset = 'train',categories = categories)
newsgroup_test = fetch_20newsgroups(subset = 'test',categories = categories)
#print category names  

pprint(list(newsgroup_train.target_names))

##---------2. Extract feature
#newsgroup_train.data is the original documents, but need to extract the
#feature vectors inorder to model the text data

  
vectorizer = HashingVectorizer(stop_words = 'english',non_negative = True)  
fea_train = vectorizer.fit_transform(newsgroup_train.data)  
fea_test = vectorizer.fit_transform(newsgroup_test.data)  
  
  
#return feature vector 'fea_train' [n_samples,n_features]  
print 'Size of fea_train:' + repr(fea_train.shape)  
print 'Size of fea_train:' + repr(fea_test.shape)  
#11314 documents, 130107 vectors for all categories  
print 'The average feature sparsity is {0:.3f}%'.format(  
fea_train.nnz/float(fea_train.shape[0]*fea_train.shape[1])*100)
 

##---------3. Evaluation
# precision results
def calculate_result(actual,pred):  
    m_precision = metrics.precision_score(actual,pred)  
    m_recall = metrics.recall_score(actual,pred) 
    print 'predict info summary'  
    print 'precision:{0:.3f}'.format(m_precision)  
    print 'recall:{0:0.3f}'.format(m_recall) 
    print 'f1-score:{0:.3f}'.format(metrics.f1_score(actual,pred))
    
##---------4. Build Classifier
#4.1 Multinomial Naive Bayes Classifier  
print '*************************\n1. Naive Bayes\n*************************'  
  
newsgroups_test = fetch_20newsgroups(subset = 'test',  
                                     categories = categories)  
fea_test = vectorizer.fit_transform(newsgroup_test.data)
  
#create Classifier

clf = MultinomialNB(alpha = 0.01)   
clf.fit(fea_train,newsgroup_train.target)  
pred = clf.predict(fea_test)  
calculate_result(newsgroup_test.target,pred) 

  
#4.2 KNN Classifier  
 
print '*************************\n2. KNN\n*************************'  
knnclf = KNeighborsClassifier()#default with k=5  
knnclf.fit(fea_train,newsgroup_train.target)  
pred = knnclf.predict(fea_test);  
calculate_result(newsgroup_test.target,pred)

  
#4.3 SVM Classifier  
  
print '*************************\n3. SVM\n*************************'  
svclf = SVC(kernel = 'linear')#default with 'rbf'  
svclf.fit(fea_train,newsgroup_train.target)  
pred = svclf.predict(fea_test)  
calculate_result(newsgroup_test.target,pred)
 
#4.4 KMeans Cluster  
  
print '*************************\n4. KMeans\n*************************'  
pred = KMeans(n_clusters=5)  
pred.fit(fea_test)  
calculate_result(newsgroup_test.target,pred.labels_)  
