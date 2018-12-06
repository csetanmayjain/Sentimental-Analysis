from sklearn.feature_extraction.text import CountVectorizer
from sklearn import neighbors
#from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
a=open("clean_ElectionGujrat2017.txt")
data=a.readlines()
l=len(data)
x=[]
y=[]
for i in range(0,l):
    sum=0
    temp=data[i].split("|")
    x.append(temp[3])
    lenf=len(temp)
    f=temp[lenf-1].split("\n")
    y.append(f[0])
tfidf=TfidfVectorizer()
x=tfidf.fit_transform(x)
x=x.todense()
knn=neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(x,y)
pred=knn.predict(x)
correct=0
for i in range(0,len(y)):
    if(pred[i] == y[i]):
        correct+=1       
print "Accuracy:- ",(float(correct)/len(y))*100
