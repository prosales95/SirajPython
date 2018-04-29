from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

clf1 = tree.DecisionTreeClassifier()
clf2 = SVC(kernel="linear", C=0.025)
clf3 = MLPClassifier(alpha=1)
clf4 = GaussianNB()

# CHALLENGE - create 3 more classifiers...
# 1
# 2
# 3

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X, Y)
clf4 = clf4.fit(X, Y)

score = clf1.score(X, Y)

print(score)

predict1 = clf1.predict([[180, 120, 39]])
predict2 = clf1.predict([[180, 120, 45]])
predict3 = clf3.predict([[190, 120, 54]])
predict4 = clf4.predict([[190, 120, 54]])

print(predict1)
print(predict2)
print(predict3)
print(predict4)
