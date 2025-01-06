from collections import Counter
import numpy as np
import organize_data
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import leaves_list, linkage


#right now everything is switched to one instead of adding one...interesting!
organize_data.animal_fact_1()
train = organize_data.Get_counts_and_instances()
train.extract_counts(organize_data.get_X_train(), organize_data.get_Y_train())
organize_data.tree_fact_1()

bigrams_vectorizer = DictVectorizer()
fit_bigrams = bigrams_vectorizer.fit_transform(train.get_bigrams())

train_bigs_Linear = np.array(train.get_bigs_list())

trigrams_vectorizer = DictVectorizer()
train_trigrams_vectorized = trigrams_vectorizer.fit_transform(train.get_trigrams())

mammals_vectorizer = DictVectorizer()
fit_mammals = mammals_vectorizer.fit_transform(train.get_mammals())

trees_and_mammals_vectorizer = DictVectorizer()
fit_trees_and_mammals = trees_and_mammals_vectorizer.fit_transform(train.get_trees_and_mammals())


train_labels_array = np.array(train.fixed_labels)

clfBigrams = MultinomialNB()
clfBigrams2 = MultinomialNB(alpha=.5)
clfBigrams3 = MultinomialNB(alpha=.25)
clfBigrams.fit(fit_bigrams, train_labels_array)
clfBigrams2.fit(fit_bigrams, train_labels_array)
clfBigrams3.fit(fit_bigrams, train_labels_array)

clfTrigrams = MultinomialNB()
clfTrigrams2 = MultinomialNB(alpha=.5)
clfTrigrams3 = MultinomialNB(alpha=.25)
clfTrigrams.fit(train_trigrams_vectorized, train_labels_array)
clfTrigrams2.fit(train_trigrams_vectorized, train_labels_array)
clfTrigrams3.fit(train_trigrams_vectorized, train_labels_array)

clfMammals = MultinomialNB()
clfMammals2 = MultinomialNB(alpha=.5)
clfMammals3 = MultinomialNB(alpha=.25)
clfMammals.fit(fit_mammals, train_labels_array)
clfMammals2.fit(fit_mammals, train_labels_array)
clfMammals3.fit(fit_mammals, train_labels_array)

clfMammalsAndTrees = MultinomialNB()
clfMammalsAndTrees2 = MultinomialNB(alpha = .5)
clfMammalsAndTrees3 = MultinomialNB(alpha =.25)
clfMammalsAndTrees.fit(fit_trees_and_mammals, train_labels_array)
clfMammalsAndTrees2.fit(fit_trees_and_mammals, train_labels_array)
clfMammalsAndTrees3.fit(fit_trees_and_mammals, train_labels_array)


#I mislabeled everything, multi is just complement

bigs_complement = ComplementNB()
bigs_complement2 = ComplementNB(alpha=.5)
bigs_complement3 = ComplementNB(alpha=.25)
bigs_complement.fit(fit_bigrams, train_labels_array)
bigs_complement2.fit(fit_bigrams, train_labels_array)
bigs_complement3.fit(fit_bigrams, train_labels_array)

multiTrigrams = ComplementNB()
multiTrigrams2 = ComplementNB(alpha=.5)
multiTrigrams3 = ComplementNB(alpha=.25)
multiTrigrams.fit(train_trigrams_vectorized, train_labels_array)
multiTrigrams2.fit(train_trigrams_vectorized, train_labels_array)
multiTrigrams3.fit(train_trigrams_vectorized, train_labels_array)

multiMammals = ComplementNB()
multiMammals2 = ComplementNB(alpha=0.5)
multiMammals3 = ComplementNB(alpha=.25)
multiMammals.fit(fit_mammals, train_labels_array)
multiMammals2.fit(fit_mammals, train_labels_array)
multiMammals3.fit(fit_mammals, train_labels_array)

mammals_and_trees_comp = ComplementNB()
mammals_and_trees_comp2 = ComplementNB(alpha=0.5)
mammals_and_trees_comp3 = ComplementNB(alpha=.25)
mammals_and_trees_comp.fit(fit_trees_and_mammals, train_labels_array)
mammals_and_trees_comp2.fit(fit_trees_and_mammals, train_labels_array)
mammals_and_trees_comp3.fit(fit_trees_and_mammals, train_labels_array)

rf_bigrams = RandomForestClassifier()
rf_bigrams2 = RandomForestClassifier(max_depth=2)
rf_bigrams3 = RandomForestClassifier(max_depth=4)
rf_bigrams.fit(fit_bigrams, train_labels_array)
rf_bigrams2.fit(fit_bigrams, train_labels_array)
rf_bigrams3.fit(fit_bigrams, train_labels_array)

rf_trigrams = RandomForestClassifier()
rf_trigrams2 = RandomForestClassifier(max_depth=2)
rf_trigrams3 = RandomForestClassifier(max_depth=4)
rf_trigrams.fit(train_trigrams_vectorized, train_labels_array)
rf_trigrams2.fit(train_trigrams_vectorized, train_labels_array)
rf_trigrams3.fit(train_trigrams_vectorized, train_labels_array)

organize_data.animal_fact_2()
rf_mammals = RandomForestClassifier()
rf_mammals2 = RandomForestClassifier(max_depth=2)
rf_mammals3 = RandomForestClassifier(max_depth=4)
rf_mammals.fit(fit_mammals, train_labels_array)
rf_mammals2.fit(fit_mammals, train_labels_array)
rf_mammals3.fit(fit_mammals, train_labels_array)

rf_trees_mammals = RandomForestClassifier()
rf_trees_mammals2 = RandomForestClassifier(max_depth=2)
rf_trees_mammals3 = RandomForestClassifier(max_depth=4)
rf_trees_mammals.fit(fit_trees_and_mammals, train_labels_array)
rf_trees_mammals2.fit(fit_trees_and_mammals, train_labels_array)
rf_trees_mammals3.fit(fit_trees_and_mammals, train_labels_array)

organize_data.bear_fact_1()
lr_bigrams = LogisticRegression()
lr_bigrams2 = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)
lr_bigrams3 = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
lr_bigrams.fit(fit_bigrams, train_labels_array)
lr_bigrams2.fit(fit_bigrams, train_labels_array)
lr_bigrams3.fit(fit_bigrams, train_labels_array)


lr_trigrams = LogisticRegression()
lr_trigrams2 = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)
lr_trigrams3 = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
lr_trigrams.fit(train_trigrams_vectorized, train_labels_array)
lr_trigrams2.fit(train_trigrams_vectorized, train_labels_array)
lr_trigrams3.fit(train_trigrams_vectorized, train_labels_array)


lr_mammals = LogisticRegression()
lr_mammals2 = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)
lr_mammals3 = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
lr_mammals.fit(fit_mammals, train_labels_array)
lr_mammals2.fit(fit_mammals, train_labels_array)
lr_mammals3.fit(fit_mammals, train_labels_array)


lr_trees_and_mammals = LogisticRegression()
lr_trees_and_mammals2 = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)
lr_trees_and_mammals3 = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
lr_trees_and_mammals.fit(fit_trees_and_mammals, train_labels_array)
lr_trees_and_mammals2.fit(fit_trees_and_mammals, train_labels_array)
lr_trees_and_mammals3.fit(fit_trees_and_mammals, train_labels_array)

print("done with train")
test = organize_data.Get_counts_and_instances()
test.extract_counts(organize_data.get_x_test(), organize_data.get_y_test())

bigs_transformed = bigrams_vectorizer.transform(test.get_bigrams())
trigs_transformed = trigrams_vectorizer.transform(test.get_trigrams())
mammals_transformed = mammals_vectorizer.transform(test.get_mammals())
trees_and_mammals_transformed = trees_and_mammals_vectorizer.transform(test.get_trees_and_mammals())
test_labels_array = np.array(test.fixed_labels)


nb_bigs_prediction = clfBigrams.predict(bigs_transformed)
nb_bigs_prediction2 = clfBigrams.predict(bigs_transformed)
nb_bigs_prediction3 = clfBigrams.predict(bigs_transformed)

nb_trigs_prediction = clfTrigrams.predict(trigs_transformed)
nb_trigs_prediction2 = clfTrigrams.predict(trigs_transformed)
nb_trigs_prediction3 = clfTrigrams.predict(trigs_transformed)


nb_mammals_prediction = clfMammals.predict(mammals_transformed)
nb_mammals_prediction2 = clfMammals.predict(mammals_transformed)
nb_mammals_prediction3 = clfMammals.predict(mammals_transformed)

nb_mammals_trees_prediction = clfMammalsAndTrees.predict(trees_and_mammals_transformed)
nb_mammals_trees_prediction2 = clfMammalsAndTrees.predict(trees_and_mammals_transformed)
nb_mammals_trees_prediction3 = clfMammalsAndTrees.predict(trees_and_mammals_transformed)

complement_bigs_pred = bigs_complement.predict(bigs_transformed)
complement_bigs_pred2 = bigs_complement.predict(bigs_transformed)
complement_bigs_pred3 = bigs_complement.predict(bigs_transformed)

multi_trigs_pred = multiTrigrams.predict(trigs_transformed)
multi_trigs_pred2 = multiTrigrams.predict(trigs_transformed)
multi_trigs_pred3 = multiTrigrams.predict(trigs_transformed)


multi_mammal_pred = multiMammals.predict(mammals_transformed)
multi_mammal_pred2 = multiMammals.predict(mammals_transformed)
multi_mammal_pred3 = multiMammals.predict(mammals_transformed)

complement_mammals_trees_pred = mammals_and_trees_comp.predict(trees_and_mammals_transformed)
complement_mammals_trees_pred2 = mammals_and_trees_comp.predict(trees_and_mammals_transformed)
complement_mammals_trees_pred3 = mammals_and_trees_comp.predict(trees_and_mammals_transformed)

rf_bigrams_pred = rf_bigrams.predict(bigs_transformed)
rf_bigrams_pred2 = rf_bigrams.predict(bigs_transformed)
rf_bigrams_pred3 = rf_bigrams.predict(bigs_transformed)

rf_trigrams_pred = rf_trigrams.predict(trigs_transformed)
rf_trigrams_pred2 = rf_trigrams.predict(trigs_transformed)
rf_trigrams_pred3 = rf_trigrams.predict(trigs_transformed)

rf_mammals_pred = rf_mammals.predict(mammals_transformed)
rf_mammals_pred2 = rf_mammals.predict(mammals_transformed)
rf_mammals_pred3 = rf_mammals.predict(mammals_transformed)

rf_trees_mammals_pred = rf_trees_mammals.predict(trees_and_mammals_transformed)
rf_trees_mammals_pred2 = rf_trees_mammals.predict(trees_and_mammals_transformed)
rf_trees_mammals_pred3 = rf_trees_mammals.predict(trees_and_mammals_transformed)

lr_bigrams_pred = lr_bigrams.predict(bigs_transformed)
lr_bigrams_pred2 = lr_bigrams.predict(bigs_transformed)
lr_bigrams_pred3 = lr_bigrams.predict(bigs_transformed)

lr_trigrams_pred = lr_trigrams.predict(trigs_transformed)
lr_trigrams_pred2 = lr_trigrams.predict(trigs_transformed)
lr_trigrams_pred3 = lr_trigrams.predict(trigs_transformed)

lr_mammals_pred = lr_mammals.predict(mammals_transformed)
lr_mammals_pred2 = lr_mammals.predict(mammals_transformed)
lr_mammals_pred3 = lr_mammals.predict(mammals_transformed)

lr_trees_and_mammals_pred = lr_trees_and_mammals.predict(trees_and_mammals_transformed)
lr_trees_and_mammals_pred2 = lr_trees_and_mammals.predict(trees_and_mammals_transformed)
lr_trees_and_mammals_pred3 = lr_trees_and_mammals.predict(trees_and_mammals_transformed)


nations: list[str] = ["australian_ethnic", "chinese", "indian", "russian", "philippine", "japanese", "arabic",
                      "north_american_native"]

print()
print("**************************************************************")
print("This is for test scoring")
print()
print("************************************************************")
print("TEST Multinomial NB bigrams, trigrams, animals, mammals+trees")
print("************************************************************")
print(classification_report(test_labels_array, nb_bigs_prediction, zero_division=0.0))
print(classification_report(test_labels_array, nb_bigs_prediction2, zero_division=0.0))
print(classification_report(test_labels_array, nb_bigs_prediction3, zero_division=0.0))
print(classification_report(test_labels_array, nb_trigs_prediction, zero_division=0.0))
print(classification_report(test_labels_array, nb_trigs_prediction2, zero_division=0.0))
print(classification_report(test_labels_array, nb_trigs_prediction3, zero_division=0.0))
print(classification_report(test_labels_array, nb_mammals_prediction, zero_division=0.0))
print(classification_report(test_labels_array, nb_mammals_prediction2, zero_division=0.0))
print(classification_report(test_labels_array, nb_mammals_prediction3, zero_division=0.0))
print(classification_report(test_labels_array, nb_mammals_trees_prediction, zero_division=0.0))
print(classification_report(test_labels_array, nb_mammals_trees_prediction2, zero_division=0.0))
print(classification_report(test_labels_array, nb_mammals_trees_prediction3, zero_division=0.0))

print()
print()
print("************************************************************")
print("TEST complement NB bigrams, trigrams, mammals, mammals+trees")
print("************************************************************")
print(classification_report(test_labels_array, complement_bigs_pred, zero_division=0.0))
print(classification_report(test_labels_array, complement_bigs_pred2, zero_division=0.0))
print(classification_report(test_labels_array, complement_bigs_pred3, zero_division=0.0))
print(classification_report(test_labels_array, multi_trigs_pred, zero_division=0.0))
print(classification_report(test_labels_array, multi_trigs_pred2, zero_division=0.0))
print(classification_report(test_labels_array, multi_trigs_pred3, zero_division=0.0))
print(classification_report(test_labels_array, multi_mammal_pred, zero_division=0.0))
print(classification_report(test_labels_array, multi_mammal_pred2, zero_division=0.0))
print(classification_report(test_labels_array, multi_mammal_pred3, zero_division=0.0))
print(classification_report(test_labels_array, complement_mammals_trees_pred, zero_division=0.0))
print(classification_report(test_labels_array, complement_mammals_trees_pred2, zero_division=0.0))
print(classification_report(test_labels_array, complement_mammals_trees_pred3, zero_division=0.0))

print()
print()
print("************************************************************")
print("TEST Random Forest bigrams, trigrams, mammals, mammals+trees")
print("************************************************************")
print(classification_report(test_labels_array, rf_bigrams_pred,  zero_division=0.0))
print(classification_report(test_labels_array, rf_bigrams_pred2,  zero_division=0.0))
print(classification_report(test_labels_array, rf_bigrams_pred3,  zero_division=0.0))
print(classification_report(test_labels_array, rf_trigrams_pred,  zero_division=0.0))
print(classification_report(test_labels_array, rf_trigrams_pred2, zero_division=0.0))
print(classification_report(test_labels_array, rf_trigrams_pred3,  zero_division=0.0))
print(classification_report(test_labels_array, rf_mammals_pred,  zero_division=0.0))
print(classification_report(test_labels_array, rf_mammals_pred2,  zero_division=0.0))
print(classification_report(test_labels_array, rf_mammals_pred3,  zero_division=0.0))
print(classification_report(test_labels_array, rf_trees_mammals_pred,  zero_division=0.0))
print(classification_report(test_labels_array, rf_trees_mammals_pred2,  zero_division=0.0))
print(classification_report(test_labels_array, rf_trees_mammals_pred3,  zero_division=0.0))



print()
print()
print("************************************************************")
print("TEST Logistic Regression bigrams, trigrams, mammals, mammals+trees")
print("************************************************************")
print()
print(classification_report(test_labels_array, lr_bigrams_pred,  zero_division=0.0))
print(classification_report(test_labels_array, lr_bigrams_pred2,  zero_division=0.0))
print(classification_report(test_labels_array, lr_bigrams_pred3,  zero_division=0.0))
print(classification_report(test_labels_array, lr_trigrams_pred, zero_division=0.0))
print(classification_report(test_labels_array, lr_trigrams_pred2,  zero_division=0.0))
print(classification_report(test_labels_array, lr_trigrams_pred3, zero_division=0.0))
print(classification_report(test_labels_array, lr_mammals_pred, zero_division=0.0))
print(classification_report(test_labels_array, lr_mammals_pred2, zero_division=0.0))
print(classification_report(test_labels_array, lr_mammals_pred3, zero_division=0.0))
print(classification_report(test_labels_array, lr_trees_and_mammals_pred, zero_division=0.0))
print(classification_report(test_labels_array, lr_trees_and_mammals_pred2, zero_division=0.0))
print(classification_report(test_labels_array, lr_trees_and_mammals_pred3, zero_division=0.0))


print("************************************************************")
print("done with test")
print("Dev below")
print("************************************************************")

dev = organize_data.Get_counts_and_instances()
dev.extract_counts(organize_data.get_x_dev(), organize_data.get_y_dev())

bigs_transformed_dev = bigrams_vectorizer.transform(dev.get_bigrams())
trigs_transformed_dev = trigrams_vectorizer.transform(dev.get_trigrams())
mammals_transformed_dev = mammals_vectorizer.transform(dev.get_mammals())
trees_and_mammals_transformed_dev = trees_and_mammals_vectorizer.transform(dev.get_trees_and_mammals())
dev_labels_array = np.array(dev.fixed_labels)

nb_bigs_prediction1 = clfBigrams.predict(bigs_transformed_dev)
nb_bigs_prediction2 = clfBigrams2.predict(bigs_transformed_dev)
nb_bigs_prediction3 = clfBigrams3.predict(bigs_transformed_dev)

nb_trigs_prediction1 = clfTrigrams.predict(trigs_transformed_dev)
nb_trigs_prediction2 = clfTrigrams.predict(trigs_transformed_dev)
nb_trigs_prediction3 = clfTrigrams.predict(trigs_transformed_dev)

nb_mammals_prediction1 = clfMammals.predict(mammals_transformed_dev)
nb_mammals_prediction2 = clfMammals.predict(mammals_transformed_dev)
nb_mammals_prediction3 = clfMammals.predict(mammals_transformed_dev)

nb_mammals_trees_prediction1 = clfMammalsAndTrees.predict(trees_and_mammals_transformed_dev)
nb_mammals_trees_prediction2 = clfMammalsAndTrees.predict(trees_and_mammals_transformed_dev)
nb_mammals_trees_prediction3 = clfMammalsAndTrees.predict(trees_and_mammals_transformed_dev)

complement_bigs_pred1 = bigs_complement.predict(bigs_transformed_dev)
complement_bigs_pred2 = bigs_complement.predict(bigs_transformed_dev)
complement_bigs_pred3= bigs_complement.predict(bigs_transformed_dev)

multi_trigs_pred1 = multiTrigrams.predict(trigs_transformed_dev)
multi_trigs_pred2 = multiTrigrams.predict(trigs_transformed_dev)
multi_trigs_pred3 = multiTrigrams.predict(trigs_transformed_dev)

multi_mammal_pred1 = multiMammals.predict(mammals_transformed_dev)
multi_mammal_pred2 = multiMammals.predict(mammals_transformed_dev)
multi_mammal_pred3 = multiMammals.predict(mammals_transformed_dev)

complement_mammals_trees_pred1 = mammals_and_trees_comp.predict(trees_and_mammals_transformed_dev)
complement_mammals_trees_pred2 = mammals_and_trees_comp.predict(trees_and_mammals_transformed_dev)
complement_mammals_trees_pred3 = mammals_and_trees_comp.predict(trees_and_mammals_transformed_dev)

rf_bigrams_pred1 = rf_bigrams.predict(bigs_transformed_dev)
rf_bigrams_pred2 = rf_bigrams.predict(bigs_transformed_dev)
rf_bigrams_pred3 = rf_bigrams.predict(bigs_transformed_dev)

rf_trigrams_pred1 = rf_trigrams.predict(trigs_transformed_dev)
rf_trigrams_pred2 = rf_trigrams.predict(trigs_transformed_dev)
rf_trigrams_pred3 = rf_trigrams.predict(trigs_transformed_dev)

rf_mammals_pred1 = rf_mammals.predict(mammals_transformed_dev)
rf_mammals_pred2 = rf_mammals.predict(mammals_transformed_dev)
rf_mammals_pred3 = rf_mammals.predict(mammals_transformed_dev)

rf_trees_mammals_pred1 = rf_trees_mammals.predict(trees_and_mammals_transformed_dev)
rf_trees_mammals_pred2 = rf_trees_mammals.predict(trees_and_mammals_transformed_dev)
rf_trees_mammals_pred3 = rf_trees_mammals.predict(trees_and_mammals_transformed_dev)

lr_bigrams_pred1 = lr_bigrams.predict(bigs_transformed_dev)
lr_bigrams_pred2 = lr_bigrams.predict(bigs_transformed_dev)
lr_bigrams_pred3 = lr_bigrams.predict(bigs_transformed_dev)

lr_trigrams_pred1 = lr_trigrams.predict(trigs_transformed_dev)
lr_trigrams_pred2 = lr_trigrams.predict(trigs_transformed_dev)
lr_trigrams_pred3 = lr_trigrams.predict(trigs_transformed_dev)

lr_mammals_pred1 = lr_mammals.predict(mammals_transformed_dev)
lr_mammals_pred2 = lr_mammals.predict(mammals_transformed_dev)
lr_mammals_pred3 = lr_mammals.predict(mammals_transformed_dev)

lr_trees_and_mammals1 = lr_trees_and_mammals.predict(trees_and_mammals_transformed_dev)
lr_trees_and_mammals2 = lr_trees_and_mammals.predict(trees_and_mammals_transformed_dev)
lr_trees_and_mammals3 = lr_trees_and_mammals.predict(trees_and_mammals_transformed_dev)

print()
print()
print("************************************************************")
print("DEV Multinomial NB bigrams, trigrams, animals, mammals+trees")
print("************************************************************")
print(classification_report(dev_labels_array, nb_bigs_prediction1, zero_division=0.0))
print(classification_report(dev_labels_array, nb_bigs_prediction2, zero_division=0.0))
print(classification_report(dev_labels_array, nb_bigs_prediction3, zero_division=0.0))
print(classification_report(dev_labels_array, nb_trigs_prediction1, zero_division=0.0))
print(classification_report(dev_labels_array, nb_trigs_prediction2, zero_division=0.0))
print(classification_report(dev_labels_array, nb_trigs_prediction3, zero_division=0.0))
print(classification_report(dev_labels_array, nb_mammals_prediction1, zero_division=0.0))

#this consistently gives an error and I don't know why, so I'm nixing it#
print(classification_report(dev_labels_array, nb_mammals_prediction2, zero_division=0.0))
print(classification_report(dev_labels_array, nb_mammals_prediction3, zero_division=0.0))
print(classification_report(dev_labels_array, nb_mammals_trees_prediction1, zero_division=0.0))
print(classification_report(dev_labels_array, nb_mammals_trees_prediction2, zero_division=0.0))
print(classification_report(dev_labels_array, nb_mammals_trees_prediction3, zero_division=0.0))

print()
print()
print("************************************************************")
print("DEV Complement NB bigrams, trigrams, mammals, mammals+trees")
print("************************************************************")
print(classification_report(dev_labels_array, complement_bigs_pred1, zero_division=0.0))
print(classification_report(dev_labels_array, complement_bigs_pred2, zero_division=0.0))
print(classification_report(dev_labels_array, complement_bigs_pred3, zero_division=0.0))
print(classification_report(dev_labels_array, multi_trigs_pred1, zero_division=0.0))
print(classification_report(dev_labels_array, multi_trigs_pred2, zero_division=0.0))
print(classification_report(dev_labels_array, multi_trigs_pred3, zero_division=0.0))
print(classification_report(dev_labels_array, multi_mammal_pred1, zero_division=0.0))
print(classification_report(dev_labels_array, multi_mammal_pred2, zero_division=0.0))
print(classification_report(dev_labels_array, multi_mammal_pred3, zero_division=0.0))
print(classification_report(dev_labels_array, complement_mammals_trees_pred1, zero_division=0.0))
print(classification_report(dev_labels_array, complement_mammals_trees_pred2, zero_division=0.0))
print(classification_report(dev_labels_array, complement_mammals_trees_pred3, zero_division=0.0))


print("************************************************************")
print("Confusion Matrix for default complement nb ")
print("************************************************************")
disp = ConfusionMatrixDisplay.from_predictions(dev_labels_array, complement_bigs_pred1)
disp.plot(colorbar = False, cmap="plasma")
plt.show()



print()
print()
print("************************************************************")
print("DEV Random Forest bigrams, trigrams, mammals, mammals+trees")
print("************************************************************")
print(classification_report(dev_labels_array, rf_bigrams_pred1, zero_division=0.0))
print(classification_report(dev_labels_array, rf_bigrams_pred2, zero_division=0.0))
print(classification_report(dev_labels_array, rf_bigrams_pred3, zero_division=0.0))
print(classification_report(dev_labels_array, rf_trigrams_pred1, zero_division=0.0))
print(classification_report(dev_labels_array, rf_trigrams_pred2, zero_division=0.0))
print(classification_report(dev_labels_array, rf_trigrams_pred3, zero_division=0.0))
print(classification_report(dev_labels_array, rf_mammals_pred1, zero_division=0.0))
print(classification_report(dev_labels_array, rf_mammals_pred2, zero_division=0.0))
print(classification_report(dev_labels_array, rf_mammals_pred3, zero_division=0.0))
print(classification_report(dev_labels_array, rf_trees_mammals_pred1, zero_division=0.0))
print(classification_report(dev_labels_array, rf_trees_mammals_pred2, zero_division=0.0))
print(classification_report(dev_labels_array, rf_trees_mammals_pred3, zero_division=0.0))


print()
print()
print("************************************************************")
print("DEV Logistic Regression bigrams, trigrams, mammals, mammals+trees")
print("************************************************************")
print(classification_report(dev_labels_array, lr_bigrams_pred1, zero_division=0.0))
print(classification_report(dev_labels_array, lr_bigrams_pred2, zero_division=0.0))
print(classification_report(dev_labels_array, lr_bigrams_pred3, zero_division=0.0))
print(classification_report(dev_labels_array, lr_trigrams_pred1, zero_division=0.0))
print(classification_report(dev_labels_array, lr_trigrams_pred2, zero_division=0.0))
print(classification_report(dev_labels_array, lr_trigrams_pred3, zero_division=0.0))
print(classification_report(dev_labels_array, lr_mammals_pred1, zero_division=0.0))
print(classification_report(dev_labels_array, lr_mammals_pred2, zero_division=0.0))
print(classification_report(dev_labels_array, lr_mammals_pred3, zero_division=0.0))
print(classification_report(dev_labels_array, lr_trees_and_mammals1, zero_division=0.0))
print(classification_report(dev_labels_array, lr_trees_and_mammals2, zero_division=0.0))
print(classification_report(dev_labels_array, lr_trees_and_mammals3, zero_division=0.0))


organize_data.tree_fact_2()




