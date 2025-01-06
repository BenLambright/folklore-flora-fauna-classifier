import nltk
import sklearn
import numpy as np
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import bigrams
from nltk.util import trigrams
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import ComplementNB


##############################################################################
"Organizing the data"

nations: list[str] = ["australian_ethnic", "chinese", "indian", "russian", "philippine", "japanese", "arabic"]


#this is for me if I want to change the nations
all_nations = ['south_african', 'lithuanian', 'jewish', 'serbian',
'scandinavian', 'english', 'nordic', 'korean',
'hungarian', 'turkish', 'north_american_native', 'chinese',
 'romanian', 'portuguese', 'indian', 'canadian_native',
 'welsh', 'french', 'italian', 'dutch', 'ukrainian',
 'polish', 'russian', 'zimbabwe', 'swedish',
 'sami', 'belgian', 'new_zealand_native', 'danish',
  'cataloanian', 'japanese', 'croatian', 'brazilian',
  'norwegian', 'armenian', 'nigerian', 'arabic',
  'slavic', 'australian_ethnic', 'estonian', 'pakistani',
  'german', 'hawaiian', 'bukovinian', 'philippine',
  'finnish', 'bulgarian', 'irish', 'icelandic',
  'celtic', 'spanish', 'albanian', 'greek',
  'tanzanian', 'maori', 'czechoslovak', 'scottish']

folk_tales = pd.read_csv(r"data/folk_tales_deduplicated.csv",
                         encoding="UTF-8", usecols=(1, 2, 3))

mammals = pd.read_csv(r"data/MDD_v1.11_6649species aut edit.csv",
                      encoding='UTF-8')


trees = pd.read_csv(r"data/Trees.csv", encoding='UTF-8')
tree_set = set()
for tree in trees['Tree Common Names']:
    tree_set.add(tree)


# organize the folk tales
folk_tales = folk_tales.apply(lambda row: row[folk_tales['nation'].isin(nations)])
folk_tales = folk_tales.dropna()
X = folk_tales["nation"]
y = folk_tales["text"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

x_test, x_dev, Y_test, Y_dev = train_test_split(X_test, y_test, test_size=0.50, random_state=1)

# organize the mammals
mammals.dropna()
mammal_set = set()
for mammal in mammals['simple name']:
    if mammal != 'nan':
        mammal_set.add(mammal)

mammal_set.discard(np.nan)


#############################################


class Folktale_instances:
    "Make instances for extracting features & such"

    def __init__(
            self, label
    ) -> None:

        self.nation = label
        self.bigrams = Counter()
        self.bigramsVocab = set()
        self.trigrams = Counter()
        self.unique_vocabulary = Counter()
        self.mammal_vocabulary = Counter()
        #self.vocabulary = []
        self.mammal_tree_vocabulary = Counter()
        pass

    def extract_data(self, animals, story, trees) -> None:
        bigs = []
        trigs = []
        sentences = sent_tokenize(story)
        for sentence in sentences:
            bigs.append(list(bigrams(word_tokenize(sentence))))
            trigs.append(list(trigrams(word_tokenize(sentence))))
            lemmatizer = WordNetLemmatizer()
            for word in word_tokenize(sentence):
                #self.unique_vocabulary[word] = 1
                #self.vocabulary.append(word)
                fixed_word = lemmatizer.lemmatize(word)

                #This is what takes forever to run
                for animal in animals:
                    if animal == fixed_word or animal == word:
                        self.mammal_vocabulary[animal.lower()] = 1
                        self.mammal_tree_vocabulary[animal] = 1

                for tree in trees:
                    if tree == fixed_word or tree == word:
                        self.mammal_tree_vocabulary[tree] = 1

                #for tree in trees['Tree Common Names']:
                    #if tree == fixed_word or tree == word:
                        #self.tree_vocabulary[tree] += 1

        for bigram in bigs:
            for gram in bigram:
                self.bigrams[gram] += 1
                self.get_bigrams_set().add(gram)

        for trigram in trigs:
            for gram in trigram:
                self.trigrams[gram] += 1


    def get_nation(self) -> str:
        # because iterating through the dataset is a pain
        return self.nation

    def get_bigrams(self) -> Counter():
        return self.bigrams

    def get_bigrams_set(self) -> set:
        return self.bigramsVocab

    def get_trigrams(self) -> list:
        return self.trigrams

    def get_mammals(self) -> list:
        return self.mammal_vocabulary

    def get_mammals_and_trees(self) -> list:
        return self.mammal_tree_vocabulary


class Get_counts_and_instances:
    def __init__(
            self,
    ) -> None:
        self.instance_labels = []
        self.all_bigrams = []
        self.all_trigrams = []
        self.all_bigrams_list = []
        self.all_mammals = []
        self.all_trees_and_mammals = []
        self.fixed_labels = []
        pass

    def extract_counts(self, x, y) -> None:
        all_bigrams_set = set()
        for i in x.keys():
            instance = Folktale_instances(x.get(i))
            instance.extract_data(mammal_set, y.get(i), tree_set)
            self.all_mammals.append(instance.mammal_vocabulary)
            self.all_trees_and_mammals.append(instance.mammal_tree_vocabulary)
            self.all_bigrams.append(instance.get_bigrams())
            self.all_trigrams.append(instance.get_trigrams())
            self.fixed_labels.append(instance.nation)
            for gram in instance.get_bigrams_set():
                all_bigrams_set.add(gram)

        for gram_counter in self.all_bigrams:
            adding = []
            for gram in all_bigrams_set:
                if gram in gram_counter.keys():
                    adding.append(1)
                else:
                    adding.append(0)
            self.all_bigrams_list.append(adding)


    def get_bigrams(self) -> object:
        return self.all_bigrams

    def get_bigs_list(self) -> list:
        return self.all_bigrams_list

    def get_trigrams(self) -> object:
        return self.all_trigrams

    def get_mammals(self) -> object:
        return self.all_mammals

    def get_trees_and_mammals(self) -> object:
        return self.all_trees_and_mammals

    def get_instance_labels(self) -> list:
        #because this is a pain to extract with the dataframe
        return self.instance_labels



def get_X_train():
    return X_train

def get_Y_train():
    return y_train

def get_x_test():
    return x_test

def get_y_test():
    return Y_test

def get_x_dev():
    return x_dev

def get_y_dev():
    return Y_dev


def animal_fact_1():
    print("extracting data for training...")
    print()
    print("animal fact while waiting: ")
    print("the echidna is one of two egg-laying mammals. The other being the platypus.")
    print("The two share an ancestor,the monotreme, which was the first mammal species.")
    print("While the platypus is an aquatic animal,"
          "the echidna is land-based.")
    print("That likely means the echidna is one of the oldest existing land-based species,")
    print("as one of the rules of evolution is that when a species evolves to become land-based")
    print("it never re-evolves into being aquatic.")
    print("Don't think that means it's an ancestor of humans, though.")
    print("The montreme just happened to evolve as a mammal from some reptiles.")



def tree_fact_1():
    print()
    print("train data extracted.")
    print()
    print("tree fact: ")
    print("The Western Whitebark Pine is a keystone species facing extinction.")
    print("Many people assume a bear's favorite food is salmon, but this is false.")
    print("Bears will overwhelmingly choose Whitebark pine nuts over salmon.")
    print("Why is it a keystone species? Here are some reasons:")
    print()
    print("-it evolved alongside the Clark's Nutcracker (a bird like a crow)")
    print("-it can live between 500-1000 years, and it provides groundcover for snowpacks in the alpine")
    print("-it's central in the stories of many native nations where they coexist.")
    print("-they can provide us tons of data in the northern Rockies,")
    print("as the sentiment in the area formed before most living organisms, thus providing us with few fossils.")


def animal_fact_2():
    print()
    print("Animal Fact")
    print("Americans tend to think of squirrels as everyday creatures but this is a North American phenomenon.")
    print("In most of the world squirrels are critically endangered or have gone extinct.")
    print("It's because north american squirrels are so ")
    print("well adapted that we're able to understand other squirrels around the world and protect them.")
    print("My favorite squirrel fact is that they vocalize with chirps, which many people mistake as bird sounds.")
    print("If you hear a squirrel chirping, try chirping back! They're pretty chatty creatures...")
    print("though most eastern grey squirrel that dominate the Boston area are territorial.")
    print("So they're likely just saying 'HEY BIG HUMAN, GET OFF MY LAWN'")


def tree_fact_2():
    print()
    print("One last fact to finish~")
    print()
    print("tree fact: ")
    print("Defining 'what is a tree' is difficult.")
    print("The reason is many people merge shrubs and trees into one category.")
    print("Ex: most maples are actually shrubs.")
    print("Most naturalists use the presence of a central trunk as the defining feature of a tree.")
    print("However, then we have organisms like Aspens.")
    print("Aspens have a central mother tree. Each 'tree'  surrounding the mother is a fruiting body of that mother.")
    print("They're considered clones of the mother tree.")
    print("So they're one organism but look like a grove of different organisms.")
    print("Do you classify that as one tree? Multiple trees? Is it even a tree?")

def bear_fact_1():
    print()
    print("Bear Facts")
    print()
    print("One of my bosses was the top North American bear biologist.")
    print("He is one of my many inspirations behind this project.")
    print("Did you know that in most cultures around the world there are stories about mother bears?")
    print("This is likely because of a mother bear's extreme maternal instinct.")
    print("Most people think of bears as hibernators.")
    print("They're not true hibernaters at all. They're patrially awake the whole time.")
    print("And if a mother bear's instinct tells her there's danger nearby")
    print("then she'll go from sleep to running you down at 30mph in about 30 seconds.")

#This was for my own curiosity :)
#most_popular_animal = Counter()
#most_popular_tree = Counter()
#for counts in all_mammals:
    #for animal, amount in counts.items():
        #most_popular_animal[animal.lower()] += amount

#for counts in all_trees:
    #for tree, amount in counts.items():
        #most_popular_tree[tree.lower()] += amount



