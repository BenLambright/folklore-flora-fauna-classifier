import utils
import model_v1
import matplotlib.pyplot as plt

# calculating statistics from previous output using predicted_alignment_list, actual_alignment_list, predicted_role_list, actual_role_list
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from collections import Counter

DICT_OF_PATHS = utils.get_paths()

def get_basic_stats():
    predicted_alignment_list, actual_alignment_list, predicted_role_list, actual_role_list, culture_list, file_name = model_v1.run()

    # accuracy
    alignment_accuracy = accuracy_score(actual_alignment_list, predicted_alignment_list)
    role_accuracy = accuracy_score(actual_role_list, predicted_role_list)

    # recall
    alignment_recall = recall_score(actual_alignment_list, predicted_alignment_list, average='macro', zero_division=1)
    role_recall = recall_score(actual_role_list, predicted_role_list, average='macro', zero_division=1)

    # precision
    alignment_precision = precision_score(actual_alignment_list, predicted_alignment_list, average='macro', zero_division=1)
    role_precision = precision_score(actual_role_list, predicted_role_list, average='macro', zero_division=1)

    # f1
    alignment_f1 = f1_score(actual_alignment_list, predicted_alignment_list, average='macro', zero_division=1)
    role_f1 = f1_score(actual_role_list, predicted_role_list, average='macro', zero_division=1)

    # calculating the number of tags in each dataset, not including 'None' categories
    actual_alignment_counts = Counter(actual_alignment_list)
    actual_role_counts = Counter(actual_role_list)


    cherokee_alignment = []
    cherokee_predicted_alignment = []
    seneca_alignment = []
    seneca_predicted_alignment = []
    maori_alignment = []
    maori_predicted_alignment = []
    japanese_alignment = []
    japanese_predicted_alignment = []
    filipinio_alignment = []
    filipino_predicted_alignment = []

    # culture of origin statistics
    for i in range(len(culture_list)):
        if culture_list[i] == "Cherokee":
            cherokee_alignment.append(actual_alignment_list[i])
            cherokee_predicted_alignment.append(predicted_alignment_list[i])
        if culture_list[i] == "Seneca":
            seneca_alignment.append(actual_alignment_list[i])
            seneca_predicted_alignment.append(predicted_alignment_list[i])
        if culture_list[i] == "Maori":
            maori_alignment.append(actual_alignment_list[i])
            maori_predicted_alignment.append(predicted_alignment_list[i])
        if culture_list[i] == "Japanese":
            japanese_alignment.append(actual_alignment_list[i])
            japanese_predicted_alignment.append(predicted_alignment_list[i])
        if culture_list[i] == "Filipino":
            filipinio_alignment.append(actual_alignment_list[i])
            filipino_predicted_alignment.append(predicted_alignment_list[i])

    cherokee_alignment_accuracy = accuracy_score(cherokee_alignment, cherokee_predicted_alignment)
    seneca_alignment_accuracy = accuracy_score(seneca_alignment, seneca_predicted_alignment)
    maori_alignment_accuracy = accuracy_score(cherokee_alignment, cherokee_predicted_alignment)
    japanese_alignment_accuracy = accuracy_score(seneca_alignment, seneca_predicted_alignment)
    filipino_alignment_accuracy = accuracy_score(seneca_alignment, seneca_predicted_alignment)
    print(cherokee_alignment_accuracy)
    print(seneca_alignment_accuracy)
    print(maori_alignment_accuracy)
    print(japanese_alignment_accuracy)
    print(filipino_alignment_accuracy)
    print()
    print("alignment_accuracy: " + str(alignment_accuracy))
    print("role_accuracy: " + str(role_accuracy))
    print("alignment_recall: " + str(alignment_recall))
    print("role_recall: " + str(role_recall))
    print("alignment_precision: " + str(alignment_precision))
    print("role_precision: " + str(role_precision))
    print("alignment_f1: " + str(alignment_f1))
    print("role_f1: " + str(role_f1))
    print()
    print("count of tags in the dataset")
    print(actual_alignment_counts)
    print(actual_role_counts)

    return culture_list, file_name



# roles of each animal depending on culture of origin
def each_animal_accuracies():
    culture_list, file_name = get_basic_stats()

    # statistics on the common tags were, as well as number of tags including 'None' categories
    # also figuring out the common motifs for each tag
    all_tags = Counter()
    all_tags_alignments = Counter()
    all_tags_roles = Counter()
    tags_with_none = Counter()
    all_alignments_plus_none = Counter()
    all_roles_plus_none = Counter()
    cherokee_tags = Counter()
    japanese_tags = Counter()
    filipino_tags = Counter()
    maori_tags = Counter()
    seneca_tags = Counter()
    for file in DICT_OF_PATHS:
        # create a datastrucutre using {flora/fauna: tuple(protag/antag, good/evil, context window)} for tags in every file
        animal_attributes = utils.dict_of_annotations(file)
        all_tags.update(Counter(animal_attributes.keys()))
        for culture in culture_list:
            if file_name.find(culture) != 0:
                for tag in animal_attributes:
                    if animal_attributes[tag] is not None:
                        # count the tags
                        all_tags_alignments.update({(tag, animal_attributes[tag][0]):1})
                        all_tags_roles.update({(tag, animal_attributes[tag][1]):1})
                    if culture == "Maori":
                        maori_tags.update({tag: 1})
                        #maori_roles.update(animal_attributes[tag][1])
                    if culture == "Seneca":
                        seneca_tags.update({tag: 1})
                    if culture == "Cherokee":
                        cherokee_tags.update({tag: 1})
                    if culture == "Filipino":
                        filipino_tags.update({tag: 1})
                    if culture == "Japanese":
                        japanese_tags.update({tag: 1})


                    # count all the alignments and roles
                    if animal_attributes[tag][0] != None:
                        all_roles_plus_none.update({animal_attributes[tag][0]: 1})
                    else:
                        all_roles_plus_none.update({"None": 1})
                    if animal_attributes[tag][1] != None:
                        all_alignments_plus_none.update({animal_attributes[tag][1]: 1})
                    else:
                        all_alignments_plus_none.update({"None": 1})
            else:
                tags_with_none.update({tag: 1})

    print("including none variables:")
    print(tags_with_none)
    print(all_alignments_plus_none)
    print(all_roles_plus_none)
    print()
    print("count of all tags")
    print(all_tags)
    print(all_tags_alignments)
    print(all_tags_roles)
    print()
    print("culture counts")
    print("cherokee" + str(cherokee_tags.most_common(20)))
    print("jpn" + str(japanese_tags.most_common(20)))
    print("filipino" + str(filipino_tags.most_common(20)))
    print("maori" + str(maori_tags.most_common(20)))
    print("seneca" + str(seneca_tags.most_common(20)))

    print(sum(all_tags.values()))

    return all_tags, all_tags_alignments, all_tags_roles

def plot():
    all_tags, all_tags_alignments, all_tags_roles = each_animal_accuracies()

    # all tags
    tag_distribution = plt.figure(1)
    tag_distribution = plt.bar(list(all_tags.keys()), list(all_tags.values()))
    # Adjust x label to make it visible
    plt.xticks(rotation='vertical')
    plt.xticks(fontsize=7)

    # motifs for each tag
    alignments = plt.figure(2, figsize=(10, 10))
    # filtering all_tags_alignments if it has fewer than 5 of a given count
    filtered_counter = {key: count for key, count in all_tags_alignments.items() if key[1] != 'Default' and count > 2}
    x = [str(key) for key in list(filtered_counter.keys())]
    alignments = plt.bar(x, list(filtered_counter.values()))
    # Adjust label and size to make it visible
    plt.xticks(rotation='vertical')
    plt.xticks(fontsize=15)

    roles = plt.figure(2, figsize=(10, 10))
    # filtering all_tags_alignments if it has fewer than 5 of a given count
    filtered_counter = {key: count for key, count in all_tags_roles.items() if key[1] != 'Neutral' and count > 2}
    x = [str(key) for key in list(filtered_counter.keys())]
    alignments = plt.bar(x, list(filtered_counter.values()))
    # Adjust label and size to make it visible
    plt.xticks(rotation='vertical')
    plt.xticks(fontsize=15)


    plt.show()