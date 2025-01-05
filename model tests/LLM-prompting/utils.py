import os

ANN_DIR = "/content/drive/Shareddrives/Folklore/Folklore and Fauna Project/data/Gold Data/Gold Data 1"
CONTEXT_WINDOW_SIZE = 500

def get_paths():
  dict_of_paths = {}
  for root, dirs, files in os.walk(ANN_DIR):
    for file in files:
      if file.endswith(".txt"):
        filename = os.path.join(root, file)
        matching_txt_file = filename[:-4] + '.ann'
        dict_of_paths[matching_txt_file] = filename

  print(f"dict_of_paths: {dict_of_paths}")
  return dict_of_paths

def dict_of_annotations(path):
  '''
  path: filepath for the .ann file
  return: a dictionary for the annotation file that includes all of the flora and fauna in the file,
  and a tuple of their motifs and a span of the tag in order to obtain a context window
  return: {flora/fauna: tuple(protag/antag, good/evil, span)}
  '''
  file_dict = {}  # dict to return
  tag_dict = {}  # dict mapping each tag to its flora/fauna
  tag_spans = {} # dict of all of the spans for each tag
  motif_tuple = ()  # tuple of motifs for each flora/fauna
  prev_attributed_tag = None # saving the tag aka flora/fauna that has the attribute in case the next attribute doesn't match

  # open the file
  with open(path, 'r') as file:
    for line in file:
      if line[0] == 'T':
        # add all the flora/fauna to the dictionary
        file_dict[line.split()[1]] = None
        tag_dict[line.split()[0]] = line.split()[1]
        tag_spans[line.split()[0]] = (line.split()[2], line.split()[3])

      elif line[0] == 'A':
        # map the motifs as tuples (protag/antag, good/evil) to their animal
        curr_attributed_tag = line.split()[2]
        # edgecase: if the annotator only added one of the two motifs for a tag
        if prev_attributed_tag != curr_attributed_tag and len(motif_tuple) != 0:
          motif_tuple += (None, tag_spans[prev_attributed_tag])
          file_dict[tag_dict[prev_attributed_tag]] = motif_tuple
          motif_tuple = ()
        # if the tuple for this flora/fauna is currently empty
        if len(motif_tuple) == 0:
          motif_tuple = (line.split()[-1],)
        # if it just has one, add the good/evil feature, add the tuple to the dict then wipe the tuple
        elif len(motif_tuple) == 1:
          motif_tuple += (line.split()[-1],)
          motif_tuple += (tag_spans[curr_attributed_tag],)
          file_dict[tag_dict[curr_attributed_tag]] = motif_tuple
          motif_tuple = ()
        prev_attributed_tag = curr_attributed_tag
    return file_dict

def context_windows(path, animal_attribute):
  '''
  given the path of the .txt file and the animal attributes from the corresponding .ann file
  return the modified animal_attributes as {flora/fauna: tuple(protag/antag, good/evil, context window)}
  '''
  # FIGURE OUT THE EDGECASE WHERE THERE ARE NO ANIMAL_ATTRIBUTES OR SOMETHING IS NONE
  if animal_attribute == None:
    return None

  # getting the text from the txt file
  with open(path, 'r') as file:
    text = ''
    for line in file:
      text = text + line

  # for each flora/fauna in the dict of flora/fauna
  for key in animal_attribute:
    if animal_attribute[key] is not None:
      # finding the span
      span = animal_attribute[key][2]
      # creating the initial and final indices for the context window
      initial_index = int(span[0]) - CONTEXT_WINDOW_SIZE
      final_index = int(span[1]) + CONTEXT_WINDOW_SIZE
      # later, we might want consider shifting the entire context window
      # currently, this just shortens the window if the tag is near the beginning or end of the text
      if initial_index < 0:
        initial_index = 0
      if final_index >= len(text):
        final_index = len(text) - 1

      # get the context in terms of characters
      context = ''
      for char in range(initial_index, final_index):
        # CONSIDER FINDING A WAY TO DO THIS ALONG WORDS SO WE DON'T HAVE CONTEXTS LIKE "his is a contex"
        context = context + text[char]

      animal_attribute[key] = animal_attribute[key][:-1] + (context, )
  return animal_attribute