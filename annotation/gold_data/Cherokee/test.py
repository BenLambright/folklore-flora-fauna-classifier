def dict_of_annotations(path):
  '''
  path: filepath for the .ann file
  return: a dictionary for the annotation file that includes all of the flora and fauna in the file, and a tuple of their motifs
  '''
  file_dict = {}  # dict to return
  tag_dict = {}  # dict mapping each tag to its flora/fauna
  motif_tuple = ()  # tuple of motifs for each flora/fauna
  prev_attributed_tag = None # saving the tag aka flora/fauna that has the attribute in case the next attribute doesn't match

  # open the file
  with open(path, 'r') as file:
    for line in file:

      if line[0] == 'T':
        # add all the flora/fauna to the dictionary
        file_dict[line.split()[1]] = None
        tag_dict[line.split()[0]] = line.split()[1]

      elif line[0] == 'A':
        # map the motifs as tuples (protag/antag, good/evil) to their animal
        curr_attributed_tag = line.split()[2]
        # edgecase: if the annotator only added one of the two motifs for a tag
        if prev_attributed_tag != curr_attributed_tag and len(motif_tuple) != 0:
          file_dict[tag_dict[prev_attributed_tag]] = motif_tuple
          motif_tuple = ()
        # if the tuple for this flora/fauna is currently empty
        if len(motif_tuple) == 0:
          motif_tuple = (line.split()[-1],)
        # if it just has one, add the good/evil feature, add the tuple to the dict then wipe the tuple
        elif len(motif_tuple) == 1:
          motif_tuple += (line.split()[-1],)
          file_dict[tag_dict[curr_attributed_tag]] = motif_tuple
          motif_tuple = ()
        prev_attributed_tag = curr_attributed_tag

    return file_dict


if __name__ == "__main__":
  d = dict_of_annotations("BearsRaceWithTurtle.ann")
  print(d)