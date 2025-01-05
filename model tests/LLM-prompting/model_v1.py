import os
import utils

#install OpenAi and get it setup
from openai import OpenAI
client = OpenAI(api_key="")

# get the current working directory
current_working_directory = os.getcwd()

# these are the strings that we will always input into the system to solicit our output
system_role_content = "You will be provided with an animal or plant. State whether the animal is a protagonist, antagonist, or default given the story and just print the word protagonist, antagonist, or default."
system_alignment_content = "You will be provided with an animal or plant. State whether the animal is a good, evil, or neutral given the story and just print the word good, evil, or neutral."

# loop through the document to get the path of each document and its annotations
# save the paths in the following dict: {annotated_doc_path: textfile_doc_path}

#all my gold data is also in 1 so I think it's a-ok
dict_of_paths = {}
dict_of_paths = utils.get_paths()

# now that we have the following data strucutre: {flora/fauna: tuple(protag/antag, good/evil, context window)}
# we can feed this into our model
def get_alignment_or_role(annotation_goal, character, context):
  '''
  Given the context window and the animal
  returns what the model predicts
  '''

  #print(annotation_goal, character, context)
  response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": annotation_goal
    },
    {
      "role": "user",
      "content": f"plant/animal: {character}. Story: {context}"
    }
  ],
  temperature=0.7,
  max_tokens=64,
  top_p=1
)

  result = response.choices[0].message.content

  if len(result.split()) == 1:
    return result
  else:
    # FIND A BETTER WAY TO DEBUG CASES WHERE THE MODEL PRODUCES SENTENCES LATER!
    return result.split()[0]

#This could be fun if we wanted to ask a bunch of questions I think
def most_common_prompt():
  stream = client.chat.completions.create(
      model="gpt-4",
      messages=[{"role": "user", "content": "What creatures are most commonly predicted as evil?"}],
      messages=[{"role": "user", "content": "What creatures are most commonly predicted as good?"}],
      messages=[{"role": "user", "content": "What are the most common creatures in our stories?"}],
      stream=True,
  )
  for part in stream:
      print(part.choices[0].delta.content or "")


def run():
  # version 2 of output stream
  # prepping metrics for stats
  predicted_alignment_list, actual_alignment_list, predicted_role_list, actual_role_list, culture_list = [], [], [], [], []
  cultures = ["Cherokee", 'Seneca', "Japanese", "Maori", "Filipino"]

  for file in dict_of_paths:
    animal_attributes = utils.dict_of_annotations(file)
    # converting to {flora/fauna: tuple(protag/antag, good/evil, context window)}
    new_attributes = utils.context_windows(dict_of_paths[file], animal_attributes)
    #finding culture of origin


    # have the model predict a result of each non-None attribute
    for character in new_attributes:
      character_tuple = new_attributes[character]
      # CURRENTLY NOT DEALING WITH ALL UN-ANNOTATED TAGS
      #ok, so this line doesn't work but I don't want to fix the tabs rn -- Autumn

      if character_tuple is not None:
          # testing role and calibrating for sklearn
          predicted_role = get_alignment_or_role(system_role_content, character, character_tuple[2])
          # testing alignment
          predicted_alignment = get_alignment_or_role(system_alignment_content, character, character_tuple[2])

          if predicted_alignment != None:
            if predicted_role != None:
              if character_tuple[0] != None:
                if character_tuple[1] != None:
                  # tabulating predicted and actual in cases where None is not present
                  predicted_alignment_list.append(predicted_alignment.lower())
                  actual_alignment_list.append(character_tuple[1].lower())
                  predicted_role_list.append(predicted_role.lower())
                  actual_role_list.append(character_tuple[0].lower())
                  file_name = file
                  for culture in cultures:
                    if file_name.find(culture) != -1:
                      culture_list.append(culture)

                  # print("predicted alignment: " + str(predicted_alignment_list))
                  # print("actual alignment: " + str(actual_alignment_list))
                  # print("predicted role: " + str(predicted_role_list))
                  # print("actual role: " + str(actual_role_list))

                #This was still throwing a Nonetype for some reason
                #if predicted_role != character_tuple[0] or predicted_alignment != character_tuple[1]:
                  # print('\n' + "predicted role: " + predicted_role + " actual role: " + character_tuple[0])
                  # print("predicted alignment: " + predicted_alignment + " actual alignment: " + character_tuple[1] + '\n')

  print("predicted alignment: " + str(predicted_alignment_list))
  print("actual alignment: " + str(actual_alignment_list))
  print("predicted role: " + str(predicted_role_list))
  print("actual role: " + str(actual_role_list))

  return predicted_alignment_list, actual_alignment_list, predicted_role_list, actual_role_list, culture_list, file_name