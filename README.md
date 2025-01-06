# folklore_origin_classifier
final project for an annotation course (COSI 230 - Brandeis University)

Developed annotation guidelines with a group to have students in the class hand-annotate folklore to identify and classify flora and fauny in stories using brat. Using these annotations, we attempted to classify the archetypal roles flora and fauna play in these stories.

---

## **Table of Contents**
1. [Dataset](#dataset)  
   - [Annotation Process](#annotation-process)  
2. [Models](#models)  
3. [Usage](#usage)  
   - [Installation](#installation)  
   - [Running the Code](#running-the-code)  
4. [Directory Structure](#directory-structure)  

---

## **Dataset**

### **Overview**:
- 479 annotations across 122 story text files.
- This comprises our manually curated dataset of stories, sourced from Project Gutenberg (Okazaki, Seneca Myths and Folktales, etc) and cultural websites (e.g. www.native-languages.org), across the Cherokee, Filipino, Cherokee, Korean, Japanese, Seneca, and Maori cultures.
- 66 characters were good, 40 evil, and 297 were neutral. Of these, 66 tags were also protagonists (though not all protagonistswere always good/evil), 35 were antagonists, and 302 were default. Importantly, almost all fauna were default and neutral with the exception of crops.

### **Annotation Process**:
- Two trained annotators dually annotated each set.
- Flora and Fauna were pre-tagged in a script in the brat software, so annotators just needed to classify the role and alignments of the flora and fauna
- Roles (protagonist, antagonist, default)
- Alignments (good, neutral, evil)
- Inter-annotator agreement was ensured through a partially automated system where I or another group member would handle any agreements the automated system flagged. We had moderate agreement according to Cohen's Kappa:  (role = .66, alignment = .58).

---

## **Models**

### **Statistical Models**:
- We tried the following models: logistic regression, multinomial and compliment naive bayes, and random forest

### **LLM Promping**:
- We used OpenAI’s GPT 3.5-turbo and set its temperature to 0 to ensure deterministic responses. We prompted the LLM with 50 tokens around the annotated spans as context, and asked it to clas- sify a certain animal’s role and alignment. Using this method, role acccuracy was 45%, alignment accuracy was 68%, and the F1 of the alignment predictions was 34% and the F1 of role predictions was 39%, performing decently above chance.
- We simply used 1-shot learning for this task, and if it had gone beyond the span of this course I would have likely tried more advanced techniques, including few-shot and implementing RAG to increase the accuracy. 

---

## **Usage**

### **Installation**
1. Clone the repository
2. For running the LLM prompting, add your api-key to the script, then run the notebook in the directory `LLM-prompting`
3. For running the statistical models, run `statistical_models.py` in the directory  `statistical models`

---

## **Directory Structure**

project/
│
├── annotation/
│   ├── adjudication.ipynb/                                 # Script to flag disagreements and get statistics
│   ├── Changelog.docx/                                     # Log of annotation guideline updates
│   ├── Folklore_Team_Annotation_Guidelines.pdf/            # Guidelines
│   ├── gold_data/            # Guidelines
│   └── 
│
├── models/
│   ├── logistic_regression/  # Model-specific files
│   ├── transformer/          # Model-specific files
│   └── results/              # Output from evaluations
│
├── notebooks/                # Jupyter notebooks for exploration
│
├── scripts/                  # Training, evaluation, and preprocessing scripts
│
├── requirements.txt          # Dependencies for the project
└── README.md                 # Project documentation







---
