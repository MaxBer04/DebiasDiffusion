import spacy
import random
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from diffusers import StableDiffusionPipeline

# Initialisierung der notwendigen Bibliotheken und Modelle
nlp = spacy.load("en_core_web_sm")
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def get_doc_from_text(text):
    return nlp(text)

def calculate_snr(pipeline: StableDiffusionPipeline, timestep: int) -> float:
    """
    Berechnet die Signal-to-Noise Ratio (SNR) für einen gegebenen Zeitschritt in der Diffusionspipeline.
    
    Args:
    pipeline (StableDiffusionPipeline): Die verwendete Diffusionspipeline.
    timestep (int): Der Zeitschritt, für den die SNR berechnet werden soll.
    
    Returns:
    float: Die berechnete SNR für den angegebenen Zeitschritt.
    """
    scheduler = pipeline.scheduler
    
    alphas = scheduler.alphas_cumprod[timestep]
    betas = scheduler.betas[timestep]
    
    snr = alphas / (1 - alphas)
    return snr



def is_human_describing_noun(word):
    """
    Überprüft, ob ein Nomen Menschen beschreibt, basierend auf dessen Hypernyms in WordNet.
    """
    synsets = wn.synsets(word, pos=wn.NOUN)
    for synset in synsets:
        for hyper in synset.closure(lambda s: s.hypernyms()):
            if 'person' in hyper.name() or 'human' in hyper.name():
                return True
    return False

def extract_nouns(text: str) -> list:
    """
    Extracts nouns from the given text and provides their textual representation, 
    index position, and token object.
    
    Args:
    text (str): The text from which to extract nouns.
    
    Returns:
    list: A list of tuples, each containing the text of the noun, its start index, 
          and the corresponding token object.
    """
    doc = nlp(text)
    nouns = [(token.text, token.idx, token, is_human_describing_noun(token.text)) for token in doc if token.pos_ == "NOUN"]
    return nouns

def replace_one_noun_with_person(text: str) -> list:
    """
    Generates a list of versions of the original text, each with a different single noun replaced by 'person'.
    
    Args:
    text (str): The original text.
    
    Returns:
    list: A list of strings, each with one noun replaced by 'person'.
    """
    nouns = extract_nouns(text)
    modified_texts = []
    
    for noun_text, noun_idx, token in nouns:
        # Start of the noun in the text
        start = noun_idx
        # End of the noun in the text
        end = noun_idx + len(noun_text)
        # Replace the noun with 'person'
        modified_text = text[:start] + "person" + text[end:]
        modified_texts.append(modified_text)
    
    return modified_texts

def extract_and_classify_nouns(text):
    """
    Extrahiert Nomen aus einem gegebenen Text und klassifiziert sie, ob sie Menschen beschreiben.
    """
    nouns = extract_nouns(text)
    human_nouns = []
    for noun, idx, token, is_human_describing_noun in nouns:
        if is_human_describing_noun:
            human_nouns.append((noun, idx, token, True))
            #print(f"'{noun}' - POSITIVE")
        else:
            human_nouns.append((noun, idx, token, False))
            #print(f"'{noun}' - NEGATIVE")
    return human_nouns

def check_existing_attribute(prompt, noun, attribute_classes):
    doc = get_doc_from_text(prompt)
    noun_token = next(token for token in doc if token.text == noun)
    modifiers = [child.text.lower() for child in noun_token.children if child.dep_ in ['amod', 'compound']]
    return any(attr.lower() in modifiers for attr in attribute_classes)

def insert_classes_before_noun(prompt, noun, noun_idx, noun_token, attributes):
    """
    Fügt spezifizierte Attribute vor einem Nomen in einem Satz ein.
    
    Args:
    - prompt: Der ursprüngliche Satz.
    - noun: Das Nomen, vor dem Attribute eingefügt werden sollen.
    - noun_idx: Der Index des Nomens im Satz.
    - noun_token: Das spacy Token des Nomens.
    - attributes: Liste der Attribute, die eingefügt werden sollen.
    
    Returns:
    - Der modifizierte Satz mit eingefügten Attributen.
    """
    doc = get_doc_from_text(prompt)
    determiners = [token.text for token in doc if token.head == noun_token and token.dep_ == 'det']
    insert_position = noun_token.idx - len(determiners[0]) - 1 if determiners else noun_token.idx
    attribute_str = ' '.join(attributes) + ' ' + (' '.join(determiners) + ' ' if determiners else '')
    return prompt[:insert_position] + attribute_str + prompt[insert_position:]


def debias_prompts(prompts, race_classes, race_chosen_idx, race_target_dist, race_dropout, gender_classes, gender_chosen_idx, gender_target_dist, gender_dropout, age_classes, age_chosen_idx, age_target_dist, age_dropout, seed, prechosen_noun=None):
    """
    Debiases the given prompts by inserting specified attributes for race, gender, and age before a randomly chosen noun from each prompt.
    
    Args:
    - prompts: List of input prompt strings.
    - race_classes, gender_classes, age_classes: Lists of possible classes for each attribute.
    - race_target_dist, gender_target_dist, age_target_dist: Lists of probabilities corresponding to each class.
    - race_dropout, gender_dropout, age_dropout: Probability of not choosing an attribute.
    - seed: Seed for deterministic behavior.
    
    Returns:
    - List of debiased prompts with attributes inserted.
    """
    random.seed(seed)
    np.random.seed(seed)
    debiased_prompts = []
    for prompt in prompts:
        results = extract_and_classify_nouns(prompt)
        results = [r for r in results if r[3]]
        prechosen_noun = None
        if results:
            if prechosen_noun:
                chosen_noun, idx, token, is_human = prechosen_noun
            else:
                chosen_noun, idx, token, is_human = random.choice(results)
            attributes = []
            #print(f"Noun to be debiased: {chosen_noun}")
            if random.random() > race_dropout:
                if race_chosen_idx:
                    race = race_classes[race_chosen_idx]
                else:
                    race = np.random.choice(race_classes, p=race_target_dist)
                attributes.append(race)
            if random.random() > gender_dropout:
                if gender_chosen_idx:
                    gender = gender_classes[gender_chosen_idx]
                else:
                    gender = np.random.choice(gender_classes, p=gender_target_dist)
                attributes.append(gender)
            if random.random() > age_dropout:
                if age_chosen_idx:
                    age = age_classes[age_chosen_idx]
                else:
                    age = np.random.choice(age_classes, p=age_target_dist)
                attributes.append(age)
            debiased_prompt = insert_classes_before_noun(prompt, chosen_noun, idx, token, attributes)
        else:
            debiased_prompt = prompt
        debiased_prompts.append(debiased_prompt)
    return debiased_prompts


def debias_prompts_defaults(prompts, seed, prechosen_noun=None, race_chosen_idx=None, gender_chosen_idx=None, age_chosen_idx=None):
    return debias_prompts(
        prompts,
        race_classes=["white", "black", "asian", "indian"],
        race_chosen_idx=race_chosen_idx,
        race_target_dist=[0.25, 0.25, 0.25, 0.25],
        race_dropout=0,
        gender_classes=["female", "male"],
        gender_chosen_idx=gender_chosen_idx,
        gender_target_dist=[0.5, 0.5],
        gender_dropout=0,
        age_classes=["young", "old"],
        age_chosen_idx=age_chosen_idx,
        age_target_dist=[0.5, 0.5],
        age_dropout=0,
        seed=seed,
        prechosen_noun=prechosen_noun
    )
