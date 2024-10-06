"""
Utility functions for fairness-related operations in the DebiasDiffusion project.

This module provides functions for analyzing and manipulating text prompts
to support fairness in text-to-image diffusion models. It includes tools for
noun extraction, attribute classification, and prompt modification.

Usage:
    from src.utils.fairness import extract_and_classify_nouns, debias_prompts_defaults

    nouns = extract_and_classify_nouns("A photo of a doctor")
    debiased_prompts = debias_prompts_defaults(["A photo of a doctor"], seed=42)
"""

import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import spacy
import nltk
from nltk.corpus import wordnet as wn
from diffusers import StableDiffusionPipeline

# Initialize necessary libraries and models
nlp = spacy.load("en_core_web_sm")
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def get_doc_from_text(text: str) -> spacy.tokens.Doc:
    """
    Create a spaCy Doc object from the input text.

    Args:
        text (str): The input text to process.

    Returns:
        spacy.tokens.Doc: The processed spaCy Doc object.
    """
    return nlp(text)

def calculate_snr(pipeline: StableDiffusionPipeline, timestep: int) -> float:
    """
    Calculate the Signal-to-Noise Ratio (SNR) for a given timestep in the diffusion pipeline.

    Args:
        pipeline (StableDiffusionPipeline): The diffusion pipeline.
        timestep (int): The timestep for which to calculate the SNR.

    Returns:
        float: The calculated SNR for the specified timestep.
    """
    scheduler = pipeline.scheduler
    alphas = scheduler.alphas_cumprod[timestep]
    snr = alphas / (1 - alphas)
    return snr

def is_human_describing_noun(word: str) -> bool:
    """
    Check if a noun describes humans based on its hypernyms in WordNet.

    Args:
        word (str): The word to check.

    Returns:
        bool: True if the word describes humans, False otherwise.
    """
    synsets = wn.synsets(word, pos=wn.NOUN)
    for synset in synsets:
        for hyper in synset.closure(lambda s: s.hypernyms()):
            if 'person' in hyper.name() or 'human' in hyper.name():
                return True
    return False

def extract_nouns(text: str) -> List[Tuple[str, int, spacy.tokens.Token, bool]]:
    """
    Extract nouns from the given text and provide their textual representation,
    index position, token object, and whether they describe humans.

    Args:
        text (str): The text from which to extract nouns.

    Returns:
        List[Tuple[str, int, spacy.tokens.Token, bool]]: A list of tuples, each containing
        the text of the noun, its start index, the corresponding token object, and
        a boolean indicating if it describes humans.
    """
    doc = nlp(text)
    nouns = [(token.text, token.idx, token, is_human_describing_noun(token.text)) 
             for token in doc if token.pos_ == "NOUN"]
    return nouns

def replace_one_noun_with_person(text: str) -> List[str]:
    """
    Generate a list of versions of the original text, each with a different single noun replaced by 'person'.

    Args:
        text (str): The original text.

    Returns:
        List[str]: A list of strings, each with one noun replaced by 'person'.
    """
    nouns = extract_nouns(text)
    modified_texts = []
    
    for noun_text, noun_idx, _, _ in nouns:
        start = noun_idx
        end = noun_idx + len(noun_text)
        modified_text = text[:start] + "person" + text[end:]
        modified_texts.append(modified_text)
    
    return modified_texts

def extract_and_classify_nouns(text: str) -> List[Tuple[str, int, spacy.tokens.Token, bool]]:
    """
    Extract nouns from a given text and classify them as human-describing or not.

    Args:
        text (str): The input text to analyze.

    Returns:
        List[Tuple[str, int, spacy.tokens.Token, bool]]: A list of tuples containing
        the noun, its index, the spaCy token, and a boolean indicating if it's human-describing.
    """
    nouns = extract_nouns(text)
    human_nouns = []
    for noun, idx, token, is_human_describing in nouns:
        human_nouns.append((noun, idx, token, is_human_describing))
    return human_nouns

def check_existing_attribute(prompt: str, noun: str, attribute_classes: List[str]) -> bool:
    """
    Check if any of the attribute classes already exist as modifiers for the given noun in the prompt.

    Args:
        prompt (str): The input prompt.
        noun (str): The noun to check for modifiers.
        attribute_classes (List[str]): List of attribute classes to check.

    Returns:
        bool: True if any attribute class exists as a modifier, False otherwise.
    """
    doc = get_doc_from_text(prompt)
    noun_token = next(token for token in doc if token.text == noun)
    modifiers = [child.text.lower() for child in noun_token.children if child.dep_ in ['amod', 'compound']]
    return any(attr.lower() in modifiers for attr in attribute_classes)

def insert_classes_before_noun(prompt: str, noun: str, noun_idx: int, noun_token: spacy.tokens.Token, attributes: List[str]) -> str:
    """
    Insert specified attributes before a noun in a sentence.

    Args:
        prompt (str): The original sentence.
        noun (str): The noun before which attributes will be inserted.
        noun_idx (int): The index of the noun in the sentence.
        noun_token (spacy.tokens.Token): The spacy Token of the noun.
        attributes (List[str]): List of attributes to be inserted.

    Returns:
        str: The modified sentence with attributes inserted.
    """
    doc = get_doc_from_text(prompt)
    determiners = [token.text for token in doc if token.head == noun_token and token.dep_ == 'det']
    insert_position = noun_token.idx - len(determiners[0]) - 1 if determiners else noun_token.idx
    attribute_str = ' '.join(attributes) + ' ' + (' '.join(determiners) + ' ' if determiners else '')
    return prompt[:insert_position] + attribute_str + prompt[insert_position:]

def debias_prompts(prompts: List[str], 
                   race_classes: List[str], 
                   race_chosen_idx: Optional[int], 
                   race_target_dist: List[float], 
                   race_dropout: float, 
                   gender_classes: List[str], 
                   gender_chosen_idx: Optional[int], 
                   gender_target_dist: List[float], 
                   gender_dropout: float, 
                   age_classes: List[str], 
                   age_chosen_idx: Optional[int], 
                   age_target_dist: List[float], 
                   age_dropout: float, 
                   seed: int, 
                   prechosen_noun: Optional[Tuple[str, int, spacy.tokens.Token, bool]] = None) -> List[str]:
    """
    Debias the given prompts by inserting specified attributes for race, gender, and age.

    Args:
        prompts (List[str]): List of input prompt strings.
        race_classes (List[str]): List of possible race classes.
        race_chosen_idx (Optional[int]): Index of chosen race class, if any.
        race_target_dist (List[float]): Target distribution for race classes.
        race_dropout (float): Probability of not choosing a race attribute.
        gender_classes (List[str]): List of possible gender classes.
        gender_chosen_idx (Optional[int]): Index of chosen gender class, if any.
        gender_target_dist (List[float]): Target distribution for gender classes.
        gender_dropout (float): Probability of not choosing a gender attribute.
        age_classes (List[str]): List of possible age classes.
        age_chosen_idx (Optional[int]): Index of chosen age class, if any.
        age_target_dist (List[float]): Target distribution for age classes.
        age_dropout (float): Probability of not choosing an age attribute.
        seed (int): Random seed for reproducibility.
        prechosen_noun (Optional[Tuple[str, int, spacy.tokens.Token, bool]]): Pre-chosen noun to debias, if any.

    Returns:
        List[str]: List of debiased prompts.
    """
    random.seed(seed)
    np.random.seed(seed)
    debiased_prompts = []
    for prompt in prompts:
        results = extract_and_classify_nouns(prompt)
        results = [r for r in results if r[3]]
        if results:
            if prechosen_noun:
                chosen_noun, idx, token, is_human = prechosen_noun
            else:
                chosen_noun, idx, token, is_human = random.choice(results)
            attributes = []
            if random.random() > race_dropout:
                if race_chosen_idx is not None:
                    race = race_classes[race_chosen_idx]
                else:
                    race = np.random.choice(race_classes, p=race_target_dist)
                attributes.append(race)
            if random.random() > gender_dropout:
                if gender_chosen_idx is not None:
                    gender = gender_classes[gender_chosen_idx]
                else:
                    gender = np.random.choice(gender_classes, p=gender_target_dist)
                attributes.append(gender)
            if random.random() > age_dropout:
                if age_chosen_idx is not None:
                    age = age_classes[age_chosen_idx]
                else:
                    age = np.random.choice(age_classes, p=age_target_dist)
                attributes.append(age)
            debiased_prompt = insert_classes_before_noun(prompt, chosen_noun, idx, token, attributes)
        else:
            debiased_prompt = prompt
        debiased_prompts.append(debiased_prompt)
    return debiased_prompts

def debias_prompts_defaults(prompts: List[str], 
                            seed: int, 
                            prechosen_noun: Optional[Tuple[str, int, spacy.tokens.Token, bool]] = None,
                            race_chosen_idx: Optional[int] = None, 
                            gender_chosen_idx: Optional[int] = None, 
                            age_chosen_idx: Optional[int] = None) -> List[str]:
    """
    Debias prompts using default settings for attributes and distributions.

    Args:
        prompts (List[str]): List of input prompt strings.
        seed (int): Random seed for reproducibility.
        prechosen_noun (Optional[Tuple[str, int, spacy.tokens.Token, bool]]): Pre-chosen noun to debias, if any.
        race_chosen_idx (Optional[int]): Index of chosen race class, if any.
        gender_chosen_idx (Optional[int]): Index of chosen gender class, if any.
        age_chosen_idx (Optional[int]): Index of chosen age class, if any.

    Returns:
        List[str]: List of debiased prompts.
    """
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