# Task definitions
TASK_DEFINITIONS = {
    'cc': 'Cellular Component',
    'bp': 'Biological Process',
    'mf': 'Molecular Function',
}

import re
from typing import List

# Task definitions
TASK_DEFINITIONS = {
    'cc': 'Cellular Component',
    'bp': 'Biological Process',
    'mf': 'Molecular Function',
}

def filter_text(text: str) -> str:
    """
    Remove unwanted patterns enclosed in curly braces.
    Args:
        text (str): Input text to be filtered.
    Returns:
        str: Filtered text.
    """
    return re.sub(r"{.*?}", "", text).strip()

def filter_species(text: str) -> str:
    """
    Extract species name from text enclosed in parentheses.
    Args:
        text (str): Input text containing species information.
    Returns:
        str: Extracted species name or original text if no parentheses found.
    """
    match = re.search(r"\((.*?)\)", text)
    return match.group(1) if match else text

def average_scores(scores: List[float]) -> float:
    """
    Calculate the average of a list of scores.
    Args:
        scores (List[float]): A list of numerical scores.
    Returns:
        float: Average score or 0.0 if the list is empty.
    """
    return sum(scores) / len(scores) if scores else 0.0

def extract_for_train(data):
    """
    Extracts the recommended names from the input dictionary and combines them into a single string.
    
    Args:
        data (dict): The dictionary containing protein information.
        
    Returns:
        str: A combined string of recommended names (RecName, short name, and AltName). 
             Missing values are excluded from the result.
    """
    rec_name = data.get('full name', {}).get('RecName', "")
    short_name = data.get('short name', "")
    alt_name = data.get('full name', {}).get('AltName', "")
    
    # Filter out empty values and join them with a comma
    names = ", ".join(name for name in [rec_name, short_name, alt_name] if name)
    return names


def extract_for_FIR(data):
    """
    Extracts full protein information and formats it into a detailed string.
    
    Args:
        data (dict): The dictionary containing protein information.
        
    Returns:
        str: A detailed string including recommended names, OrderedLocusNames, and species. 
             Missing values are excluded from the result.
    """
    rec_name = data.get('full name', {}).get('RecName', "")
    short_name = data.get('short name', "")
    alt_name = data.get('full name', {}).get('AltName', "")
    ordered_locus_names = data.get('gene name', {}).get('name', [{}])[0].get('OrderedLocusNames', [])
    species = data.get('species', "")
    
    # Combine recommended names
    names = ", ".join(name for name in [rec_name, short_name, alt_name] if name)
    
    # Format OrderedLocusNames and species
    ordered_locus_names_str = f"The OrderedLocusNames is {ordered_locus_names}" if ordered_locus_names else ""
    species_str = f"The species is {species}" if species else ""
    
    # Combine all parts, separated by tabs
    full_info = "\t".join(part for part in [
        f"The recommended names are {names}" if names else "",
        ordered_locus_names_str,
        species_str
    ] if part)
    
    return full_info