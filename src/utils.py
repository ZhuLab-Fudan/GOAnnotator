import re
from typing import List
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    """Configuration for file paths and parameters."""
    PRO_INDEX_PATH = os.path.join(BASE_DIR, "dependencies", "pro_index")
    GO_INDEX_PATH = os.path.join(BASE_DIR, "dependencies", "go_index")
    PMID_INDEX_PATH = os.path.join(BASE_DIR, "dependencies", "2024-pubmed-uniprot-index")
    PMID2TEXT_FILE = os.path.join(BASE_DIR, "dependencies", "pmid2text.npy")
    TASK_PRO2GO_FILE = os.path.join(BASE_DIR, "dependencies", "{}_pro2go.npy")
    METADATA_PATH = os.path.join(BASE_DIR, "dependencies", "protein_metadata.npy")


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

def extract_for_FIR(data):
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
    return filter_text(names)


def extract_for_train(data):
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
    gene_info = data.get('gene name', {}).get('name', [])

    if isinstance(gene_info, list) and gene_info:
        first_entry = gene_info[0]  
        ordered_locus_names = first_entry.get('Name', '')  
        if not ordered_locus_names:  
            ordered_locus_names = first_entry.get('OrderedLocusNames', [''])[0] 
    else:
        ordered_locus_names = ''
    species = data.get('species', "")
    
    # Combine recommended names
    names = ", ".join(name for name in [rec_name, short_name, alt_name] if name)
    
    # Format gene name and species
    gene_name_str = f"The Name is {ordered_locus_names}" if ordered_locus_names else ""
    species_str = f"The species is {species}" if species else ""
    
    # Combine all parts, separated by tabs
    full_info = "\t".join(part for part in [
        f"The recommended names are {names}" if names else "",
        gene_name_str,
        species_str
    ] if part)
    
    return filter_text(full_info)