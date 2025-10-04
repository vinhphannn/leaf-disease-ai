from pathlib import Path
from typing import Dict, List, Tuple


def parse_species_and_disease_from_classname(class_name: str) -> Tuple[str, str]:
    """Parse PlantVillage-style class name into (species, disease).

    Examples:
      - "Apple___Cedar_apple_rust" -> ("Apple", "Cedar_apple_rust")
      - "Tomato___healthy" -> ("Tomato", "healthy")
    """
    if "___" in class_name:
        species, disease = class_name.split("___", 1)
    else:
        # Fallback: try split by first underscore or return as-is
        parts = class_name.split("_", 1)
        species = parts[0]
        disease = parts[1] if len(parts) > 1 else "unknown"
    return species, disease


def build_species_mapping(class_names: List[str]) -> Tuple[List[str], Dict[int, int]]:
    """Return (species_list, class_index_to_species_index).

    species_list is a sorted unique list of species.
    Mapping maps original class index -> species index.
    """
    species_order: List[str] = []
    species_set = set()
    for cname in class_names:
        species, _ = parse_species_and_disease_from_classname(cname)
        if species not in species_set:
            species_set.add(species)
            species_order.append(species)
    species_order = sorted(species_order)

    class_to_species_index: Dict[int, int] = {}
    for idx, cname in enumerate(class_names):
        s, _ = parse_species_and_disease_from_classname(cname)
        class_to_species_index[idx] = species_order.index(s)
    return species_order, class_to_species_index


def build_disease_mapping_for_species(class_names: List[str], species: str) -> Tuple[List[str], Dict[int, int]]:
    """For a target species, return (disease_list, class_index_to_disease_index).

    The disease_list contains all diseases (including "healthy") for that species
    in sorted order. Any class that does not belong to the species is mapped to -1.
    """
    diseases: List[str] = []
    for cname in class_names:
        s, d = parse_species_and_disease_from_classname(cname)
        if s == species and d not in diseases:
            diseases.append(d)
    diseases = sorted(diseases)

    mapping: Dict[int, int] = {}
    for idx, cname in enumerate(class_names):
        s, d = parse_species_and_disease_from_classname(cname)
        if s != species:
            mapping[idx] = -1
        else:
            mapping[idx] = diseases.index(d)
    return diseases, mapping





