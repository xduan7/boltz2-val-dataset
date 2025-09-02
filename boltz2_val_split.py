#!/usr/bin/env python3
"""
Boltz-2 Validation Dataset Construction Pipeline

Implements the validation dataset construction methodology from:
"Predicting Biomolecular Interactions with Boltz-2"
Section A.1.5: Validation and Test Set Construction
"""

import os
import json
import datetime
import random
import pickle
import argparse
import time
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Ligand exclusion list - common crystallization agents, buffers, solvents
ligand_exclusion = {
    "144", "15P", "1PE", "2F2", "2JC", "3HR", "3SY", "7N5", "7PE", "9JE",
    "AAE", "ABA", "ACE", "ACN", "ACT", "ACY", "AZI", "BAM", "BCN", "BCT",
    "BDN", "BEN", "BME", "BO3", "BTB", "BTC", "BU1", "C8E", "CAD", "CAQ",
    "CBM", "CCN", "CIT", "CL", "CLR", "CM", "CMO", "CO3", "CPT", "CXS",
    "D10", "DEP", "DIO", "DMS", "DN", "DOD", "DOX", "EDO", "EEE", "EGL",
    "EOH", "EOX", "EPE", "ETF", "FCY", "FJO", "FLC", "FMT", "FW5", "GOL",
    "GSH", "GTT", "GYF", "HED", "IHP", "IHS", "IMD", "IOD", "IPA", "IPH",
    "LDA", "MB3", "MEG", "MES", "MLA", "MLI", "MOH", "MPD", "MRD", "MSE",
    "MYR", "N", "NA", "NH2", "NH4", "NHE", "NO3", "O4B", "OHE", "OLA",
    "OLC", "OMB", "OME", "OXA", "P6G", "PE3", "PE4", "PEG", "PEO", "PEP",
    "PG0", "PG4", "PGE", "PGR", "PLM", "PO4", "POL", "POP", "PVO", "SAR",
    "SCN", "SEO", "SEP", "SIN", "SO4", "SPD", "SPM", "SR", "STE", "STO",
    "STU", "TAR", "TBU", "TME", "TPO", "TRS", "UNK", "UNL", "UNX", "UPL",
    "URE"
}

# Paper parameters
DATE_START = datetime.date(2023, 6, 1)
DATE_END = datetime.date(2024, 1, 1)
MAX_RESOLUTION = 4.5
MIN_SEQ_ID = 0.4
MAX_RESIDUES = 1024
MAX_ENTITIES = 20
MULTIMER_KEEP_RATE = 0.9
MONOMER_KEEP_RATE = 0.6
DEFAULT_RANDOM_SEED = 42

# Data paths
MANIFEST = "/homes/duan/data/boltz_2/processed/manifest.json"
CLUSTERS = "/homes/duan/data/boltz_2/processed/clustering/clust_prot_cluster.tsv"
DATA_DIR = "/homes/duan/data/boltz_2/processed/records"
CCD_PATH = "/homes/duan/data/boltz_2/raw/ccd.pkl"
PDB_DIR = "/homes/duan/data/boltz_2/raw/mmcif"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def parse_cif_nonpoly(cif_file, exclude_set=None):
    """Extract non-polymer CCD codes from a CIF file."""
    ccd_codes = set()
    if not os.path.exists(cif_file):
        return ccd_codes
    
    try:
        cif = MMCIF2Dict(cif_file)
        if '_pdbx_entity_nonpoly.comp_id' in cif:
            non_poly = cif['_pdbx_entity_nonpoly.comp_id']
            if isinstance(non_poly, str):
                non_poly = [non_poly]
            
            for comp_id in non_poly:
                if exclude_set is None or comp_id not in exclude_set:
                    ccd_codes.add(comp_id)
    except:
        pass
    
    return ccd_codes

def extract_ccd_codes(record):
    """
    Extract CCD codes from a structure (for multiprocessing).
    
    Paper quote (Section A.1.5, point 4):
    "small-molecule is defined as any non-polymer entity containing more than one 
    heavy atom and not included in the ligand exclusion list"
    """
    pdb_id = record['id']
    cif_file = os.path.join(PDB_DIR, f"{pdb_id}.cif")
    return parse_cif_nonpoly(cif_file, ligand_exclusion)

def check_date_range(record, start_date, end_date):
    """
    Check if structure release date is within specified range.
    
    Based on official Boltz implementation at:
    boltz/src/boltz/data/filter/dynamic/date.py (lines 59-76)
    
    The official DateFilter uses fallback logic:
    - For 'released' ref: tries released date, falls back to deposited if not available
    - Returns False only if no date is available at all
    """
    try:
        # First try to get the released date
        date_str = record['structure'].get('released')
        
        # Fallback to deposited date if released is not available
        # This matches the official implementation's fallback logic (lines 62-64)
        if not date_str:
            date_str = record['structure'].get('deposited')
        
        # If still no date, return False (matches line 73)
        if not date_str:
            return False
            
        release_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        return start_date < release_date <= end_date
    except (KeyError, ValueError):
        return False

def check_resolution(record, max_resolution):
    """Check if structure resolution is below threshold."""
    try:
        resolution = record['structure']['resolution']
        return resolution <= max_resolution
    except (KeyError, TypeError):
        return False

def count_residues(record):
    """
    Count total residues across ALL chains.
    
    Based on official Boltz implementation at:
    boltz/src/boltz/data/filter/dynamic/max_residues.py (line 36)
    
    IMPORTANT: The official implementation counts residues from ALL chains,
    not just protein/DNA/RNA (mol_type 0,1,2). This includes non-polymer 
    chains (mol_type 3) as well.
    
    Original code: sum(chain.num_residues for chain in record.chains)
    """
    return sum(
        chain['num_residues'] 
        for chain in record['chains']
        # No mol_type filtering - count ALL chains as per official implementation
    )

def count_entities(record, exclusion_tracker, mode):
    """
    Count unique entities in a structure.
    
    Paper quote (Section A.1.5):
    "Exclude complexes with more than 20 entities"
    
    Note: Entities are unique molecular components, not chains.
    A homodimer has 2 chains but only 1 entity.
    
    Args:
        record: Structure record
        exclusion_tracker: Dictionary to track exclusion reasons
        mode: 'all' - count all entities, 'polymeric-only' - count only polymeric entities,
              'chains' - count all chains
    
    Returns:
        int: Number of entities if determinable
        None: If entity count cannot be determined (structure should be excluded)
    """
    pdb_id = record['id']
    cif_file = os.path.join(PDB_DIR, f"{pdb_id}.cif")
    chain_count = len(record['chains'])
    
    # If mode is 'chains', simply return the chain count
    if mode == 'chains':
        return chain_count
    
    if not os.path.exists(cif_file):
        # Cannot determine entity count - mark for exclusion
        if exclusion_tracker is not None:
            exclusion_tracker['no_cif'].append((pdb_id, chain_count, 'CIF file not found'))
        return None
    
    try:
        cif = MMCIF2Dict(cif_file)
        if '_entity.id' in cif:
            entities = cif['_entity.id']
            entity_count = len(entities) if isinstance(entities, list) else 1
            
            # If mode is 'polymeric-only', filter to only polymeric entities
            if mode == 'polymeric-only' and '_entity.type' in cif:
                entity_types = cif['_entity.type']
                if not isinstance(entity_types, list):
                    entity_types = [entity_types]
                # Count only polymer entities
                entity_count = sum(1 for etype in entity_types if etype in ['polymer', 'branched'])
            
            # Track cases where entity count differs significantly from chain count
            if exclusion_tracker is not None and entity_count != chain_count:
                ratio = chain_count / entity_count if entity_count > 0 else 0
                if ratio >= 2:  # Track homo-oligomers with 2+ chains per entity
                    exclusion_tracker['oligomers_detected'].append((pdb_id, entity_count, chain_count))
            
            return entity_count
        else:
            # Cannot determine entity count - mark for exclusion
            if exclusion_tracker is not None:
                exclusion_tracker['no_entity_data'].append((pdb_id, chain_count, 'No _entity.id field in CIF'))
            return None
    except Exception as e:
        # Cannot determine entity count - mark for exclusion
        if exclusion_tracker is not None:
            exclusion_tracker['parse_error'].append((pdb_id, chain_count, str(e)[:100]))
        return None

def has_small_molecules(record):
    """Check if structure contains small molecules."""
    return any(chain['mol_type'] == 3 for chain in record['chains'])

def has_nucleic_acids(record):
    """Check if structure contains RNA or DNA."""
    return any(chain['mol_type'] in [1, 2] for chain in record['chains'])

def count_protein_chains(record):
    """Count number of protein chains."""
    return sum(1 for chain in record['chains'] if chain['mol_type'] == 0)

# ==============================================================================
# MOLECULAR PROCESSING
# ==============================================================================

def satisfies_lipinski_rule(mol):
    """
    Check Lipinski's Rule of Five.
    
    Paper quote (Section A.1.5, point 4):
    "The small-molecule satisfies Lipinski's Rule of Five"
    """
    if mol is None:
        return False
    
    try:
        return (
            Descriptors.MolWt(mol) <= 500 and
            Descriptors.MolLogP(mol) <= 5 and
            Lipinski.NumHDonors(mol) <= 5 and
            Lipinski.NumHAcceptors(mol) <= 10
        )
    except:
        return False

def structure_passes_lipinski(mols, mode='any'):
    """
    Check if structure's molecules pass Lipinski's Rule of Five.
    
    Paper quote (Section A.1.5, point 4):
    "The small-molecule satisfies Lipinski's Rule of Five"
    
    Args:
        mols: Dictionary of CCD code to RDKit molecule
        mode: 'any' - at least one molecule must pass (default)
              'all' - all molecules must pass
    """
    if not mols:
        return False
    
    passing = [satisfies_lipinski_rule(mol) for _, mol in mols.items()]
    return any(passing) if mode == 'any' else all(passing)

def generate_morgan_fingerprints(mols_dict, fpSize=2048):
    """
    Generate Morgan fingerprints for molecular similarity.
    
    Paper context: Used for Tanimoto similarity calculation.
    "At least one of the small-molecules exhibits a Tanimoto similarity of 0.85 or less"
    """
    fp_gen = GetMorganGenerator(radius=2, fpSize=fpSize)
    fps = {}
    
    for ccd_code, mol in mols_dict.items():
        if mol is not None:
            try:
                # Ensure ring info is initialized safely
                if not mol.GetRingInfo().IsAtomRingResultsInitialized():
                    mol.GetRingInfo().AtomRings()
                fps[ccd_code] = fp_gen.GetFingerprint(mol)
            except:
                # Skip molecules that cause issues
                continue
    
    return fps

def calculate_tanimoto_similarities(target_fps, training_fps):
    """
    Calculate maximum Tanimoto similarity.
    
    Paper quote (Section A.1.5, point 4):
    "At least one of the small-molecules exhibits a Tanimoto similarity of 0.85 or less
    to any small-molecule in the training set"
    """
    similarities = {}
    
    for target_ccd, target_fp in target_fps.items():
        max_sim = 0.0
        for train_ccd, train_fp in training_fps.items():
            sim = DataStructs.TanimotoSimilarity(target_fp, train_fp)
            max_sim = max(max_sim, sim)
        similarities[target_ccd] = max_sim
    
    return similarities

# ==============================================================================
# STRUCTURE ANALYSIS
# ==============================================================================

def get_structure_clusters(record, cluster_map, unmapped_chain_policy, mapping_mode):
    """
    Get protein cluster IDs for a structure.
    
    Paper quote (Section A.1.5, point 3):
    "All the protein sequences of the chains are not present in any training set clusters"
    
    Args:
        record: Structure record
        cluster_map: Mapping of sequence hashes to cluster IDs
        unmapped_chain_policy: 'reject' to exclude structures with unmapped chains,
                              'ignore' to skip unmapped chains
        mapping_mode: How to get the key to query cluster_map ('sequence_hash' or 'pdb_chain')
    
    Returns:
        set: Cluster IDs if chains are successfully processed
        None: If any protein chain cannot be mapped (when policy='reject')
    
    Note: Returning None ensures strict compliance with the paper's requirement that
    ALL protein chains must belong to known clusters. Unmapped chains indicate
    incomplete data and such structures should be excluded from iterative selection.
    """
    clusters = set()
    has_protein_chains = False
    pdb_id = record['id']
    
    for chain in record['chains']:
        if chain['mol_type'] == 0:  # Protein
            has_protein_chains = True
            
            if mapping_mode == 'manifest':
                # Use the cluster_id already present in the manifest
                cluster_id = chain.get('cluster_id')
                if cluster_id:
                    clusters.add(cluster_id)
                elif unmapped_chain_policy == 'reject':
                    return None
            elif mapping_mode == 'pdb_chain':
                # Use PDB_ID + chain name to look up in external cluster file
                chain_name = chain.get('chain_name')
                if chain_name:
                    key = f"{pdb_id}_{chain_name}"
                    if key in cluster_map:
                        clusters.add(cluster_map[key])
                    elif unmapped_chain_policy == 'reject':
                        return None
                elif unmapped_chain_policy == 'reject':
                    return None
            elif mapping_mode == 'sequence_hash':
                # Use sequence_hash to look up (note: this won't work with current data)
                seq_hash = chain.get('sequence_hash')
                if seq_hash and seq_hash in cluster_map:
                    clusters.add(cluster_map[seq_hash])
                elif unmapped_chain_policy == 'reject':
                    return None
            else:
                raise ValueError(f"Unknown mapping mode: {mapping_mode}")
    
    # If no protein chains exist, return empty set (vacuously satisfies the condition)
    # If all protein chains mapped successfully (or ignored), return their cluster IDs
    return clusters

def get_biologically_relevant_molecules(record, valid_ccd, include_ions):
    """
    Extract biologically relevant small molecules.
    
    Paper quotes (Section A.1.5):
    "small-molecule is defined as any non-polymer entity containing more than one
    heavy atom and not included in the ligand exclusion list"
    """
    molecules = {}
    pdb_id = record['id']
    cif_file = os.path.join(PDB_DIR, f"{pdb_id}.cif")
    
    ccd_codes = parse_cif_nonpoly(cif_file, ligand_exclusion)
    
    for comp_id in ccd_codes:
        if isinstance(valid_ccd, dict) and comp_id in valid_ccd:
            mol = valid_ccd[comp_id]
            if mol is not None:
                # Paper definition: "containing more than one heavy atom"
                try:
                    if mol.GetNumHeavyAtoms() > 1:
                        molecules[comp_id] = mol
                    elif include_ions and mol.GetNumHeavyAtoms() == 1:
                        # Single heavy atom molecules are ions
                        molecules[comp_id] = mol
                except:
                    # If we can't check, include it to be safe
                    molecules[comp_id] = mol
    
    return molecules

def categorize_structure(record, valid_ccd, include_ions):
    """
    Categorize structure by molecular content.
    
    Paper quote (Section A.1.5, iterative selection):
    "3. Retaining all the structures containing RNA or DNA entities
    4. Iteratively adding structures containing small-molecules or ions...
    5. Iteratively adding multimeric structures...
    6. Iteratively adding monomers..."
    """
    if has_nucleic_acids(record):
        return "RNA/DNA structures"
    
    if valid_ccd:
        mols = get_biologically_relevant_molecules(record, valid_ccd, include_ions)
        if mols:
            return "Small molecule structures"
    elif has_small_molecules(record):
        return "Small molecule structures"
    
    protein_chains = count_protein_chains(record)
    if protein_chains > 1:
        return "Multimer structures"
    elif protein_chains == 1:
        return "Monomer structures"
    else:
        return "Other structures"

# ==============================================================================
# ITERATIVE SELECTION
# ==============================================================================

def run_iterative_selection_static(validation_candidates, rna_dna_structures, cluster_map, valid_ccd,
                                   unmapped_chain_policy, include_ions, multimer_rate, monomer_rate, mapping_mode='sequence_hash'):
    """
    Static iterative selection (deterministic).
    
    Paper quote (Section A.1.5):
    "Iteratively adding structures... under the condition that all their protein chains
    belong to new unseen clusters"
    """
    validation_set = list(rna_dna_structures)
    seen_clusters = set()
    
    # Initialize seen_clusters with clusters from RNA/DNA structures
    for record in rna_dna_structures:
        clusters = get_structure_clusters(record, cluster_map, unmapped_chain_policy, mapping_mode)
        if clusters is not None:
            seen_clusters.update(clusters)
    
    categories = defaultdict(list)
    for record in validation_candidates:
        category = categorize_structure(record, valid_ccd, include_ions)
        categories[category].append(record)
    
    for category_name, keep_rate in [
        ("Small molecule structures", 1.0),
        ("Multimer structures", multimer_rate),
        ("Monomer structures", monomer_rate)
    ]:
        structures = categories[category_name]
        print(f"\nProcessing {category_name}: {len(structures)} candidates")
        
        added = 0
        for i, record in enumerate(structures):
            clusters = get_structure_clusters(record, cluster_map, unmapped_chain_policy, mapping_mode)
            
            # Only consider structures where ALL protein chains are mapped
            # clusters is None: at least one protein chain is unmapped (exclude)
            # clusters is empty set: no protein chains (include if novel)
            # clusters is non-empty set: has protein chains, all mapped (check novelty)
            if clusters is not None and not clusters.intersection(seen_clusters):
                if i / len(structures) < keep_rate:
                    validation_set.append(record)
                    seen_clusters.update(clusters)
                    added += 1
        
        print(f"  Added {added} structures")
    
    return validation_set

def run_iterative_selection_paper(validation_candidates, rna_dna_structures, cluster_map, valid_ccd,
                                  unmapped_chain_policy, include_ions, multimer_rate, monomer_rate, mapping_mode='sequence_hash'):
    """
    Paper-style iterative selection with random sampling.
    
    Paper quotes (Section A.1.5):
    "5. ...randomly keeping only 90% of the passing structures.
    6. ...randomly filtered out by keeping only 60% of the passing structures."
    """
    validation_set = list(rna_dna_structures)
    seen_clusters = set()
    
    for record in rna_dna_structures:
        clusters = get_structure_clusters(record, cluster_map, unmapped_chain_policy, mapping_mode)
        if clusters is not None:
            seen_clusters.update(clusters)
    
    print(f"Initial clusters from RNA/DNA structures: {len(seen_clusters)}")
    
    categories = defaultdict(list)
    for record in validation_candidates:
        category = categorize_structure(record, valid_ccd, include_ions)
        categories[category].append(record)
    
    for category_name, keep_rate in [
        ("Small molecule structures", 1.0),
        ("Multimer structures", multimer_rate),
        ("Monomer structures", monomer_rate)
    ]:
        structures = categories[category_name]
        print(f"\n{category_name[:-11]}: {len(structures)} candidates")
        
        random.shuffle(structures)
        
        added = 0
        candidates_checked = 0
        
        for record in structures:
            clusters = get_structure_clusters(record, cluster_map, unmapped_chain_policy, mapping_mode)
            
            # Only consider structures where ALL protein chains are mapped
            # clusters is None: at least one protein chain is unmapped (exclude)
            # clusters is empty set: no protein chains (include if novel)
            # clusters is non-empty set: has protein chains, all mapped (check novelty)
            if clusters is not None and not clusters.intersection(seen_clusters):
                candidates_checked += 1
                
                if random.random() < keep_rate:
                    validation_set.append(record)
                    seen_clusters.update(clusters)  # Only update AFTER selection
                    added += 1
        
        print(f"  Novel cluster candidates: {candidates_checked}")
        print(f"  Added after {int(keep_rate*100)}% sampling: {added}")
    
    categories_final = defaultdict(int)
    for record in validation_set:
        category = categorize_structure(record, valid_ccd, include_ions)
        categories_final[category] += 1
    
    print("\nFinal validation set breakdown:")
    for category, count in sorted(categories_final.items()):
        print(f"  {category}: {count}")
    
    return validation_set

# ==============================================================================
# PARALLEL WORKER FUNCTIONS (must be defined at module level for pickling)
# ==============================================================================

def check_novelty_worker(record, cluster_map, training_clusters, unmapped_chain_policy, mapping_mode):
    """Check if a structure has novel protein sequences (for parallel processing)."""
    clusters = get_structure_clusters(record, cluster_map, unmapped_chain_policy, mapping_mode)
    
    result = {
        'record': record,
        'novel': False,
        'unmapped': False
    }
    
    if clusters is None:
        result['unmapped'] = True
    elif not clusters.intersection(training_clusters):
        result['novel'] = True
        
    return result

def check_size_constraints_worker(record, entity_mode, max_residues, max_entities):
    """Check size constraints for a single record (for parallel processing)."""
    residues = count_residues(record)
    entities = count_entities(record, None, entity_mode)  # No tracker in parallel
    
    result = {
        'record': record,
        'residues': residues,
        'entities': entities,
        'passed': False,
        'exclusion_reason': None
    }
    
    if residues > max_residues:
        result['exclusion_reason'] = 'residue_limit'
    elif entities is None:
        result['exclusion_reason'] = 'entity_unknown'
    elif entities > max_entities:
        result['exclusion_reason'] = 'entity_limit'
    else:
        result['passed'] = True
        
    return result

def check_small_molecules_worker(record, ccd_data, include_ions, lipinski_mode, training_fps):
    """Check small molecule criteria for a single record (for parallel processing)."""
    # Import needed for parallel workers
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    from rdkit import DataStructs
    
    structure_mols = get_biologically_relevant_molecules(record, ccd_data, include_ions)
    
    result = {
        'record': record,
        'passed': False,
        'reason': None
    }
    
    if not structure_mols:
        result['passed'] = True
        result['reason'] = 'no_molecules'
        return result
    
    if structure_passes_lipinski(structure_mols, mode=lipinski_mode):
        result['passed'] = True
        result['reason'] = 'lipinski'
        return result
    
    # Generate fingerprints with error handling
    structure_fps = {}
    fp_gen = GetMorganGenerator(radius=2, fpSize=2048)
    for ccd_code, mol in structure_mols.items():
        if mol is not None:
            try:
                # Ensure ring info is initialized in worker process
                if not mol.GetRingInfo().IsAtomRingResultsInitialized():
                    mol.GetRingInfo().AtomRings()
                structure_fps[ccd_code] = fp_gen.GetFingerprint(mol)
            except:
                continue
    
    # Optimized Tanimoto check - early exit when we find one <= 0.85
    if structure_fps and training_fps:
        for target_ccd, target_fp in structure_fps.items():
            max_sim = 0.0
            for train_ccd, train_fp in training_fps.items():
                sim = DataStructs.TanimotoSimilarity(target_fp, train_fp)
                max_sim = max(max_sim, sim)
                # Early exit optimization - no need to check all if we already exceed threshold
                if max_sim > 0.85:
                    break
            
            # If ANY molecule has similarity <= 0.85, the structure passes
            if max_sim <= 0.85:
                result['passed'] = True
                result['reason'] = 'tanimoto'
                break
    
    return result

# ==============================================================================
# PRELOADED DATA CONTAINER
# ==============================================================================

class PreloadedData:
    """Container for preloaded data to avoid repeated disk access."""
    def __init__(self):
        self.manifest = None
        self.cluster_map = None
        self.ccd_data = None
        self.training_fingerprints = None
        self.training_molecules = None
        self.training_clusters = None
        self.training_structures = None
        self.loaded = False

# Global data container
PRELOADED = PreloadedData()

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    """
    Main pipeline implementing Boltz-2 validation split.
    
    Paper quote (Section A.1.5):
    "Our training, validation and test splitting strategy largely follows Boltz-1 procedure.
    We first cluster the protein sequences in PDB by sequence identity with the command
    `mmseqs easy-cluster... min-seq-id 0.4`. Then, we select all structures in PDB
    satisfying the following filters:"
    """
    parser = argparse.ArgumentParser(
        description='Boltz-2 validation split following paper methodology',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Core arguments
    parser.add_argument('--output-prefix', type=str, required=True,
                        help='Prefix for all output files (e.g., output/test_01)')
    parser.add_argument('--seed', type=int, default=DEFAULT_RANDOM_SEED,
                        help=f'Random seed (default: {DEFAULT_RANDOM_SEED})')
    parser.add_argument('--threads', type=int, default=cpu_count(),
                        help=f'Number of threads (default: {cpu_count()})')
    parser.add_argument('--static', action='store_true',
                        help='Use static (deterministic) iterative selection instead of random sampling.')
    parser.add_argument('--log-exclusions', action='store_true', default=True,
                        help='Log detailed reasons for structure exclusions (default: True)')

    # Arguments for resolving ambiguities in the paper
    ambiguity_group = parser.add_argument_group('Ambiguity Resolution Arguments')
    ambiguity_group.add_argument('--cluster-mapping-mode', choices=['manifest', 'pdb_chain', 'sequence_hash'], default='manifest',
                                 help="""Method for mapping a protein chain to its cluster ID.
'manifest': (Default) Use the cluster_id already present in each chain from the manifest.
'pdb_chain': Use PDB ID + chain name (e.g., 7X2A_A) to look up in external cluster file.
'sequence_hash': Use sequence_hash to look up in external cluster file (WARNING: won't work with current data).""")
    ambiguity_group.add_argument('--lipinski-mode', choices=['any', 'all'], default='all',
                                 help="Policy for applying Lipinski's rule to structures with multiple small molecules.\n"
                                      "The paper says \'The small-molecule satisfies...\', which is ambiguous.\n"
                                      "'all': (Default) All small molecules must pass.\n"
                                      "'any': At least one small molecule must pass.")
    ambiguity_group.add_argument('--include-ions', action='store_true', default=True,
                                 help="Include single-atom ions in the 'small molecule' category for selection.\n"
                                      "The paper is ambiguous: the definition implies >1 heavy atom,\n"
                                      "but the selection step says '...small-molecules or ions'.\n"
                                      "(Default: True, based on the more specific selection text)")
    ambiguity_group.add_argument('--unmapped-chain-policy', choices=['reject', 'ignore'], default='reject',
                                 help="Policy for handling protein chains not found in the cluster map.\n"
                                      "The paper requires chains to be novel, but is ambiguous about unmapped chains.\n"
                                      "'reject': (Default) Exclude structure if any chain is unmappable.\n"
                                      "'ignore': Ignore unmappable chains and judge novelty on the rest.")
    ambiguity_group.add_argument('--entity-mode', choices=['chains', 'polymeric-only', 'all'], default='all',
                                 help='Method for counting entities for the "> 20 entities" filter.\n'
                                      'The paper does not define \'entity\'.\n'
                                      '\'all\': (Default) Count unique polymeric sequences and unique non-polymers.\n'
                                      '\'polymeric-only\': Count only unique polymeric sequences.\n'
                                      '\'chains\': Count all chains.')

    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # ==================================================
    # Configuration Summary
    # ==================================================
    print("="*80)
    print("BOLTZ-2 VALIDATION SPLIT - PAPER METHODOLOGY")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Output prefix: {args.output_prefix}")
    print(f"  Random seed: {args.seed}")
    print(f"  Threads: {args.threads}")
    print(f"  Selection mode: {'static (deterministic)' if args.static else 'paper (with random sampling)'}")
    print(f"  Log exclusions: {args.log_exclusions}")
    print(f"\nAmbiguity Settings:")
    print(f"  Lipinski mode: {args.lipinski_mode.upper()} molecules must pass")
    print(f"  Include ions: {args.include_ions}")
    print(f"  Unmapped chain policy: {args.unmapped_chain_policy}")
    print(f"  Entity counting mode: {args.entity_mode}")
    print(f"\nPaper parameters:")
    print(f"  Date range: {DATE_START} to {DATE_END}")
    print(f"  Max resolution: {MAX_RESOLUTION} Å")
    print(f"  Sequence identity: {MIN_SEQ_ID * 100:.0f}%")
    print(f"  Max residues: {MAX_RESIDUES}")
    print(f"  Max entities: {MAX_ENTITIES}")
    print(f"  Multimer keep rate: {MULTIMER_KEEP_RATE * 100:.0f}%")
    print(f"  Monomer keep rate: {MONOMER_KEEP_RATE * 100:.0f}%")
    
    # ==================================================
    # STEP 0: Preload frequently-used data into memory (if not already loaded)
    # ==================================================
    global PRELOADED
    if not PRELOADED.loaded:
        print("\n" + "="*80)
        print("PRELOADING DATA INTO MEMORY (one-time, uses up to 100GB)")
        print("="*80)
        
        # Load manifest (736MB)
        print("\nLoading manifest (736MB)...")
        start = time.time()
        with open(MANIFEST, 'r') as f:
            PRELOADED.manifest = json.load(f)
        print(f"  Loaded {len(PRELOADED.manifest)} structures in {time.time()-start:.1f}s")
        
        # Load clusters (small)
        print("\nLoading cluster mappings...")
        start = time.time()
        PRELOADED.cluster_map = {}
        
        if args.cluster_mapping_mode == 'manifest':
            # No external cluster file needed for manifest mode
            print("  Using cluster_ids from manifest (no external file needed)")
        elif args.cluster_mapping_mode == 'pdb_chain':
            # The file is tab-separated: cluster_id<TAB>pdb_chain
            with open(CLUSTERS, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        cluster_id, pdb_chain = parts
                        PRELOADED.cluster_map[pdb_chain] = cluster_id
            print(f"  Loaded {len(PRELOADED.cluster_map)} PDB_Chain-to-cluster mappings in {time.time()-start:.1f}s")
        elif args.cluster_mapping_mode == 'sequence_hash':
            # This expects a different file format that doesn't exist
            with open(CLUSTERS, 'r') as f:
                for line in f:
                    try:
                        cluster_id, seq_id = line.strip().split('\t')
                        PRELOADED.cluster_map[seq_id] = cluster_id
                    except ValueError:
                        # This will happen if the file is not in sequence_hash format
                        continue
            print(f"  Loaded {len(PRELOADED.cluster_map)} sequence_hash-to-cluster mappings in {time.time()-start:.1f}s")
        
        # Load CCD data (364MB)
        if os.path.exists(CCD_PATH):
            print("\nLoading CCD data (364MB)...")
            start = time.time()
            with open(CCD_PATH, 'rb') as f:
                PRELOADED.ccd_data = pickle.load(f)
            print(f"  Loaded {len(PRELOADED.ccd_data)} entries in {time.time()-start:.1f}s")
        else:
            print("WARNING: CCD data not found")
            PRELOADED.ccd_data = {}
        
        PRELOADED.loaded = True
        print("\nAll data preloaded into memory!")
    else:
        print("\n" + "="*80)
        print("Using previously loaded data from memory")
        print("="*80)
    
    # Use preloaded data
    manifest = PRELOADED.manifest
    cluster_map = PRELOADED.cluster_map
    ccd_data = PRELOADED.ccd_data
    
    # ==================================================
    # STEP 1: Load and filter structures
    # ==================================================
    print("\n" + "="*80)
    print("STEP 1: Applying date/resolution filters")
    print("="*80)
    
    print(f"Total structures in manifest: {len(manifest)}")
    
    print("\nApplying filters from paper Section A.1.5:")
    print("Paper quote: '1. Initial release date is before 2023-06-01 (exclusive) and 2024-01-01 (inclusive)'")
    print("Paper quote: '2. Resolution is below 4.5 Å'")
    
    # Apply date filter
    date_filtered = [
        record for record in manifest
        if check_date_range(record, DATE_START, DATE_END)
    ]
    print(f"  After date filter ({DATE_START} < date <= {DATE_END}): {len(date_filtered)} structures")
    
    # Apply resolution filter
    validation_candidates = [
        record for record in date_filtered
        if check_resolution(record, MAX_RESOLUTION)
    ]
    print(f"  After resolution filter (<= {MAX_RESOLUTION} Å): {len(validation_candidates)} structures")
    
    # ==================================================
    # STEP 2: Identify training clusters
    # ==================================================
    print("\n" + "="*80)
    print("STEP 2: Identifying training clusters")
    print("="*80)
    
    print(f"Using {len(cluster_map)} preloaded sequence-to-cluster mappings")
    
    # Check if training data is already preloaded
    if PRELOADED.training_clusters is None:
        print("\nIdentifying training clusters (structures before 2023-06-01)...")
        training_clusters = set()
        training_structures = []
        
        print("  Scanning manifest for training structures...")
        for record in tqdm(manifest, desc="  Processing structures"):
            if check_date_range(record, datetime.date(1900, 1, 1), DATE_START):
                training_structures.append(record)
                for chain in record['chains']:
                    if chain['mol_type'] == 0:  # Protein
                        if args.cluster_mapping_mode == 'manifest':
                            # Use the cluster_id already in the manifest
                            cluster_id = chain.get('cluster_id')
                            if cluster_id:
                                training_clusters.add(cluster_id)
                        elif args.cluster_mapping_mode == 'pdb_chain':
                            # Use PDB_ID + chain name
                            chain_name = chain.get('chain_name')
                            if chain_name:
                                key = f"{record['id']}_{chain_name}"
                                if key in cluster_map:
                                    training_clusters.add(cluster_map[key])
                        elif args.cluster_mapping_mode == 'sequence_hash':
                            # Use sequence_hash (won't work with current data)
                            seq_hash = chain.get('sequence_hash')
                            if seq_hash and seq_hash in cluster_map:
                                training_clusters.add(cluster_map[seq_hash])
        
        # Store in preloaded data
        PRELOADED.training_clusters = training_clusters
        PRELOADED.training_structures = training_structures
        
        print(f"  Training structures (released before {DATE_START}): {len(training_structures)}")
        print(f"  Unique training clusters identified: {len(training_clusters)}")
    else:
        print("\nUsing preloaded training clusters...")
        training_clusters = PRELOADED.training_clusters
        training_structures = PRELOADED.training_structures
        print(f"  Training structures: {len(training_structures)}")
        print(f"  Training clusters: {len(training_clusters)}")
    
    # ==================================================
    # STEP 3: Filter by novel sequences (PARALLELIZED)
    # ==================================================
    print("\n" + "="*80)
    print("STEP 3: Filtering by novel protein sequences")
    print("="*80)
    print("Paper quote (Section A.1.5, point 3):")
    print("'All the protein sequences of the chains are not present in any training set clusters'")
    
    print(f"  Filtering structures with novel protein sequences (policy: {args.unmapped_chain_policy}, using {args.threads} threads)...")
    
    # Process in parallel using partial function
    from functools import partial
    check_novelty_func = partial(check_novelty_worker, 
                                  cluster_map=cluster_map, 
                                  training_clusters=training_clusters,
                                  unmapped_chain_policy=args.unmapped_chain_policy,
                                  mapping_mode=args.cluster_mapping_mode)
    
    novel_structures = []
    excluded_by_training_cluster = 0
    excluded_by_unmapped = 0
    
    with Pool(args.threads) as pool:
        results = list(tqdm(
            pool.imap_unordered(check_novelty_func, validation_candidates, chunksize=100),
            total=len(validation_candidates),
            desc="  Checking novelty"
        ))
    
    # Process results
    for result in results:
        if result['unmapped']:
            excluded_by_unmapped += 1
        elif result['novel']:
            novel_structures.append(result['record'])
        else:
            excluded_by_training_cluster += 1
            
    validation_candidates = novel_structures
    print(f"  Structures with only novel protein sequences: {len(validation_candidates)}")
    print(f"  Structures excluded (contain training sequences): {excluded_by_training_cluster}")
    if args.unmapped_chain_policy == 'reject':
        print(f"  Structures excluded (unmapped chain policy=reject): {excluded_by_unmapped}")

    # ==================================================
    # STEP 4: Small molecule filtering
    # ==================================================
    print("\n" + "="*80)
    print("STEP 4: Small molecule filtering")
    print("="*80)
    
    print("Using preloaded CCD (Chemical Component Dictionary) data...")
    if ccd_data:
        print(f"Using {len(ccd_data)} CCD entries from memory")
    else:
        print("WARNING: CCD data not available, skipping small molecule filtering")
    
    if ccd_data:
        # Check if training molecules are already preloaded
        if PRELOADED.training_molecules is None or PRELOADED.training_fingerprints is None:
            print("\nExtracting small molecules from training structures...")
            
            print("  Identifying training structures with small molecules...")
            training_with_molecules = []
            for record in tqdm(training_structures, desc="  Scanning"):
                if has_small_molecules(record):
                    training_with_molecules.append(record)
            
            print(f"  Training structures with small molecules: {len(training_with_molecules)}")
            
            print(f"Extracting CCD codes from training structures (using {args.threads} threads)...")
            with Pool(args.threads) as pool:
                ccd_sets = list(tqdm(
                    pool.imap(extract_ccd_codes, training_with_molecules),
                    total=len(training_with_molecules),
                    desc="  Extracting CCD codes"
                ))
            
            training_ccd_codes = set().union(*ccd_sets)
            print(f"Unique CCD codes in training set: {len(training_ccd_codes)}")
            
            print("Generating Morgan fingerprints for training molecules...")
            training_mols = {
                ccd_code: ccd_data[ccd_code]
                for ccd_code in tqdm(training_ccd_codes, desc="  Loading molecules")
                if ccd_code in ccd_data and ccd_data[ccd_code] is not None
            }
            
            training_fps = generate_morgan_fingerprints(training_mols)
            print(f"Generated fingerprints for {len(training_fps)} training molecules")
            
            # Store in preloaded data
            PRELOADED.training_molecules = training_mols
            PRELOADED.training_fingerprints = training_fps
        else:
            print("\nUsing preloaded training molecules and fingerprints...")
            training_mols = PRELOADED.training_molecules
            training_fps = PRELOADED.training_fingerprints
            print(f"  Training molecules: {len(training_mols)}")
            print(f"  Training fingerprints: {len(training_fps)}")
        
        print("\nFiltering validation candidates by small molecule criteria...")
        print("Paper quote (Section A.1.5, point 4): 'Either:")
        print("  * No small-molecule is present.")
        print("  * At least one of the small-molecules exhibits a Tanimoto similarity of 0.85 or less")
        print("    to any small-molecule in the training set.")
        print("  * The small-molecule satisfies Lipinski's Rule of Five.'")
        print(f"\n  Processing {len(validation_candidates)} structures (using {args.threads} threads)...")
        
        # Process in parallel using partial function
        check_small_molecules_func = partial(check_small_molecules_worker,
                                              ccd_data=ccd_data,
                                              include_ions=args.include_ions,
                                              lipinski_mode=args.lipinski_mode,
                                              training_fps=training_fps)
        
        filtered_candidates = []
        no_mol_count = 0
        lipinski_count = 0
        tanimoto_count = 0
        
        with Pool(args.threads) as pool:
            results = list(tqdm(
                pool.imap_unordered(check_small_molecules_func, validation_candidates, chunksize=10),
                total=len(validation_candidates),
                desc="  Processing structures"
            ))
        
        # Process results
        for result in results:
            if result['passed']:
                filtered_candidates.append(result['record'])
                if result['reason'] == 'no_molecules':
                    no_mol_count += 1
                elif result['reason'] == 'lipinski':
                    lipinski_count += 1
                elif result['reason'] == 'tanimoto':
                    tanimoto_count += 1
        
        validation_candidates = filtered_candidates
        print(f"\n  Filter results:")
        print(f"    No small molecules: {no_mol_count}")
        print(f"    Passed Lipinski: {lipinski_count}")
        print(f"    Passed Tanimoto (≤0.85): {tanimoto_count}")
        print(f"  Total after small molecule filtering: {len(validation_candidates)}")
    
    # ==================================================
    # STEP 5: Size constraints (PARALLELIZED)
    # ==================================================
    print("\n" + "="*80)
    print("STEP 5: Applying size constraints")
    print("="*80)
    print("Paper quotes (Section A.1.5):")
    print(f"  '1. Retain structures with at most {MAX_RESIDUES} residues'")
    print(f"  '2. Exclude complexes with more than {MAX_ENTITIES} entities'")
    
    print(f"\n  Applying filters (entity mode: {args.entity_mode}, using {args.threads} threads)...")
    
    # Process in parallel using partial function
    check_size_func = partial(check_size_constraints_worker,
                              entity_mode=args.entity_mode,
                              max_residues=MAX_RESIDUES,
                              max_entities=MAX_ENTITIES)
    
    size_filtered = []
    residue_excluded = 0
    entity_excluded = 0
    entity_unknown = 0
    exclusion_tracker = defaultdict(list) if args.log_exclusions else None
    
    with Pool(args.threads) as pool:
        results = list(tqdm(
            pool.imap_unordered(check_size_func, validation_candidates, chunksize=10),
            total=len(validation_candidates),
            desc="  Checking sizes"
        ))
    
    # Process results
    for result in results:
        if result['passed']:
            size_filtered.append(result['record'])
        else:
            if result['exclusion_reason'] == 'residue_limit':
                residue_excluded += 1
                if exclusion_tracker:
                    exclusion_tracker['residue_limit'].append((result['record']['id'], result['residues']))
            elif result['exclusion_reason'] == 'entity_unknown':
                entity_unknown += 1
            elif result['exclusion_reason'] == 'entity_limit':
                entity_excluded += 1
                if exclusion_tracker:
                    exclusion_tracker['entity_limit'].append((result['record']['id'], result['entities']))
    
    validation_candidates = size_filtered
    print(f"  Excluded by residue count (>{MAX_RESIDUES}): {residue_excluded}")
    print(f"  Excluded by entity count (>{MAX_ENTITIES}): {entity_excluded}")
    print(f"  Excluded due to unknown entity count: {entity_unknown}")
    print(f"  Structures after size filtering: {len(validation_candidates)}")
    
    if args.log_exclusions and exclusion_tracker:
        print("\n  Detailed exclusion reasons:")
        if exclusion_tracker.get('no_cif_for_entity_count'):
             print(f"    No CIF for entity count: {len(exclusion_tracker['no_cif_for_entity_count'])} structures")
        if exclusion_tracker.get('entity_parse_error'):
             print(f"    CIF parse errors for entity count: {len(exclusion_tracker['entity_parse_error'])} structures")

    # ==================================================
    # STEP 6: Categorize structures
    # ==================================================
    print("\n" + "="*80)
    print("STEP 6: Categorizing structures by content")
    print("="*80)
    
    print("\n  Analyzing molecular content...")
    rna_dna_structures = []
    remaining = []
    
    for record in tqdm(validation_candidates, desc="  Categorizing"):
        if has_nucleic_acids(record):
            rna_dna_structures.append(record)
        else:
            remaining.append(record)
    
    print(f"  RNA/DNA structures (retained automatically): {len(rna_dna_structures)}")
    print(f"  Other structures (for iterative selection): {len(remaining)}")
    
    # ==================================================
    # STEP 7: Iterative selection
    # ==================================================
    print("\n" + "="*80)
    print("STEP 7: Iterative selection to balance dataset")
    print("="*80)
    print("\nPaper quotes (Section A.1.5, iterative selection):")
    print("  '3. Retaining all the structures containing RNA or DNA entities'")
    print("  '4. Iteratively adding structures containing small-molecules or ions under the")
    print("      condition that all their protein chains belong to new unseen clusters'")
    print(f"  '5. Iteratively adding multimeric structures under the condition that all the")
    print(f"      protein chains belong to new unseen clusters. These are further filtered by")
    print(f"      randomly keeping only {int(MULTIMER_KEEP_RATE*100)}% of the passing structures'")
    print(f"  '6. Iteratively adding monomers under the condition that their chain belongs to")
    print(f"      a new unseen cluster. These are further randomly filtered out by keeping")
    print(f"      only {int(MONOMER_KEEP_RATE*100)}% of the passing structures'")
    
    if args.static:
        validation_set = run_iterative_selection_static(
            remaining, rna_dna_structures, cluster_map, ccd_data,
            args.unmapped_chain_policy, args.include_ions,
            MULTIMER_KEEP_RATE, MONOMER_KEEP_RATE,
            args.cluster_mapping_mode
        )
    else:
        validation_set = run_iterative_selection_paper(
            remaining, rna_dna_structures, cluster_map, ccd_data,
            args.unmapped_chain_policy, args.include_ions,
            MULTIMER_KEEP_RATE, MONOMER_KEEP_RATE,
            args.cluster_mapping_mode
        )
    
    # ==================================================
    # Final results
    # ==================================================
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\nTotal validation structures: {len(validation_set)}")
    
    output_file = f"{args.output_prefix}_ids.txt"
    with open(output_file, 'w') as f:
        for record in sorted(validation_set, key=lambda x: x['id']):
            f.write(record['id'] + '\n')
    print(f"\nValidation IDs saved to: {output_file}")
    
    # Save metadata file with parameters and exclusion details
    metadata_file = f"{args.output_prefix}_metadata.json"
    metadata = {
        'parameters': {
            'seed': args.seed,
            'lipinski_mode': args.lipinski_mode,
            'selection_mode': 'static' if args.static else 'random',
            'include_ions': args.include_ions,
            'unmapped_chain_policy': args.unmapped_chain_policy,
            'entity_mode': args.entity_mode,
            'date_range': f"{DATE_START} to {DATE_END}",
            'max_resolution': MAX_RESOLUTION,
            'max_residues': MAX_RESIDUES,
            'max_entities': MAX_ENTITIES,
            'multimer_keep_rate': MULTIMER_KEEP_RATE,
            'monomer_keep_rate': MONOMER_KEEP_RATE
        },
        'counts': {
            'final_validation_set': len(validation_set)
        },
        'exclusions': {
            'residue_limit_exceeded': residue_excluded,
            'entity_limit_exceeded': entity_excluded,
            'entity_count_unknown': entity_unknown,
            'excluded_by_training_cluster': excluded_by_training_cluster
        }
    }
    
    if args.log_exclusions and exclusion_tracker:
        metadata['detailed_exclusions'] = {
            key: len(values) for key, values in exclusion_tracker.items()
        }
        
        # Save detailed exclusion log
        exclusion_log_file = f"{args.output_prefix}_exclusions.json"
        with open(exclusion_log_file, 'w') as f:
            json.dump(dict(exclusion_tracker), f, indent=2)
        print(f"Detailed exclusion log saved to: {exclusion_log_file}")
    
    # Add excluded_by_unmapped only if relevant
    if args.unmapped_chain_policy == 'reject':
        metadata['exclusions']['excluded_by_unmapped_chain'] = excluded_by_unmapped
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_file}")
    
    # Calculate detailed breakdown
    categories = defaultdict(list)
    for record in validation_set:
        category = categorize_structure(record, ccd_data, args.include_ions)
        categories[category].append(record['id'])
    
    print("\nFinal breakdown by category:")
    for category, ids in sorted(categories.items()):
        print(f"  {category}: {len(ids)}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f"Started with: {len(manifest)} structures in manifest")
    print(f"Final validation set: {len(validation_set)} structures")

if __name__ == "__main__":
    main()