# Boltz-2 Validation Dataset Split

Replication of the validation dataset split from the Boltz-2 paper (398 structures).

## Files

- `boltz2_val_split.py` - Main script implementing the Boltz-2 validation split methodology
- `test_script.sh` - Test harness running multiple configurations to match the paper's count
- `test_script.out` - Test results showing different configuration outcomes
- `reference.ipynb` - Reference notebook by sh.j with initial implementation attempts

## Usage

```bash
python boltz2_val_split.py
```

Or run all test configurations:
```bash
./test_script.sh
```

## Paper Methodology (Section A.1.5)

**Exact quote from Boltz-2 paper:**

"Our training, validation and test splitting strategy largely follows Boltz-1 procedure. We first cluster the protein sequences in PDB by sequence identity with the command `mmseqs easy-cluster... min-seq-id 0.4`. Then, we select all structures in PDB satisfying the following filters:

1. Initial release date is before 2023-06-01 (exclusive) and 2024-01-01 (inclusive).
2. Resolution is below 4.5 Å.
3. All the protein sequences of the chains are not present in any training set clusters (i.e. before 2023-06-01).
4. Either:
   * No small-molecule is present.
   * At least one of the small-molecules exhibits a Tanimoto similarity of 0.85 or less to any small-molecule in the training set. Here, a small-molecule is defined as any non-polymer entity containing more than one heavy atom and not included in the ligand exclusion list.
   * The small-molecule satisfies Lipinski's Rule of Five.

We further refine through the following steps:
1. Retain structures with at most 1024 residues.
2. Exclude complexes with more than 20 entities.
3. Retaining all the structures containing RNA or DNA entities.
4. Iteratively adding structures containing small-molecules or ions under the condition that all their protein chains belong to new unseen clusters.
5. Iteratively adding multimeric structures under the condition that all the protein chains belong to new unseen clusters. These are further filtered by randomly keeping only 90% of the passing structures.
6. Iteratively adding monomers under the condition that their chain belongs to a new unseen cluster. These are further randomly filtered out by keeping only 60% of the passing structures.

This results in a total of 398 structures from PDB in our validation set."

## Script Options (boltz2_val_split.py)

The script has **7 configurable options** to resolve paper ambiguities:

1. `--cluster-mapping-mode`: How to map chains to clusters (manifest/pdb_chain/sequence_hash)
2. `--lipinski-mode`: Whether ALL or ANY small molecule must pass Lipinski
3. `--include-ions`: Whether to include single-atom ions as small molecules
4. `--unmapped-chain-policy`: How to handle chains not in cluster map (reject/ignore)
5. `--entity-mode`: How to count entities for 20-entity limit (chains/polymeric-only/all)
6. `--static`: Use deterministic selection instead of random sampling
7. `--seed`: Random seed for reproducible sampling

The test script explores 26 different combinations to match the reported 398 structures.