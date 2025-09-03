#!/bin/bash

# This script runs the Boltz-2 validation split with a few, highly-focused
# configurations to determine the correct cluster mapping strategy.

# Unset variables are errors, and pipe failures are caught.
set -uo pipefail

# --- Global Variables ---
OUTPUT_DIR="./results"
PASSED_TESTS=()
FAILED_TESTS=()

# --- Prerequisite Checks ---
check_prerequisites() {
    echo "--- Checking prerequisites ---"
    if ! command -v python3 &> /dev/null; then
        echo "ERROR: python3 command not found. Please ensure Python 3 is installed and in your PATH." >&2
        exit 1
    fi
    local python_script="boltz2_validation_split_clean.py"
    if [ ! -f "$python_script" ]; then
        echo "ERROR: The script '$python_script' was not found in the current directory." >&2
        exit 1
    fi
    echo "--- Prerequisites met ---"
}

# --- Helper Function to Run a Single Test Configuration ---
run_test() {
    local test_name="$1"
    shift # Remove the first argument, the rest are the python script args
    local args="$@"

    local output_prefix="${OUTPUT_DIR}/${test_name}"
    local log_file="${output_prefix}.log"
    local command_to_run="python3 boltz2_validation_split_clean.py --output-prefix $output_prefix $args"

    echo -n "Running test: $test_name... "
    
    # Run the command once, redirecting all stdout and stderr to the log file.
    eval "$command_to_run" > "$log_file" 2>&1
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "FAILED (Exit Code: $exit_code). See log: $log_file" >&2
        FAILED_TESTS+=("$test_name")
    else
        # Grep for the result line and extract the 4th field (the number)
        local final_count=$(grep "Total validation structures:" "$log_file" | awk '{print $4}' || true)

        if [ -n "$final_count" ]; then
            echo "OK. Result: $final_count"
            PASSED_TESTS+=("$test_name")
        else
            echo "FAILED. Final count not found in log: $log_file" >&2
            FAILED_TESTS+=("$test_name")
        fi
    fi
}

# --- Test Matrix ---
main() {
    check_prerequisites
    mkdir -p "$OUTPUT_DIR"

    echo
    echo "--- Running Comprehensive Configuration Tests ---"

    # ======= TIER 1: Most Probable Configurations =======
    
    # Test 1: Polymeric entity counting with strict Lipinski
    # Rationale: In structural biology, "entities" typically means unique polymers, not every small molecule
    # The paper's strict wording for Lipinski suggests all molecules must pass
    run_test "01_polymeric_all_lipinski" \
        --threads 60 \
        --cluster-mapping-mode manifest \
        --lipinski-mode all \
        --entity-mode polymeric-only \
        --include-ions \
        --unmapped-chain-policy reject \
        --seed 42

    # Test 2: Count all entities (including small molecules)
    # Rationale: Literal interpretation of "entities" from PDB definition includes all unique molecular components
    run_test "02_all_entities_strict" \
        --threads 60 \
        --cluster-mapping-mode manifest \
        --lipinski-mode all \
        --entity-mode all \
        --include-ions \
        --unmapped-chain-policy reject \
        --seed 42

    # Test 3: Lenient Lipinski with polymeric entities
    # Rationale: Paper says "at least one" for Tanimoto, might apply same logic to Lipinski
    run_test "03_polymeric_any_lipinski" \
        --threads 60 \
        --cluster-mapping-mode manifest \
        --lipinski-mode any \
        --entity-mode polymeric-only \
        --include-ions \
        --unmapped-chain-policy reject \
        --seed 42

    # Test 4: Lenient Lipinski with all entities
    # Rationale: Combines lenient small molecule filtering with comprehensive entity counting
    run_test "04_all_entities_lenient" \
        --threads 60 \
        --cluster-mapping-mode manifest \
        --lipinski-mode any \
        --entity-mode all \
        --include-ions \
        --unmapped-chain-policy reject \
        --seed 42

    # ======= TIER 2: Alternative Mapping & Entity Methods =======

    # Test 5: PDB chain mapping with polymeric entities
    # Rationale: External cluster file might have more complete mappings for newer structures
    run_test "05_pdb_chain_polymeric" \
        --threads 60 \
        --cluster-mapping-mode pdb_chain \
        --lipinski-mode all \
        --entity-mode polymeric-only \
        --include-ions \
        --unmapped-chain-policy reject \
        --seed 42

    # Test 6: PDB chain mapping with all entities
    # Rationale: Combines external mapping with comprehensive entity counting
    run_test "06_pdb_chain_all_entities" \
        --threads 60 \
        --cluster-mapping-mode pdb_chain \
        --lipinski-mode all \
        --entity-mode all \
        --include-ions \
        --unmapped-chain-policy reject \
        --seed 42

    # Test 7: Count chains instead of entities
    # Rationale: Simplest interpretation - "entities" might mean individual chains in the structure
    run_test "07_chain_counting" \
        --threads 60 \
        --cluster-mapping-mode manifest \
        --lipinski-mode all \
        --entity-mode chains \
        --include-ions \
        --unmapped-chain-policy reject \
        --seed 42

    # Test 8: PDB chain with lenient Lipinski
    # Rationale: External mapping combined with relaxed small molecule criteria
    run_test "08_pdb_chain_lenient" \
        --threads 60 \
        --cluster-mapping-mode pdb_chain \
        --lipinski-mode any \
        --entity-mode polymeric-only \
        --include-ions \
        --unmapped-chain-policy reject \
        --seed 42

    # ======= TIER 3: Ion Exclusion Variants =======

    # Test 9: Exclude ions with polymeric entities
    # Rationale: Paper definition says ">1 heavy atom", ions have only 1
    run_test "09_no_ions_polymeric" \
        --threads 60 \
        --cluster-mapping-mode manifest \
        --lipinski-mode all \
        --entity-mode polymeric-only \
        --unmapped-chain-policy reject \
        --seed 42
        # Note: --include-ions flag omitted

    # Test 10: Exclude ions with all entities
    # Rationale: Strict interpretation of small molecule definition excludes single atoms
    run_test "10_no_ions_all_entities" \
        --threads 60 \
        --cluster-mapping-mode manifest \
        --lipinski-mode all \
        --entity-mode all \
        --unmapped-chain-policy reject \
        --seed 42
        # Note: --include-ions flag omitted

    # Test 11: Exclude ions with lenient Lipinski
    # Rationale: Combines strict ion exclusion with relaxed Lipinski criteria
    run_test "11_no_ions_lenient" \
        --threads 60 \
        --cluster-mapping-mode manifest \
        --lipinski-mode any \
        --entity-mode polymeric-only \
        --unmapped-chain-policy reject \
        --seed 42
        # Note: --include-ions flag omitted

    # ======= TIER 4: Deterministic Selection =======

    # Test 12: Static selection with polymeric entities
    # Rationale: Deterministic selection (first N%) instead of random sampling for reproducibility
    run_test "12_static_polymeric" \
        --threads 60 \
        --cluster-mapping-mode manifest \
        --lipinski-mode all \
        --entity-mode polymeric-only \
        --include-ions \
        --unmapped-chain-policy reject \
        --seed 42 \
        --static

    # Test 13: Static selection with all entities
    # Rationale: Deterministic approach with comprehensive entity counting
    run_test "13_static_all_entities" \
        --threads 60 \
        --cluster-mapping-mode manifest \
        --lipinski-mode all \
        --entity-mode all \
        --include-ions \
        --unmapped-chain-policy reject \
        --seed 42 \
        --static

    # Test 14: Static selection with lenient Lipinski
    # Rationale: Deterministic selection combined with relaxed small molecule criteria
    run_test "14_static_lenient" \
        --threads 60 \
        --cluster-mapping-mode manifest \
        --lipinski-mode any \
        --entity-mode polymeric-only \
        --include-ions \
        --unmapped-chain-policy reject \
        --seed 42 \
        --static

    # ======= TIER 5: Alternative Random Seeds =======

    # Test 15: Seed 0
    # Rationale: Different random seed might hit exact target if close with seed 42
    run_test "15_seed_0" \
        --threads 60 \
        --cluster-mapping-mode manifest \
        --lipinski-mode all \
        --entity-mode polymeric-only \
        --include-ions \
        --unmapped-chain-policy reject \
        --seed 0

    # Test 16: Seed 1
    # Rationale: Another common seed value for reproducibility testing
    run_test "16_seed_1" \
        --threads 60 \
        --cluster-mapping-mode manifest \
        --lipinski-mode all \
        --entity-mode polymeric-only \
        --include-ions \
        --unmapped-chain-policy reject \
        --seed 1

    # Test 17: Seed 123
    # Rationale: Alternative seed that might match author's choice
    run_test "17_seed_123" \
        --threads 60 \
        --cluster-mapping-mode manifest \
        --lipinski-mode all \
        --entity-mode polymeric-only \
        --include-ions \
        --unmapped-chain-policy reject \
        --seed 123

    # ======= TIER 6: Unmapped Chain Policy Variations =======

    # Test 18: Ignore unmapped chains with polymeric entities
    # Rationale: More lenient - structures with partial mapping issues still included
    run_test "18_ignore_unmapped_polymeric" \
        --threads 60 \
        --cluster-mapping-mode manifest \
        --lipinski-mode all \
        --entity-mode polymeric-only \
        --include-ions \
        --unmapped-chain-policy ignore \
        --seed 42

    # Test 19: Ignore unmapped chains with all entities
    # Rationale: Lenient mapping policy with comprehensive entity counting
    run_test "19_ignore_unmapped_all" \
        --threads 60 \
        --cluster-mapping-mode manifest \
        --lipinski-mode all \
        --entity-mode all \
        --include-ions \
        --unmapped-chain-policy ignore \
        --seed 42

    # Test 20: Ignore unmapped with PDB chain mapping
    # Rationale: External mapping might have gaps, ignoring unmapped prevents over-exclusion
    run_test "20_pdb_ignore_unmapped" \
        --threads 60 \
        --cluster-mapping-mode pdb_chain \
        --lipinski-mode all \
        --entity-mode polymeric-only \
        --include-ions \
        --unmapped-chain-policy ignore \
        --seed 42

    # ======= TIER 7: Less Likely Combinations =======

    # Test 21: Chain counting with lenient Lipinski
    # Rationale: Simplest entity interpretation with relaxed small molecule filter
    run_test "21_chains_lenient" \
        --threads 60 \
        --cluster-mapping-mode manifest \
        --lipinski-mode any \
        --entity-mode chains \
        --include-ions \
        --unmapped-chain-policy reject \
        --seed 42

    # Test 22: Chain counting without ions
    # Rationale: Count individual chains but exclude single-atom ions
    run_test "22_chains_no_ions" \
        --threads 60 \
        --cluster-mapping-mode manifest \
        --lipinski-mode all \
        --entity-mode chains \
        --unmapped-chain-policy reject \
        --seed 42
        # Note: --include-ions flag omitted

    # Test 23: PDB chain mapping with chain counting
    # Rationale: External mapping with simplest entity interpretation
    run_test "23_pdb_chains" \
        --threads 60 \
        --cluster-mapping-mode pdb_chain \
        --lipinski-mode all \
        --entity-mode chains \
        --include-ions \
        --unmapped-chain-policy reject \
        --seed 42

    # Test 24: Static selection without ions
    # Rationale: Deterministic selection excluding single-atom ions
    run_test "24_static_no_ions" \
        --threads 60 \
        --cluster-mapping-mode manifest \
        --lipinski-mode all \
        --entity-mode polymeric-only \
        --unmapped-chain-policy reject \
        --seed 42 \
        --static
        # Note: --include-ions flag omitted

    # Test 25: Ignore unmapped without ions
    # Rationale: Most lenient mapping policy while excluding ions
    run_test "25_ignore_no_ions" \
        --threads 60 \
        --cluster-mapping-mode manifest \
        --lipinski-mode all \
        --entity-mode polymeric-only \
        --unmapped-chain-policy ignore \
        --seed 42
        # Note: --include-ions flag omitted

    # Test 26: Sequence hash mapping (likely to fail)
    # Rationale: Theoretical best mapping by sequence identity, but data might not support it
    run_test "26_sequence_hash" \
        --threads 60 \
        --cluster-mapping-mode sequence_hash \
        --lipinski-mode all \
        --entity-mode polymeric-only \
        --include-ions \
        --unmapped-chain-policy reject \
        --seed 42

    echo
    echo "=================================================================="
    echo "All tests finished."
    echo "=================================================================="
    echo "Passed: ${#PASSED_TESTS[@]}"
    echo "Failed: ${#FAILED_TESTS[@]}"
    if [ ${#FAILED_TESTS[@]} -ne 0 ]; then
        echo "Failed tests: ${FAILED_TESTS[*]}"
        exit 1
    fi
}

# Execute the main function
main