#!/usr/bin/env python3
import re
import csv
import argparse
import numpy as np


# ----------------------------
# Defaults (same as your script)
# ----------------------------
DEFAULT_INPUT_FILE = "45.out"

# Fragment index ranges (0-based, inclusive)
DEFAULT_FRAG1_START = 0
DEFAULT_FRAG1_END_INCL = 179   # range(0, 202)
DEFAULT_FRAG2_START = 180
DEFAULT_FRAG2_END_INCL = 481   # range(202, 414)

DEFAULT_CT_THRESHOLD = 0.1


def parse_ground_state_charges(filename):
    """
    Parse the first 'MULLIKEN ATOMIC CHARGES' block (ground state).
    Returns {atom_index: charge}
    """
    charges = {}
    with open(filename, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "MULLIKEN ATOMIC CHARGES" in line and "UNRELAXED" not in line:
            i += 2  # skip header and dashed line
            while i < len(lines) and re.match(r"\s*\d+", lines[i]):
                parts = lines[i].replace(":", " ").split()
                if len(parts) >= 3:
                    idx = int(parts[0])
                    charge = float(parts[2])
                    charges[idx] = charge
                i += 1
            break

    return charges


def parse_excited_state_blocks(filename):
    """
    Parse all 'UNRELAXED CIS/TDA DENSITY POPULATION ANALYSIS' sections.
    Distinguish singlet/triplet and return:
        { state_label: {atom_index: charge} }
    where state_label is 'S1', 'S2', ... or 'T1', 'T2', ...
    """
    results = {}
    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if "UNRELAXED CIS/TDA DENSITY POPULATION ANALYSIS" in lines[i]:
            # Determine multiplicity (default singlet, but detect TRIPLET)
            header_line = lines[i + 1] if i + 1 < len(lines) else ""
            mult = "T" if "TRIPLET" in header_line.upper() else "S"

            # Find IROOT number nearby
            root = None
            for j in range(i, min(i + 10, len(lines))):
                m = re.search(r"IROOT\s+(\d+)", lines[j])
                if m:
                    root = int(m.group(1))
                    if "TRIPLET" in lines[j].upper():
                        mult = "T"
                    break

            if root is None:
                # Fallback if IROOT not found: label sequentially within file
                root = 1

            # Extra fallback/confirmation from "Input electron density ..." line
            # (often contains 'singlet' or 'triplet' in the filename)
            for j in range(i, min(i + 30, len(lines))):
                if "Input electron density" in lines[j]:
                    low = lines[j].lower()
                    if "triplet" in low:
                        mult = "T"
                    elif "singlet" in low:
                        mult = "S"
                    break

            state_key = f"{mult}{root}"

            # Find Mulliken charges section for this excited-state block
            while i < len(lines) and "MULLIKEN ATOMIC CHARGES" not in lines[i]:
                i += 1
            if i >= len(lines):
                break
            i += 2  # skip header + dashed line

            charges = {}
            # Typical ORCA line: "  0 C :  -0.123456"
            while i < len(lines) and re.match(r"\s*\d+\s+\S+\s*[:=]\s*-?\d", lines[i]):
                parts = lines[i].replace(":", " ").split()
                if len(parts) >= 3:
                    idx = int(parts[0])
                    charge = float(parts[2])
                    charges[idx] = charge
                i += 1

            # IMPORTANT: Do not overwrite if same label appears (rare but possible)
            # If it does, append suffix.
            if state_key in results:
                k = 2
                while f"{state_key}_{k}" in results:
                    k += 1
                state_key = f"{state_key}_{k}"

            results[state_key] = charges
        else:
            i += 1

    return results


def compute_fragment_charge(charges, frag_indices):
    """Sum Mulliken charges for a fragment (safe if some atoms missing)."""
    return float(np.sum([charges[i] for i in frag_indices if i in charges]))


def parse_fragment_ranges(f1s, f1e, f2s, f2e):
    """
    Convert inclusive endpoints to Python ranges.
    """
    if f1e < f1s or f2e < f2s:
        raise ValueError("Fragment end index must be >= start index.")
    frag1 = range(f1s, f1e + 1)
    frag2 = range(f2s, f2e + 1)
    return frag1, frag2


def sort_state_key(label):
    """
    Sort states in a human way:
      S1, S2, ..., T1, T2, ... then any suffixed duplicates.
    """
    # Handle possible suffix: "S1_2"
    base = label
    suffix = 0
    if "_" in label:
        base, suf = label.rsplit("_", 1)
        if suf.isdigit():
            suffix = int(suf)

    mult = base[0] if base else "Z"
    num = 10**9
    if len(base) > 1 and base[1:].isdigit():
        num = int(base[1:])
    return (mult, num, suffix, label)


def identify_ct_states(filename, frag1, frag2, threshold, csv_path=None):
    ground = parse_ground_state_charges(filename)
    excited_states = parse_excited_state_blocks(filename)

    if not ground:
        print("⚠️ No ground-state Mulliken charges found — using 0 as reference.")
    if not excited_states:
        print("❌ No excited-state Mulliken sections found.")
        return

    # Ground-state fragment charges
    ground_f1 = compute_fragment_charge(ground, frag1) if ground else 0.0
    ground_f2 = compute_fragment_charge(ground, frag2) if ground else 0.0

    print(f"\nAnalyzing file: {filename}")
    print(f"Fragment 1 atoms: {frag1.start}–{frag1.stop - 1} (0-based, inclusive)")
    print(f"Fragment 2 atoms: {frag2.start}–{frag2.stop - 1} (0-based, inclusive)")
    print(f"Charge-transfer threshold: {threshold:.3f} e\n")

    print("=== Ground-State Fragment Charges ===")
    print(f"Frag1_Q (ground) = {ground_f1:.6f}")
    print(f"Frag2_Q (ground) = {ground_f2:.6f}")
    print("-" * 78)

    header = (
        f"{'State':<8}"
        f"{'Frag1_Q':>12}"
        f"{'Frag2_Q':>12}"
        f"{'ΔQ_F1':>12}"
        f"{'ΔQ_F2':>12}"
        f"{'CT_strength':>14}"
        f"{'CT?':>6}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for state, charges in sorted(excited_states.items(), key=lambda kv: sort_state_key(kv[0])):
        f1 = compute_fragment_charge(charges, frag1)
        f2 = compute_fragment_charge(charges, frag2)

        d1 = f1 - ground_f1
        d2 = f2 - ground_f2

        # A simple magnitude metric for "how much charge moved"
        ct_strength = 0.5 * (abs(d1) + abs(d2))

        # CT definition: significant transfer and opposite signs between fragments
        # (meaning one fragment gains while the other loses)
        ct_flag = (abs(d1) >= threshold) and (np.sign(d1) != np.sign(d2)) and (d1 != 0.0) and (d2 != 0.0)

        print(
            f"{state:<8}"
            f"{f1:12.6f}"
            f"{f2:12.6f}"
            f"{d1:12.6f}"
            f"{d2:12.6f}"
            f"{ct_strength:14.6f}"
            f"{'YES' if ct_flag else 'NO':>6}"
        )

        rows.append({
            "state": state,
            "frag1_q": f1,
            "frag2_q": f2,
            "delta_q_frag1": d1,
            "delta_q_frag2": d2,
            "ct_strength": ct_strength,
            "ct_flag": "YES" if ct_flag else "NO",
            "threshold": threshold,
            "ground_frag1_q": ground_f1,
            "ground_frag2_q": ground_f2,
            "frag1_start": frag1.start,
            "frag1_end_incl": frag1.stop - 1,
            "frag2_start": frag2.start,
            "frag2_end_incl": frag2.stop - 1,
        })

    if csv_path:
        fieldnames = list(rows[0].keys()) if rows else []
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n✅ Wrote CSV results to: {csv_path}")

    print("\nDone.\n")


def build_argparser():
    p = argparse.ArgumentParser(
        description="Detect charge-transfer (CT) excited states from ORCA Mulliken charges (singlet + triplet supported)."
    )
    p.add_argument("-i", "--input", default=DEFAULT_INPUT_FILE, help="ORCA output file (default: 35.out)")
    p.add_argument("--frag1-start", type=int, default=DEFAULT_FRAG1_START, help="Fragment 1 start atom index (0-based)")
    p.add_argument("--frag1-end", type=int, default=DEFAULT_FRAG1_END_INCL, help="Fragment 1 end atom index (inclusive)")
    p.add_argument("--frag2-start", type=int, default=DEFAULT_FRAG2_START, help="Fragment 2 start atom index (0-based)")
    p.add_argument("--frag2-end", type=int, default=DEFAULT_FRAG2_END_INCL, help="Fragment 2 end atom index (inclusive)")
    p.add_argument("-t", "--threshold", type=float, default=DEFAULT_CT_THRESHOLD, help="CT threshold in e (default: 0.1)")
    p.add_argument("--csv", default=None, help="Optional CSV output path (e.g., results.csv)")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()

    frag1, frag2 = parse_fragment_ranges(
        args.frag1_start, args.frag1_end,
        args.frag2_start, args.frag2_end
    )

    identify_ct_states(
        filename=args.input,
        frag1=frag1,
        frag2=frag2,
        threshold=args.threshold,
        csv_path=args.csv
    )

