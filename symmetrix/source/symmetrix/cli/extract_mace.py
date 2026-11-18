#!/usr/bin/env python3

from argparse import ArgumentParser

from ..extract_mace_data import extract_mace_data


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", "-m", required=True, help="Torch model file.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--atomic-numbers",
        "-Z",
        "-z",
        nargs="+",
        help="Atomic numbers to extract.",
        default=[],
    )
    group.add_argument(
        "--chemical-symbols",
        "-s",
        nargs="+",
        help="Chemical symbols to extract.",
        default=[],
    )
    parser.add_argument(
        "--head",
        "-H",
        help="Head to keep for multihead MACE; LSC validates its single head. "
        "Defaults to first non-PT head for MACE, as in mace.tools.script_utils.remove_pt_head",
    )
    parser.add_argument(
        "--num-spline-points",
        type=int,
        default=1024,
        help="Number of spline points.",
    )
    parser.add_argument(
        "--r-min",
        type=float,
        default=0.5,
        help="Minimum distance covered by the splines.",
    )
    parser.add_argument(
        "--compress-basis",
        nargs="?",
        type=float,
        const=0.5,
        metavar="MARGIN",
        help="Compress the first-layer radial basis if its error stays below MARGIN "
        "(0.5 if omitted) times the spline error. Off by default: a few percent "
        "faster, but its error is systematic and shifts the pressure.",
    )
    parser.add_argument("--output", "-o", help="Output filename.")
    args = parser.parse_args()

    import json
    from importlib.metadata import version
    from pathlib import Path

    if len(Path(args.model).suffix) == 0:
        model_name = Path(args.model).name
    else:
        model_name = Path(args.model).stem

    species = (
        args.atomic_numbers if args.chemical_symbols == [] else args.chemical_symbols
    )
    output = extract_mace_data(
        args.model,
        species,
        args.head,
        num_spline_points=args.num_spline_points,
        r_min=args.r_min,
        compression_margin=args.compress_basis,
    )
    packages = ("symmetrix", "mace-torch", "torch")
    if output.get("model_type") == "LocalSplitCharges":
        packages += ("mace-scf", "graph-longrange", "e3nn")
    output["extracted_with"] = {p: version(p) for p in packages}

    ### ----- WRITE JSON -----

    if args.output is None:
        args.output = (
            model_name + "-" + "-".join(str(a) for a in sorted(species)) + ".json"
        )
    if output.get("export_status") == "architecture_only":
        print(
            "LSC ARCHITECTURE ONLY: learned layer parameters and radial splines "
            "are not exported yet; this file is not runnable."
        )
    if output.get("export_status") == "parameters_and_splines":
        print(
            "LSC PARAMETERS AND SPLINES: neural and electrostatic energy parameters "
            "are exported for native finite-cluster/fully-periodic energy, force and stress evaluation. "
        )
    print("WRITING JSON TO", args.output)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()
