import argparse
import os
from typing import Dict, List

from modules.custom_dataset_builder import (
    build_dataset,
    load_jsonl,
    save_raw_dataset,
)


def _parse_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_variant_specs(specs: List[str]) -> Dict[str, List[str]]:
    # spec format: "<variant_name>:field_a,field_b"
    parsed: Dict[str, List[str]] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Invalid --variant spec: {spec}")
        name, fields = spec.split(":", 1)
        name = name.strip()
        candidates = _parse_csv(fields)
        if not name or not candidates:
            raise ValueError(f"Invalid --variant spec: {spec}")
        parsed[name] = candidates
    return parsed


def run(args):
    rows = load_jsonl(args.input_jsonl)
    original_field_candidates = _parse_csv(args.original_fields)
    variant_specs = _parse_variant_specs(args.variant)

    for variant_name, sampled_field_candidates in variant_specs.items():
        data = build_dataset(
            rows,
            original_field_candidates=original_field_candidates,
            sampled_field_candidates=sampled_field_candidates,
        )

        output_prefix = os.path.join(args.output_dir, f"{args.prefix}_{variant_name}")
        save_raw_dataset(
            output_prefix,
            args={
                "input_jsonl": args.input_jsonl,
                "variant": variant_name,
                "original_field_candidates": original_field_candidates,
                "sampled_field_candidates": sampled_field_candidates,
                "n_rows_input": len(rows),
                "n_rows_output": len(data["original"]),
            },
            data=data,
        )
        print(f"{variant_name} rows: {len(data['original'])}")
        print(f"wrote: {output_prefix}.raw_data.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="/app/datasets/visual_understanding/sample_001.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./exp_custom/data",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="custom",
    )
    parser.add_argument(
        "--original_fields",
        type=str,
        default="abstract_hwt",
        help="Comma-separated field candidates for original/human text.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[
            "onlyllm:abstract_onlyllm,abstract_only_llm",
            "rag:abstract_rag",
        ],
        help='Repeatable. Format: "<variant_name>:field1,field2".',
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
