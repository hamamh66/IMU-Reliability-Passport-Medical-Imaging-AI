#!/usr/bin/env python3
"""Derive the model-scoped claim-gate records (schema 1.1.1) from the run's own evidence.

Schema 1.1.0, emitted by software version 0.2, binds each claim gate to its rules,
prerequisites and threshold provenance, but not to a named model.  Once more than one
image model is evaluated, a reliability gate is ambiguous unless the model it was computed
from is identified: gate C5 was computed from the hierarchical frozen-feature model, on
whose temperature-scaled probabilities the conformal quantiles and the selective policy
were built, while the fine-tuned model has its own coverage.

Schema 1.1.1 adds one optional field, `model_scope`, to a claim-gate record, and this
script re-emits the gate records with it.  It recomputes nothing: every value is read from
the evidence files of the run and the states are reproduced by the same propagation rule
as the notebook.  Software version 0.3 emits these records directly, together with a
canonical digest of each model's test-set probability matrix.

Inputs (all from the run):  passport.json, configuration.json,
                            CXR_reference_model_evidence.json, CXR_finetuned_model_evidence.json
Outputs:                    passport_1.1.1.json, passport.schema-1.1.1.json, claim_gates_1.1.1.csv

Usage:  python regenerate_claim_gates.py [directory]      (default: this file's directory)
"""
import csv
import json
import sys
from pathlib import Path

SUPPORTED = "supported_in_evaluated_setting"


def propagate(gates):
    """Effective state after prerequisite propagation (Eq. 2 of the manuscript)."""
    by_id = {g["claim_id"]: g for g in gates}

    def effective(cid):
        g = by_id[cid]
        pre = [effective(d) for d in g["depends_on"]]
        if g["state"] == SUPPORTED and any(p != SUPPORTED for p in pre):
            return "conditional"
        return g["state"]

    for g in gates:
        g["effective_state"] = effective(g["claim_id"])
    return gates


def main(root: Path) -> int:
    passport = json.loads((root / "passport.json").read_text())
    schema = json.loads((root / "passport.schema.json").read_text())
    config = json.loads((root / "configuration.json").read_text())
    reference = json.loads((root / "CXR_reference_model_evidence.json").read_text())
    finetuned = json.loads((root / "CXR_finetuned_model_evidence.json").read_text())

    if passport["schema_version"] != "1.1.0":
        print(f"expected a schema 1.1.0 passport, found {passport['schema_version']}")
        return 1

    # --- schema 1.1.1: model_scope is optional, so 1.1.0 records remain valid -------------
    gate_items = schema["properties"]["claim_gates"]["items"]
    gate_items["properties"]["model_scope"] = {
        "type": ["object", "null"],
        "description": "The model a model-dependent gate was computed from. Null for gates "
                       "that do not depend on a fitted model.",
        "required": ["model", "determined_by", "prediction_digest"],
        "properties": {
            "model": {"type": "string"},
            "determined_by": {"type": "object"},
            "prediction_digest": {
                "type": ["string", "null"],
                "description": "Canonical digest of the model's test-set probability matrix. "
                               "Emitted from software version 0.3; null for 0.2 runs.",
            },
        },
    }
    schema["$id"] = "https://example.org/imaging-passport.schema-1.1.1.json"

    # --- model identities, read from the run ---------------------------------------------
    deterministic = {
        "software_version": passport["execution_identity"]["software_version"],
        "configuration_hash": passport["execution_identity"]["configuration_hash"],
        "seed": config["seed"],
        "encoder": config["encoder"],
        "note": "Frozen-feature models are deterministic given the configuration hash and seed.",
    }
    hierarchical_scope = {
        "model": "hierarchical (temperature-scaled normal-vs-abnormal and pneumonia-vs-tuberculosis "
                 "logistic regressions on frozen %s features)" % config["encoder"],
        "determined_by": dict(deterministic, temperatures=reference["temperatures"]),
        "prediction_digest": None,
    }
    direct_scope = {
        "model": "direct (%s on frozen %s features)" % (reference["selected_model"], config["encoder"]),
        "determined_by": dict(deterministic),
        "prediction_digest": None,
    }
    finetuned_scope = {
        "model": "fine-tuned %s, all layers trainable" % config["encoder"],
        "determined_by": {
            **deterministic,
            "hyperparameters": finetuned["hyperparameters"],
            "epochs_trained": finetuned["epochs_trained"],
            "best_epoch": finetuned["best_epoch"],
            "temperature": finetuned["temperature"],
            "checkpoint": "Models/finetuned_best.pt",
        },
        "prediction_digest": None,
    }

    gates = passport["claim_gates"]
    for g in gates:
        g.setdefault("model_scope", None)

    by_id = {g["claim_id"]: g for g in gates}

    # C4 is scoped to the best-performing image model: a shortcut claim must be tested
    # against the strongest available representation.
    by_id["C4"]["model_scope"] = {
        "model": "best image model: " + finetuned_scope["model"],
        "determined_by": finetuned_scope["determined_by"],
        "prediction_digest": None,
    }
    by_id["C4"]["model_scope"]["compared_against"] = direct_scope["model"]

    # C5 is scoped to the hierarchical model: the conformal quantiles and the selective
    # policy were constructed on its temperature-scaled probabilities.
    by_id["C5"]["model_scope"] = hierarchical_scope
    by_id["C5"]["claim"] += " (hierarchical model)"
    by_id["C5"]["rationale"] = (
        "Coverage is evaluated only within the reconstructed CXR setting, and only for the "
        "hierarchical model, on which the conformal sets and the selective policy were built."
    )

    # C5b: the same rule applied to the fine-tuned model, recorded separately rather than
    # merged into C5, so that no gate mixes model scopes.
    target = 1 - config["conformal_alpha"]
    coverage = finetuned["conformal"]["marginal_coverage"]
    c5b = {
        "claim_id": "C5b",
        "claim": "Conformal marginal coverage meets the internal tolerance (fine-tuned model)",
        "state": SUPPORTED if coverage >= target - config["conformal_tolerance"] else "blocked",
        "rule_id": "R-CONFORMAL-001",
        "evidence": {
            "observed": coverage,
            "target": target,
            "tolerance": config["conformal_tolerance"],
            "empty_sets": finetuned["conformal"]["empty_sets"],
        },
        "rationale": "Coverage of the fine-tuned model, recorded separately from C5 so that no "
                     "gate mixes model scopes. The selective policy was not refitted for this model.",
        "depends_on": list(by_id["C5"]["depends_on"]),
        "threshold_provenance": dict(by_id["C5"]["threshold_provenance"]),
        "model_scope": finetuned_scope,
    }
    gates.insert(gates.index(by_id["C5"]) + 1, c5b)

    passport["claim_gates"] = propagate(gates)
    passport["schema_version"] = "1.1.1"

    try:
        import jsonschema
        jsonschema.validate(passport, schema)
        print("passport_1.1.1.json validates against schema 1.1.1")
    except ImportError:
        print("jsonschema not installed; skipping validation")

    (root / "passport_1.1.1.json").write_text(json.dumps(passport, indent=2))
    (root / "passport.schema-1.1.1.json").write_text(json.dumps(schema, indent=2))
    with (root / "claim_gates_1.1.1.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["claim_id", "claim", "depends_on", "local_state", "effective_state",
                         "rule_id", "model_scope", "rationale", "evidence", "threshold_provenance"])
        for g in passport["claim_gates"]:
            writer.writerow([
                g["claim_id"], g["claim"], ";".join(g["depends_on"]), g["state"], g["effective_state"],
                g["rule_id"],
                "" if g["model_scope"] is None else g["model_scope"]["model"],
                g["rationale"], json.dumps(g["evidence"]), json.dumps(g["threshold_provenance"]),
            ])
    for g in passport["claim_gates"]:
        scope = "" if g["model_scope"] is None else "  [%s]" % g["model_scope"]["model"][:48]
        print(f"{g['claim_id']:4s} {g['state']:30s} -> {g['effective_state']:30s}{scope}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent))
