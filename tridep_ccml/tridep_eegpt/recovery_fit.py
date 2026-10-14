"""Resume seven uncompleted fits by transforming the frozen original entry.

Only entry authorization, reuse of the 23 complete fits, and execution
provenance are added. The model construction, optimizer and entire epoch
loop are retained as identical AST nodes. Importing or inspecting this module
does not construct a model, perform a forward, or create an optimizer.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path

from . import train_eval as T


def identities():
    """Original seed/fold/paired-condition order, without any computation."""
    return [(condition, seed, fold) for seed in T.SEEDS for fold in range(1, 6)
            for condition in T.CONDITIONS]


def completed_identities():
    return identities()[:23]


def pending_identities():
    return identities()[23:]


def _dump(node):
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _hash(node):
    return hashlib.sha256(_dump(node).encode("utf-8")).hexdigest()


def _statement(source):
    nodes = ast.parse(source).body
    T.require(len(nodes) == 1, "Recovery AST contract requires one statement")
    return nodes[0]


def _expect(node, source, message):
    T.require(_dump(node) == _dump(_statement(source)), message)


def _epochs(node):
    result = [n for n in ast.walk(node) if isinstance(n, ast.For)
              and isinstance(n.target, ast.Name) and n.target.id == "epoch"]
    T.require(len(result) == 1, "Recovery requires the single original epoch loop")
    return result[0]


def _setup_nodes(node):
    """Model/RNG/data/optimizer nodes preceding training, in source order."""
    names = {"model", "indices", "loader", "optimizer"}
    selected = []
    for candidate in ast.walk(node):
        if isinstance(candidate, ast.Expr) and isinstance(candidate.value, ast.Call):
            call = candidate.value
            if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) \
                    and call.func.value.id == "C" and call.func.attr == "runtime":
                selected.append(candidate)
        elif isinstance(candidate, ast.Assign):
            target = candidate.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                selected.append(candidate)
            elif isinstance(target, ast.Tuple) and any(isinstance(n, ast.Name) and n.id == "indices"
                                                      for n in target.elts):
                selected.append(candidate)
    # The original fit has exactly one runtime/model/dataset/loader/optimizer.
    T.require(len(selected) == 5, "Original fitting setup AST differs")
    return ast.Module(body=selected, type_ignores=[])


def build_fit_ast(source_text=None):
    """Return original and minimally transformed AST; execute neither."""
    if source_text is None:
        source_text = Path(T.__file__).read_text(encoding="utf-8")
    functions = [n for n in ast.parse(source_text).body
                 if isinstance(n, ast.FunctionDef) and n.name == "fit_all"]
    T.require(len(functions) == 1, "Frozen source must contain one fit_all")
    original = functions[0]
    T.require(not original.decorator_list and not original.args.args
              and len(original.body) == 12, "Frozen fit signature/body differs")
    generated = copy.deepcopy(original)
    generated.name = "_fit_remaining_impl"

    _expect(generated.body[0], '_entry_guard("fit_all")', "Original fitting entry guard differs")
    generated.body[0] = _statement("_recovery.guard_training()")
    precheck = '''for condition in CONDITIONS:
    for seed in SEEDS:
        for fold in range(1, 6):
            require(not _folder(condition, seed, fold).exists(),
                    "Existing complete/partial fit requires explicit review; no automatic retry")'''
    _expect(generated.body[3], precheck, "Original fit precheck differs")
    del generated.body[3]
    _expect(generated.body[6], "completed = []", "Original fit ledger initialization differs")
    generated.body[6] = _statement("completed = _preload_completed(_recovery)")
    generated.body.insert(7, _statement("_completed_by_id = {(m['condition'], m['seed'], m['fold']): m for m in completed}"))

    condition_loops = [n for n in ast.walk(generated) if isinstance(n, ast.For)
                       and isinstance(n.target, ast.Name) and n.target.id == "condition"]
    T.require(len(condition_loops) == 1, "Expected one paired condition fitting loop")
    body = condition_loops[0].body
    _expect(body[0], "folder = _folder(condition, seed, fold)", "Original fit folder construction differs")
    skip = '''if (condition, seed, fold) in _completed_by_id:
    pair = _reuse_completed_pair(_completed_by_id[condition, seed, fold], folder, pair)
    continue'''
    body.insert(1, _statement(skip))
    metas = [i for i, node in enumerate(body) if isinstance(node, ast.Assign)
             and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "meta"]
    T.require(len(metas) == 1, "Original final metadata assignment differs")
    body.insert(metas[0] + 1, _statement('''meta.update(
    execution_registration=C.desc(C.OUT / "recovery/registration.json"),
    execution_source=C.desc(Path(_recovery_fit_source)))'''))
    ast.fix_missing_locations(generated)
    T.require(_hash(_epochs(original)) == _hash(_epochs(generated)),
              "Recovery must preserve the complete original epoch loop")
    T.require(_hash(_setup_nodes(original)) == _hash(_setup_nodes(generated)),
              "Recovery must preserve original model/RNG/data/optimizer setup")
    return original, generated


def ast_evidence():
    """Pure proof of the unchanged numerical training algorithm."""
    original, generated = build_fit_ast()
    original_epoch, generated_epoch = _hash(_epochs(original)), _hash(_epochs(generated))
    original_setup, generated_setup = _hash(_setup_nodes(original)), _hash(_setup_nodes(generated))
    return {
        "frozen_training_source": T.C.desc(Path(T.__file__)),
        "recovery_training_source": T.C.desc(Path(__file__)),
        "original_fit_ast_sha256": _hash(original),
        "generated_fit_ast_sha256": _hash(generated),
        "epoch_ast_sha256": {"original": original_epoch, "generated": generated_epoch,
                             "identical": original_epoch == generated_epoch},
        "model_rng_dataset_loader_optimizer_ast_sha256": {
            "original": original_setup, "generated": generated_setup,
            "identical": original_setup == generated_setup},
        "changes": ["recovery entry guard", "reuse only the bound 23 complete fits",
                    "preserve paired initialization/order while skipping completed fits",
                    "add execution provenance to the seven new fit metadata records"],
        "pending_fit_identities": [list(key) for key in pending_identities()],
        "new_formal_fits": 7, "new_optimizer_updates": 840,
        "new_gradient_passes": 840, "new_model_execution_during_proof": 0,
    }


def _preload_completed(recovery):
    values = recovery.original_completed_metadata()
    T.require(isinstance(values, list) and len(values) == 23,
              "Recovery must reuse exactly 23 original complete fits")
    actual = [(m["condition"], m["seed"], m["fold"]) for m in values]
    T.require(actual == completed_identities(), "Recovery completed identity/order differs")
    T.require(all(m["updates"] == m["gradient_passes"] == 120 and m["epochs"] == 60
                  and m["training"] == T.TRAINING for m in values),
              "Recovery complete fit training contract differs")
    # Existing pending folders can never be silently replaced or retried. The
    # previous interrupted folder must first be archived under recovery.
    T.require(all(not T._folder(*key).exists() for key in pending_identities()),
              "Pending complete/partial fit exists; explicit recovery review required")
    return list(values)


def _reuse_completed_pair(meta, folder, pair):
    T.require(folder.is_dir() and T.C.read(folder / "metadata.json") == meta,
              "A preserved original completed fit changed during recovery")
    current = (meta["initial_state_sha256"], meta["batch_order_sha256"])
    T.require(pair is None or pair == current,
              "Preserved original conditions are not paired in initialization and order")
    return current


def fit_remaining():
    """Run only the registered seven fits; the caller owns its action marker."""
    from . import recovery
    original, generated = build_fit_ast()
    namespace = dict(T.__dict__)
    namespace.update(_recovery=recovery, _preload_completed=_preload_completed,
                     _reuse_completed_pair=_reuse_completed_pair,
                     _recovery_fit_source=str(Path(__file__)))
    program = ast.Module(body=[generated], type_ignores=[])
    exec(compile(program, str(Path(T.__file__)), "exec"), namespace)
    return namespace[generated.name]()

