"""
Transform adversarial robustness experiment files to use common attack indices
(intersection of correctly classified samples across ALL models).
"""

import re
from pathlib import Path

BASE = Path("/Users/aysecoskuner/Desktop/tabpfn-adversarial-robustness")

OLD_FUNC = '''def get_stratified_attack_indices(y_test, y_pred, n_samples, random_state=42):
    """Select n_samples indices stratified by class from correctly classified samples."""
    correct_indices = np.where(y_pred == y_test)[0]
    if len(correct_indices) <= n_samples:
        return correct_indices
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=n_samples, random_state=random_state)
        _, sel = next(sss.split(correct_indices.reshape(-1, 1), y_test[correct_indices]))
        return correct_indices[sel]
    except ValueError:
        rng = np.random.RandomState(random_state)
        sel = rng.choice(len(correct_indices), n_samples, replace=False)
        return correct_indices[sel]'''

NEW_FUNC = '''def get_common_attack_indices(y_test, all_preds, n_samples, random_state=42):
    """Select n_samples from samples correctly classified by ALL models, stratified by class."""
    correct_sets = [set(np.where(preds == y_test)[0]) for preds in all_preds.values()]
    common_correct = np.array(sorted(set.intersection(*correct_sets)))
    if len(common_correct) <= n_samples:
        return common_correct
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=n_samples, random_state=random_state)
        _, sel = next(sss.split(common_correct.reshape(-1, 1), y_test[common_correct]))
        return common_correct[sel]
    except ValueError:
        rng = np.random.RandomState(random_state)
        sel = rng.choice(len(common_correct), n_samples, replace=False)
        return common_correct[sel]'''


# ===========================================================================
# CATEGORY A: Real dataset boundary + NES files
# Pattern: flat structure, models dict at top level, for model_name, model loop
# ===========================================================================

def transform_category_a(filepath):
    """Transform Category A files: real dataset flat structure."""
    content = Path(filepath).read_text()

    # 1. Replace old function with new function
    content = content.replace(OLD_FUNC, NEW_FUNC)

    # 2. Insert pre-training block BEFORE the main for loop
    # Find the line "for model_name, model in models.items():"
    # Insert the pre-training block before it
    pretrain_block = (
        "# Pre-train all models and collect predictions for common indices\n"
        "print(\"\\nPre-training all models...\")\n"
        "all_preds = {}\n"
        "for _name, _model in models.items():\n"
        "    _model.fit(X_train, y_train)\n"
        "    all_preds[_name] = _model.predict(X_test)\n"
        "    print(f\"  {_name} trained\")\n"
        "\n"
        "attack_indices = get_common_attack_indices(y_test, all_preds, n_samples)\n"
        "\n"
    )

    # Insert before "for model_name, model in models.items():"
    content = content.replace(
        "for model_name, model in models.items():\n",
        pretrain_block + "for model_name, model in models.items():\n",
        1  # only first occurrence
    )

    # 3. Remove "    model.fit(X_train, y_train)\n    \n" or "    model.fit(X_train, y_train)\n"
    # inside the loop (4-space indent)
    content = re.sub(r'    model\.fit\(X_train, y_train\)\n    \n', '    \n', content)
    content = re.sub(r'    model\.fit\(X_train, y_train\)\n', '', content)

    # 4. Replace "    y_pred = model.predict(X_test)\n    clean_acc = ...\n    attack_indices = get_stratified_attack_indices(...)"
    # with "    y_pred = all_preds[model_name]\n    clean_acc = ..."
    # The attack_indices line to remove may vary; remove lines matching the pattern
    content = re.sub(
        r'    y_pred = model\.predict\(X_test\)\n'
        r'    clean_acc = np\.mean\(y_pred == y_test\)\n'
        r'    attack_indices = get_stratified_attack_indices\(y_test, y_pred, n_samples\)\n',
        '    y_pred = all_preds[model_name]\n'
        '    clean_acc = np.mean(y_pred == y_test)\n',
        content
    )

    # diabetes/run_boundary.py has model.fit before y_pred without blank line,
    # and the y_pred line is just "    y_pred = model.predict(X_test)\n"
    # Let's also handle "    y_pred = model.predict(X_test)\n    clean_acc = np.mean(y_pred == y_test)\n    attack_indices"
    # (already handled by regex above)

    Path(filepath).write_text(content)
    print(f"  Transformed: {filepath}")


# ===========================================================================
# CATEGORY B: Synthetic files - outer loop with inner models dict
# ===========================================================================

def transform_category_b(filepath, outer_var, n_attacks_var='n_test_attacks'):
    """
    Transform Category B files: synthetic with outer loop.
    outer_var: the loop variable name (e.g. 'n_features', 'noise', 'ratio', 'cat_ratio')
    n_attacks_var: variable name for attack count ('n_test_attacks')
    """
    content = Path(filepath).read_text()

    # 1. Replace old function with new function
    content = content.replace(OLD_FUNC, NEW_FUNC)

    # 2. Find the models dict inside the outer loop, and insert pre-training block after it
    # The models dict ends with a line like:
    #     }
    # followed by
    #     \n    for model_name, model in models.items():
    # We need to insert before "    for model_name, model in models.items():"
    # but AFTER the models dict is created.

    pretrain_block_b = (
        "\n"
        "    # Pre-train all models and collect predictions for common indices\n"
        "    all_preds = {}\n"
        "    for _name, _model in models.items():\n"
        "        _model.fit(X_train, y_train)\n"
        "        all_preds[_name] = _model.predict(X_test)\n"
        "\n"
        f"    attack_indices = get_common_attack_indices(y_test, all_preds, {n_attacks_var})\n"
    )

    # Insert before "    for model_name, model in models.items():"
    content = content.replace(
        "    for model_name, model in models.items():\n",
        pretrain_block_b + "\n    for model_name, model in models.items():\n",
        1
    )

    # 3. Remove model.fit inside the inner loop (8-space indent)
    content = re.sub(r'        model\.fit\(X_train, y_train\)\n', '', content)

    # 4. Replace inner loop's y_pred + clean_acc + attack_indices lines
    # Pattern with "# Train" comment (run_noise_boundary.py, run_scaling_boundary.py, etc.)
    # Note: some files have "        # Train\n        model.fit(X_train, y_train)\n\n"
    # but we've already removed model.fit above.
    # Handle: "        y_pred = model.predict(X_test)\n        clean_acc = ...\n        attack_indices = ..."
    content = re.sub(
        r'        y_pred = model\.predict\(X_test\)\n'
        r'        clean_acc = np\.mean\(y_pred == y_test\)\n'
        r'        attack_indices = get_stratified_attack_indices\(y_test, y_pred, '
        + re.escape(n_attacks_var) +
        r'\)\n',
        '        y_pred = all_preds[model_name]\n'
        '        clean_acc = np.mean(y_pred == y_test)\n',
        content
    )

    Path(filepath).write_text(content)
    print(f"  Transformed: {filepath}")


# ===========================================================================
# CATEGORY C: Transfer attack files
# ===========================================================================

def transform_category_c_iris_breast_cancer_diabetes_heart(filepath):
    """
    Transform Category C files (iris/run_transfer.py, breast_cancer/run_transfer.py,
    diabetes/run_transfer.py, heart/run_transfer.py).
    These loop over models.items() for source, and use get_stratified_attack_indices per source.
    """
    content = Path(filepath).read_text()

    # 1. Replace old function with new function
    content = content.replace(OLD_FUNC, NEW_FUNC)

    # 2. After the training loop (the block that trains all models and prints accuracy),
    # add a pre-computation block before the adversarial example generation loop.
    #
    # The pattern to find is:
    #   print(f"\n[2/4] Generating adversarial examples...")
    # Insert before this line:
    #   # Pre-compute common attack indices
    #   all_preds = {name: model.predict(X_test) for name, model in models.items()}
    #   attack_indices = get_common_attack_indices(y_test, all_preds, n_samples)
    #
    common_block = (
        "# Pre-compute common attack indices (samples correct for ALL models)\n"
        "all_preds_for_common = {name: model.predict(X_test) for name, model in models.items()}\n"
        "attack_indices = get_common_attack_indices(y_test, all_preds_for_common, n_samples)\n"
        "\n"
    )

    content = content.replace(
        'print(f"\\n[2/4] Generating adversarial examples...")\n',
        common_block + 'print(f"\\n[2/4] Generating adversarial examples...")\n'
    )

    # 3. In the source loop, remove per-source attack_indices computation:
    #   source_y_pred = source_model.predict(X_test)
    #   attack_indices = get_stratified_attack_indices(y_test, source_y_pred, n_samples)
    content = re.sub(
        r'    source_y_pred = source_model\.predict\(X_test\)\n'
        r'    attack_indices = get_stratified_attack_indices\(y_test, source_y_pred, n_samples\)\n',
        '',
        content
    )

    Path(filepath).write_text(content)
    print(f"  Transformed: {filepath}")


def transform_category_c_wine(filepath):
    """
    Transform wine/run_transfer.py which has a different structure:
    it uses transfer_configs list and per-config source_y_pred + attack_indices.
    """
    content = Path(filepath).read_text()

    # 1. Replace old function with new function
    content = content.replace(OLD_FUNC, NEW_FUNC)

    # 2. After all models are trained (after "print(f"  LightGBM: ...")" line),
    # add common attack indices computation before the transfer_configs loop.
    # Find the marker: "# Transfer attack configurations"
    # Insert before it.
    common_block = (
        "# Pre-compute common attack indices (samples correct for ALL models)\n"
        "all_preds_for_common = {\n"
        "    'TabPFN': tabpfn.predict(X_test),\n"
        "    'XGBoost': xgboost.predict(X_test),\n"
        "    'LightGBM': lightgbm.predict(X_test)\n"
        "}\n"
        "attack_indices = get_common_attack_indices(y_test, all_preds_for_common, n_samples)\n"
        "\n"
    )

    content = content.replace(
        "# Transfer attack configurations\n",
        common_block + "# Transfer attack configurations\n"
    )

    # 3. Remove per-config source_y_pred + attack_indices computation inside transfer_configs loop
    content = re.sub(
        r'    source_y_pred = source_model\.predict\(X_test\)\n'
        r'    attack_indices = get_stratified_attack_indices\(y_test, source_y_pred, n_samples\)\n',
        '',
        content
    )

    Path(filepath).write_text(content)
    print(f"  Transformed: {filepath}")


# ===========================================================================
# RUN TRANSFORMATIONS
# ===========================================================================

print("=" * 70)
print("CATEGORY A: Real dataset boundary + NES files")
print("=" * 70)

cat_a_files = [
    "experiments/iris/run_boundary.py",
    "experiments/wine/run_boundary.py",
    "experiments/breast_cancer/run_boundary.py",
    "experiments/diabetes/run_boundary.py",
    "experiments/heart/run_boundary.py",
    "experiments/iris/run_nes.py",
    "experiments/wine/run_nes.py",
    "experiments/breast_cancer/run_nes.py",
    "experiments/diabetes/run_nes.py",
    "experiments/heart/run_nes.py",
]

for f in cat_a_files:
    transform_category_a(BASE / f)

print()
print("=" * 70)
print("CATEGORY B: Synthetic boundary + NES files")
print("=" * 70)

# (filepath, outer_var, n_attacks_var)
cat_b_files = [
    ("experiments/synthetic/run_nes.py",               "n_features",  "n_test_attacks"),
    ("experiments/synthetic/run_noise_nes.py",         "noise",       "n_test_attacks"),
    ("experiments/synthetic/run_scaling_nes.py",       "n_features",  "n_test_attacks"),
    ("experiments/synthetic/run_imbalance_nes.py",     "ratio",       "n_test_attacks"),
    ("experiments/synthetic/run_categorical_nes.py",   "cat_ratio",   "n_test_attacks"),
    ("experiments/synthetic/run_noise_boundary.py",    "noise",       "n_test_attacks"),
    ("experiments/synthetic/run_scaling_boundary.py",  "n_features",  "n_test_attacks"),
    ("experiments/synthetic/run_imbalance_boundary.py","ratio",       "n_test_attacks"),
    ("experiments/synthetic/run_categorical_boundary.py","cat_ratio", "n_test_attacks"),
]

for f, outer_var, n_attacks_var in cat_b_files:
    transform_category_b(BASE / f, outer_var, n_attacks_var)

print()
print("=" * 70)
print("CATEGORY C: Transfer attack files")
print("=" * 70)

transform_category_c_iris_breast_cancer_diabetes_heart(BASE / "experiments/iris/run_transfer.py")
transform_category_c_wine(BASE / "experiments/wine/run_transfer.py")
transform_category_c_iris_breast_cancer_diabetes_heart(BASE / "experiments/breast_cancer/run_transfer.py")
transform_category_c_iris_breast_cancer_diabetes_heart(BASE / "experiments/diabetes/run_transfer.py")
transform_category_c_iris_breast_cancer_diabetes_heart(BASE / "experiments/heart/run_transfer.py")

print()
print("=" * 70)
print("ALL TRANSFORMATIONS COMPLETE")
print("=" * 70)
