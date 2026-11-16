"""
Unit test to verify that no plotting, table, or experiment script contains
hard-coded metric numbers, synthetic curves, or hard-coded numpy arrays.

Non-negotiable integrity rule:
"Plots are drawn from predictions, never from hand-written arrays.
Before every commit, grep plotting code for hard-coded numeric arrays;
there must be none."
"""

import ast
import os
import unittest


class TestNoHardcodedNumbers(unittest.TestCase):
    """Ensure that experimental and plotting scripts do not contain hard-coded metric arrays."""

    FORBIDDEN_VAR_NAMES = {
        'fpr', 'tpr', 'prec', 'rec', 'auc', 'fpr_ours', 'tpr_ours',
        'fpr_mc', 'tpr_mc', 'fpr_vh', 'tpr_vh', 'fpr_as', 'tpr_as',
        'fpr_b2', 'tpr_b2', 'fpr_b5', 'tpr_b5', 'fpr_sc', 'tpr_sc',
        'rec_ours', 'prec_ours', 'rec_mc', 'prec_mc', 'rec_vh', 'prec_vh',
        'rec_as', 'prec_as', 'rec_b5', 'prec_b5',
        'harmoniq_acc', 'mc_acc', 'prompt_acc'
    }

    def setUp(self):
        self.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.experiments_dir = os.path.join(self.repo_root, "experiments")

    def test_generate_paper_artifacts_superseded(self):
        """Verify that the old hardcoded generator is moved to archive and not present in experiments/."""
        old_script = os.path.join(self.experiments_dir, "generate_paper_artifacts.py")
        self.assertFalse(
            os.path.exists(old_script),
            f"Found superseded script {old_script}! It must be archived in archive/ and removed from experiments/."
        )

    def test_no_hardcoded_arrays_in_experiments(self):
        """Parse AST of every .py file in experiments/ and check for hardcoded metric arrays."""
        py_files = [
            os.path.join(self.experiments_dir, f)
            for f in os.listdir(self.experiments_dir)
            if f.endswith('.py')
        ]
        self.assertTrue(len(py_files) > 0, "No Python files found in experiments/")

        violations = []

        for fpath in py_files:
            fname = os.path.basename(fpath)
            with open(fpath, 'r', encoding='utf-8') as f:
                source = f.read()

            try:
                tree = ast.parse(source, filename=fname)
            except SyntaxError as e:
                self.fail(f"Syntax error parsing {fname}: {e}")

            for node in ast.walk(tree):
                # 1. Check variable assignment names
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        target_id = getattr(target, 'id', None)
                        if target_id and target_id.lower() in self.FORBIDDEN_VAR_NAMES:
                            # Check if value is a list, tuple, or call to np.array with literals
                            if isinstance(node.value, (ast.List, ast.Tuple)):
                                violations.append(
                                    f"{fname}:{node.lineno}: Variable '{target_id}' assigned hard-coded list/tuple"
                                )
                            elif isinstance(node.value, ast.Call):
                                func_name = getattr(node.value.func, 'id', getattr(node.value.func, 'attr', ''))
                                if func_name == 'array':
                                    violations.append(
                                        f"{fname}:{node.lineno}: Variable '{target_id}' assigned hard-coded np.array"
                                    )

                # 2. Check for suspicious literal float lists (e.g., >3 floats between 0 and 1)
                # except in legitimate places like figure dimensions or RGB/hex definitions
                if isinstance(node, ast.Call):
                    func_name = getattr(node.func, 'id', getattr(node.func, 'attr', ''))
                    if func_name == 'array' and node.args:
                        arg0 = node.args[0]
                        if isinstance(arg0, (ast.List, ast.Tuple)):
                            elements = arg0.elts
                            if len(elements) > 3 and all(
                                isinstance(e, ast.Constant) and isinstance(e.value, (int, float))
                                for e in elements
                            ):
                                violations.append(
                                    f"{fname}:{node.lineno}: Suspicious hardcoded np.array with {len(elements)} numeric literals"
                                )

        self.assertEqual(
            violations, [],
            f"Hard-coded metric arrays detected in experiments/:\n" + "\n".join(violations)
        )

    def test_make_figures_script_exists_and_uses_data(self):
        """Verify experiments/make_figures.py exists and references data files rather than constants."""
        make_figs_path = os.path.join(self.experiments_dir, "make_figures.py")
        self.assertTrue(
            os.path.exists(make_figs_path),
            f"Expected {make_figs_path} to exist as the dynamic figure generator."
        )

        with open(make_figs_path, 'r', encoding='utf-8') as f:
            code = f.read()

        # Must reference dynamic loading or sklearn metrics
        self.assertIn("predictions", code.lower(), "make_figures.py must process predictions data")
        self.assertIn("roc_curve", code, "make_figures.py must dynamically compute ROC curves using roc_curve")


if __name__ == '__main__':
    unittest.main()
