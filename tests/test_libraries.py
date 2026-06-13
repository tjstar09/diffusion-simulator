"""
Auto-generated library import tests.
These tests are dynamically driven by test_config.json.
When you add new libraries to test_config.json, new tests are automatically generated.
"""
import json
import os
import sys
import importlib
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load config once at module level
_config_path = os.path.join(PROJECT_ROOT, "test_config.json")
with open(_config_path, "r") as _f:
    _CONFIG = json.load(_f)


def pytest_generate_tests(metafunc):
    """Dynamically parametrize tests based on test_config.json.

    For each library defined in the config, this generates test cases for:
      - Import availability (the library can be imported)
      - Specific function availability (key functions exist in the module)
      - Submodule availability (if defined)
    """
    if "library" in metafunc.fixturenames:
        metafunc.parametrize(
            "library",
            _CONFIG["libraries"],
            ids=[lib["name"] for lib in _CONFIG["libraries"]]
        )

    if "script_module" in metafunc.fixturenames:
        metafunc.parametrize(
            "script_module",
            _CONFIG["scripts"],
            ids=[s["module"] for s in _CONFIG["scripts"]]
        )


# ================================================================
# LIBRARY TESTS  (driven by test_config.json)
# ================================================================

class TestLibraryImports:
    """Verify all declared libraries can be imported."""

    def test_library_import(self, library):
        """Test that the library can be imported successfully."""
        importlib.import_module(library["import_name"])

    def test_library_functions_exist(self, library):
        """Test that key functions exist within the imported library."""
        if "test_functions" not in library or not library["test_functions"]:
            pytest.skip(f"No functions to test for {library['name']}")

        mod = importlib.import_module(library["import_name"])
        for func_name in library["test_functions"]:
            # Handle dotted paths like "random.normal"
            parts = func_name.split(".")
            obj = mod
            for part in parts:
                obj = getattr(obj, part, None)
                if obj is None:
                    # For pandas-style "DataFrame.groupby", check the class exists
                    if "." in func_name:
                        parent_name, child_name = func_name.split(".", 1)
                        parent = getattr(mod, parent_name, None)
                        if parent is not None:
                            obj = getattr(parent, child_name, None)
            assert obj is not None, (
                f"Function '{func_name}' not found in library '{library['name']}'"
            )

    def test_library_submodules(self, library):
        """Test that submodules (if declared) can be imported."""
        if "submodules" not in library or not library["submodules"]:
            pytest.skip(f"No submodules to test for {library['name']}")

        for submod_name in library["submodules"]:
            importlib.import_module(submod_name)


# ================================================================
# SCRIPT MODULE IMPORT TESTS
# ================================================================

class TestScriptImports:
    """Verify all declared script modules can be imported."""

    def test_script_import(self, script_module):
        """Test that the script module can be imported."""
        module_name = script_module["module"]
        # Import the module dynamically
        spec = importlib.util.spec_from_file_location(
            module_name,
            os.path.join(PROJECT_ROOT, script_module["file"])
        )
        assert spec is not None, (
            f"Could not find spec for module '{module_name}' at '{script_module['file']}'"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    def test_script_tests_defined(self, script_module):
        """Test that the config declares at least one test for each script."""
        assert "tests" in script_module, (
            f"No tests defined for script '{script_module['module']}'"
        )
        assert len(script_module["tests"]) > 0, (
            f"Empty tests list for script '{script_module['module']}'"
        )