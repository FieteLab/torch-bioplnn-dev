#!/usr/bin/env python3
"""
Script to programmatically update the __init__.py file by inspecting each module
and adding non-underscore functions and classes to __all__.
"""

import importlib
import inspect
import sys
from pathlib import Path


def get_public_symbols(module_name):
    """
    Get all public symbols (those not starting with _) from a module.

    Args:
        module_name (str): The name of the module to inspect.

    Returns:
        list[str]: A list of public symbol names.
    """
    try:
        module = importlib.import_module(module_name)
        public_symbols = []

        for name, obj in inspect.getmembers(module):
            # Skip objects that start with underscore
            if name.startswith("_"):
                continue

            # Only include functions, classes, and module-level constants
            if (
                inspect.isfunction(obj)
                or inspect.isclass(obj)
                or not inspect.ismodule(obj)
                and not inspect.isbuiltin(obj)
            ):
                public_symbols.append(name)

        return public_symbols
    except ImportError as e:
        print(f"Error importing {module_name}: {e}")
        return []


def update_init_file(package_path, package_name="bioplnn.utils"):
    """
    Update the __init__.py file with explicit imports and __all__.

    Args:
        package_path (str): Path to the package directory.
        package_name (str): Fully qualified package name.
    """
    package_dir = Path(package_path)
    init_file = package_dir / "__init__.py"

    # Get all Python files in the directory
    modules = []
    for item in package_dir.iterdir():
        if (
            item.is_file()
            and item.suffix == ".py"
            and item.name != "__init__.py"
            and not item.name.startswith("_")
        ):
            modules.append(item.stem)

    # Import each module and get its public symbols
    imports = {}
    all_symbols = []

    for module in sorted(modules):
        module_path = f"{package_name}.{module}"
        public_symbols = get_public_symbols(module_path)

        if public_symbols:
            imports[module] = sorted(public_symbols)
            all_symbols.extend(public_symbols)

    # Generate the new __init__.py content
    import_lines = []
    for module, symbols in imports.items():
        if len(symbols) == 1:
            import_lines.append(f"from .{module} import {symbols[0]}")
        else:
            import_lines.append(f"from .{module} import (")
            for symbol in symbols:
                import_lines.append(f"    {symbol},")
            import_lines.append(")")
        import_lines.append("")  # Add a blank line after each import block

    all_list = ["__all__ = ["]
    for symbol in sorted(all_symbols):
        all_list.append(f'    "{symbol}",')
    all_list.append("]")

    # Write the new content
    with open(init_file, "w") as f:
        f.write("\n".join(import_lines))
        f.write("\n")
        f.write("\n".join(all_list))
        f.write("\n")

    print(
        f"Updated {init_file} with {len(all_symbols)} public symbols from {len(modules)} modules."
    )


if __name__ == "__main__":
    # Get the directory of this script
    script_dir = Path(__file__).parent

    # Add the src directory to sys.path to make imports work
    sys.path.insert(0, str(script_dir.parent.parent.parent))

    # Update the __init__.py file in the same directory
    update_init_file(script_dir)
