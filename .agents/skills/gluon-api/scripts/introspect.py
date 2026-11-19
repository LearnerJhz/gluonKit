#!/usr/bin/env python3
"""Runtime introspection for triton.experimental.gluon (platform-agnostic)."""

from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
import re
import sys

ROOT_PKG = "triton.experimental.gluon"
LANG_PKG = f"{ROOT_PKG}.language"
_INTERNAL_PARAMS = {"_semantic", "_builder", "_generator"}
_NOISE_NAMES = {"TYPE_CHECKING", "annotations", "builtin"}


def resolve_module(spec: str | None):
    if not spec or spec == "gl":
        return importlib.import_module(LANG_PKG)
    if spec == "gluon":
        return importlib.import_module(ROOT_PKG)
    for candidate in (f"{LANG_PKG}.{spec}", f"{ROOT_PKG}.{spec}", spec):
        try:
            return importlib.import_module(candidate)
        except ImportError:
            continue
    raise ImportError(f"Cannot resolve module '{spec}' under {ROOT_PKG}")


def resolve_symbol(base_mod, symbol: str):
    roots = [base_mod]
    for pkg_name in (LANG_PKG, ROOT_PKG):
        pkg = importlib.import_module(pkg_name)
        if pkg not in roots:
            roots.append(pkg)

    parts = symbol.split(".")
    for root in roots:
        cur = root
        ok = True
        for i, part in enumerate(parts):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            elif inspect.ismodule(cur):
                sub_name = f"{cur.__name__}.{'.'.join(parts[i:])}"
                try:
                    return importlib.import_module(sub_name)
                except ImportError:
                    ok = False
                    break
            else:
                ok = False
                break
        if ok:
            return cur
    raise AttributeError(f"Symbol '{symbol}' not found in {base_mod.__name__}")


def _is_public_api(name: str, obj) -> bool:
    if name.startswith("_") or name in _NOISE_NAMES:
        return False
    mod = getattr(obj, "__module__", "") or ""
    if mod == "typing" or mod.startswith("triton.compiler"):
        return False
    return True


def _discover_submodules(mod) -> list[str]:
    if not hasattr(mod, "__path__"):
        return []
    prefix = mod.__name__ + "."
    found = []
    for info in pkgutil.walk_packages(mod.__path__, prefix):
        rel = info.name[len(prefix):]
        if any(part.startswith("_") for part in rel.split(".")):
            continue
        found.append(rel)
    return sorted(found)


def _clean_signature(obj) -> str:
    target = getattr(obj, "fn", obj)
    try:
        sig = inspect.signature(target)
    except (ValueError, TypeError):
        return "(...)"
    params = [p for k, p in sig.parameters.items() if k not in _INTERNAL_PARAMS]
    return str(sig.replace(parameters=params))


def _first_paragraph(obj) -> str:
    target = getattr(obj, "fn", obj)
    doc = inspect.getdoc(target) or inspect.getdoc(obj) or ""
    if not doc:
        return ""
    para = doc.strip().split("\n\n", 1)[0]
    return re.sub(r"\s+", " ", para).strip()


def cmd_list(mod) -> None:
    layouts_types, callables, direct_mods = [], [], []
    for name in dir(mod):
        obj = getattr(mod, name)
        if inspect.ismodule(obj):
            if name.startswith("_"):
                continue
            if obj.__name__.startswith(ROOT_PKG):
                direct_mods.append(name)
            continue
        if not _is_public_api(name, obj):
            continue
        if inspect.isclass(obj):
            layouts_types.append(name)
        elif callable(obj):
            callables.append(name)

    print(f"Module: {mod.__name__}")
    submods = _discover_submodules(mod)
    if submods:
        print(f"Submodules ({len(submods)}): {', '.join(submods)}")
    elif direct_mods:
        print(f"Submodules ({len(direct_mods)}): {', '.join(sorted(direct_mods))}")
    if layouts_types:
        print(f"Types/Layouts ({len(layouts_types)}): {', '.join(sorted(layouts_types))}")
    if callables:
        print(f"Callables ({len(callables)}): {', '.join(sorted(callables))}")


def cmd_inspect(mod, symbols: list[str]) -> int:
    rc = 0
    for sym in symbols:
        try:
            obj = resolve_symbol(mod, sym)
        except Exception as exc:
            print(f"[ERROR] {sym}: {exc}", file=sys.stderr)
            rc = 1
            continue

        if inspect.ismodule(obj):
            cmd_list(obj)
            print()
            continue

        sig = _clean_signature(obj)
        doc = _first_paragraph(obj)
        kind = "class" if inspect.isclass(obj) else "def"
        print(f"{kind} {sym}{sig}")
        if doc:
            print(f"  {doc}")
        if inspect.isclass(obj):
            methods = [
                m for m, v in inspect.getmembers(obj)
                if not m.startswith("_") and callable(v)
            ]
            if methods:
                print(f"  methods: {', '.join(methods)}")
        print()
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect live Gluon APIs in the current Python environment."
    )
    parser.add_argument("symbols", nargs="*", help="Symbol names to inspect (e.g. max sum exp load)")
    parser.add_argument("-m", "--module", default=None, help="Target module (default: gl)")
    parser.add_argument("--list", nargs="?", const="__DEFAULT__", default=None, metavar="MODULE",
                        help="List public APIs and backend submodules of MODULE (default: gl)")
    args = parser.parse_args(argv)

    if args.list is not None:
        target = args.module if args.list == "__DEFAULT__" else args.list
        cmd_list(resolve_module(target))
        return 0
    if not args.symbols:
        parser.print_help()
        return 2
    return cmd_inspect(resolve_module(args.module), args.symbols)


if __name__ == "__main__":
    raise SystemExit(main())
