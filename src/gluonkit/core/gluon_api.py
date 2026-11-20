"""Runtime introspection of triton.experimental.gluon (platform-agnostic, no GPU required)."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re

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


def resolve_symbol(base_mod, symbol: str, *, fallback: bool = True):
    """Resolve `symbol` and return (object, module it was found in).

    With fallback=False only `base_mod` is searched (used for an explicit -m);
    otherwise gl and then gluon are tried after `base_mod`.
    """
    roots = [base_mod]
    for pkg_name in (LANG_PKG, ROOT_PKG) if fallback else ():
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
                    return importlib.import_module(sub_name), root
                except ImportError:
                    ok = False
                    break
            else:
                ok = False
                break
        if ok:
            return cur, root
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


def list_module(mod) -> str:
    """Summarise the public submodules, types and callables of `mod` as text."""
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

    lines = [f"Module: {mod.__name__}"]
    submods = _discover_submodules(mod)
    if submods:
        lines.append(f"Submodules ({len(submods)}): {', '.join(submods)}")
    elif direct_mods:
        lines.append(f"Submodules ({len(direct_mods)}): {', '.join(sorted(direct_mods))}")
    if layouts_types:
        lines.append(f"Types/Layouts ({len(layouts_types)}): {', '.join(sorted(layouts_types))}")
    if callables:
        lines.append(f"Callables ({len(callables)}): {', '.join(sorted(callables))}")
    return "\n".join(lines)


def show_symbols(mod, symbols: list[str], *, fallback: bool = True) -> tuple[str, list[str]]:
    """Describe each symbol as text. Returns (stdout text, error lines for unresolved symbols).

    fallback=False restricts lookup to `mod` (explicit -m); otherwise symbols found only in the
    gluon root package are annotated with where they resolved.
    """
    blocks: list[str] = []
    errors: list[str] = []
    for sym in symbols:
        try:
            obj, found_in = resolve_symbol(mod, sym, fallback=fallback)
        except Exception as exc:
            errors.append(f"[ERROR] {sym}: {exc}")
            continue

        if inspect.ismodule(obj):
            blocks.append(list_module(obj))
            continue

        kind = "class" if inspect.isclass(obj) else "def"
        lines = [f"{kind} {sym}{_clean_signature(obj)}"]
        doc = _first_paragraph(obj)
        if doc:
            lines.append(f"  {doc}")
        if found_in is not mod and found_in.__name__ == ROOT_PKG:
            lines.append(f"  (resolved in {ROOT_PKG})")
        if inspect.isclass(obj):
            methods = [
                m for m, v in inspect.getmembers(obj)
                if not m.startswith("_") and callable(v)
            ]
            if methods:
                lines.append(f"  methods: {', '.join(methods)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks), errors
