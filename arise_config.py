"""Zero-dependency configuration shim for the ARISE experiment scripts.

WHY THIS EXISTS
---------------
Every script in the original code base selected its dataset, checkpoint and
hyper-parameters by editing module-level constants (and un-commenting one of
several stacked alternatives).  That made runs impossible to reproduce: the
result files record only a model filename, never the configuration in force.

This shim removes that coupling WITHOUT changing any behaviour.  Each constant
becomes

    NAME = cfg('NAME', <the original literal>)

so a script run with no configuration behaves exactly as before.

PRECEDENCE (highest first)
--------------------------
1. environment variable  ARISE_<NAME>          e.g. ARISE_DATASET=Kitsune
2. a preset file:        $ARISE_CONFIG         (explicit path), or
                         <repo>/configs/$ARISE_PRESET.yaml
3. the call-site default == the original hard-coded literal

Values coming from the environment are coerced to the type of the default.
"""
import ast
import json
import os

__all__ = ["cfg", "active_preset", "describe"]

_CACHE = {}
_USED = {}


def _repo_root():
    return os.path.dirname(os.path.abspath(__file__))


def _load():
    if _CACHE:
        return _CACHE["data"]
    path = os.environ.get("ARISE_CONFIG")
    if not path:
        preset = os.environ.get("ARISE_PRESET")
        if preset:
            for ext in (".yaml", ".yml", ".json"):
                p = os.path.join(_repo_root(), "configs", preset + ext)
                if os.path.isfile(p):
                    path = p
                    break
            if not path:
                raise SystemExit("ARISE_PRESET=%r: no configs/%s.{yaml,yml,json}" % (preset, preset))
    data = {}
    if path:
        if not os.path.isfile(path):
            raise SystemExit("ARISE_CONFIG=%r does not exist" % path)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        if path.endswith(".json"):
            data = json.loads(text)
        else:
            try:
                import yaml
                data = yaml.safe_load(text) or {}
            except ImportError:
                raise SystemExit("PyYAML is required to read %s" % path)
        if not isinstance(data, dict):
            raise SystemExit("%s must contain a mapping at the top level" % path)
    _CACHE["data"] = data
    _CACHE["path"] = path
    return data


def _coerce(raw, default):
    if isinstance(default, bool):
        s = str(raw).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
        raise SystemExit("cannot read %r as a boolean" % raw)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, (list, tuple, dict)):
        try:
            return json.loads(raw)
        except ValueError:
            return ast.literal_eval(raw)
    return raw


def cfg(name, default):
    """Return the configured value for `name`, else `default` (the original literal)."""
    env = os.environ.get("ARISE_" + name.upper())
    if env is not None:
        value = _coerce(env, default)
        _USED[name] = (value, "env")
        return value
    data = _load()
    if name in data:
        value = data[name]
        _USED[name] = (value, "preset")
        return value
    _USED[name] = (default, "default")
    return default


def active_preset():
    _load()
    return _CACHE.get("path")


def describe():
    """One line per resolved setting -- print this at the top of a run."""
    src = active_preset() or "(none: all built-in defaults)"
    lines = ["# ARISE config: %s" % src]
    for k in sorted(_USED):
        v, origin = _USED[k]
        lines.append("#   %-28s = %-46r [%s]" % (k, v, origin))
    return "\n".join(lines)
