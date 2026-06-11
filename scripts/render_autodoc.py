"""A small, dependency-free HTML renderer for the autodoc IR.

This is a *dev tool*, deliberately living outside ``vocal.autodoc`` — that
package's contract is that the IR is the deliverable and no renderer/CLI ships
with it. The renderer here exists only so a human can eyeball the IR.

The layout is a **dense flat grid**, driven from one IR→view-model walk:

* **meta** and **global attributes** render as tables — ``Name | Type | Kind |
  Details & rules``. The ``type`` constraint is hoisted to its own column; every
  other constraint (pattern / length / range / enum) is phrased as a rule
  statement and listed in the Details column alongside any validator rules.
* each **variable** is a collapsed ``<details>`` whose body is its own headered
  table; all tables share one fixed colgroup so columns line up across
  variables. Its ``long_name`` is lifted to a muted subheading (and still
  appears in the table).
* **groups** are collapsible, nesting-aware containers; nested groups indent.
* **model-bound validator rules** (e.g. a rule on the ``Variable`` model) are
  wrapped in a "Rules for <model>" callout so they don't read as a stray list.

It handles both IR roots, dispatching on ``doc.mode``: a **project** node carries
rule-bearing specs (type / required / optional / description / constraints) and
its variable/dimension/group slots are ``NodeRef`` redirects to the templates in
``defs``; a **product** node carries concrete facts (constant value / derived /
optional) with groups inlined.

Usage::

    python scripts/render_autodoc.py --project PATH_TO_PACKAGE [--out out.html] [--open]
    python scripts/render_autodoc.py --product PATH_TO_PRODUCT_JSON [--out out.html] [--open]
"""

from __future__ import annotations

import argparse
import html
import sys
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

from vocal.autodoc.ir import (
    ConstraintDoc,
    DimensionDoc,
    GroupDoc,
    NodeRef,
    ProductDoc,
    ProjectDoc,
    VariableDoc,
)

# --------------------------------------------------------------------------- #
# Leaf helpers — escaping, the type hoist, constraint phrasing, value display.
# --------------------------------------------------------------------------- #


def _esc(value: object) -> str:
    """HTML-escape any value's string form (None becomes an empty string)."""
    return html.escape("" if value is None else str(value))


def _format_value(value: object) -> str:
    """Render a concrete product value for display.

    Product mode carries no strict dtype, so quote strings to set them apart
    from numeric / boolean values at a glance ("FAAM" vs 42). ``None`` shows as
    ``null`` (the value really was JSON null), everything else as-is.
    """
    if isinstance(value, str):
        return f'"{_esc(value)}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return _esc(value)


def _split_type(
    constraints: list[ConstraintDoc] | None,
) -> tuple[str | None, list[ConstraintDoc]]:
    """Hoist the ``type`` constraint to a string for the prominent type column,
    returning it plus the remaining constraints (preserving order). Every
    non-type constraint is then phrased as a rule statement by the caller."""
    type_value: str | None = None
    rest: list[ConstraintDoc] = []
    for c in constraints or []:
        if c.kind == "type" and c.detail:
            type_value = str(c.detail.get("type"))
        else:
            rest.append(c)
    return type_value, rest


def _constraint_rule_text(c: ConstraintDoc) -> str:
    """Phrase a constraint as a rule statement.

    Mirrors the descriptions vocal's own ``PlaceholderConstraints`` checks emit
    ("Value matches regex …", "Value length is at least …"), so a documented
    constraint reads the same as the runtime check it stands for. Range, enum and
    any unrecognised kind get an analogous "Value …" phrasing.
    """
    detail = c.detail or {}
    if c.kind == "pattern":
        return f"Value matches regex <code>{_esc(detail.get('pattern'))}</code>"
    if c.kind == "length":
        lo = detail.get("min_length")
        hi = detail.get("max_length")
        if lo is not None and hi is not None:
            return f"Value length is between {_esc(lo)} and {_esc(hi)}"
        if lo is not None:
            return f"Value length is at least {_esc(lo)}"
        return f"Value length is at most {_esc(hi)}"
    if c.kind == "range":
        clauses = []
        if "gt" in detail:
            clauses.append(f"greater than {_esc(detail['gt'])}")
        if "ge" in detail:
            clauses.append(f"at least {_esc(detail['ge'])}")
        if "lt" in detail:
            clauses.append(f"less than {_esc(detail['lt'])}")
        if "le" in detail:
            clauses.append(f"at most {_esc(detail['le'])}")
        return "Value is " + " and ".join(clauses)
    if c.kind == "enum":
        values = ", ".join(_esc(v) for v in detail.get("values", []))
        return f"Value is one of: {values}"
    return f"{_esc(c.kind)}: {_esc(detail)}"


# --------------------------------------------------------------------------- #
# View model — one walk of the IR into plain, layout-neutral display facts.
# --------------------------------------------------------------------------- #


@dataclass
class RuleVM:
    text: str  # html-ready (may contain <code>)
    members: list[str] = field(default_factory=list)


@dataclass
class AttrVM:
    name: str
    badges: list[tuple[str, str]]  # (text, kind)
    rules: list[RuleVM]
    value: str | None = None  # product: formatted concrete value (escaped)
    value_muted: str | None = None  # product: e.g. "derived at runtime"
    desc: str | None = None  # project
    example: str | None = None  # project


@dataclass
class DimVM:
    name: str
    badge: tuple[str, str] | None
    rules: list[RuleVM]
    ref: str | None = None  # NodeRef target, if this is a template reference


@dataclass
class VarVM:
    name: str
    badges: list[tuple[str, str]]
    attrs: list[AttrVM]
    rules: list[RuleVM]
    datatype: str | None = None
    dims: list[str] = field(default_factory=list)
    long_name: str | None = None
    required: bool | None = None
    ref: str | None = None


@dataclass
class GroupVM:
    name: str
    attrs: list[AttrVM]
    vars: list[VarVM]
    dims: list[DimVM]
    groups: list["GroupVM"]
    rules: list[RuleVM]
    anchor: str | None = None
    ref: str | None = None


@dataclass
class TemplateVM:
    name: str
    anchor: str
    kind: str  # group | variable | dimension
    group: GroupVM | None
    struct: list[tuple[str, str, str]]  # (name, marker, desc)
    attrs: list[AttrVM]
    rules: list[RuleVM]


@dataclass
class DocVM:
    mode: str
    title: str | None  # dataset short_name (product) — raw, escaped at render time
    meta: list[AttrVM]
    attributes: list[AttrVM]
    dataset_rules: list[RuleVM]
    dims: list[DimVM]
    vars: list[VarVM]
    groups: list[GroupVM]
    templates: list[TemplateVM]
    diagnostics: list[str]


def _rules_vm(rules, rule_constraints=()) -> list[RuleVM]:
    out = [RuleVM(_constraint_rule_text(c)) for c in rule_constraints]
    for r in rules or []:
        out.append(RuleVM(_esc(r.description), [_esc(m) for m in (r.members or [])]))
    return out


def _attr_vm(attr, mode: str) -> AttrVM:
    type_value, rule_constraints = _split_type(attr.constraints)
    badges: list[tuple[str, str]] = []
    value = value_muted = desc = example = None
    if mode == "project":
        if type_value:
            badges.append((f"type: {type_value}", "type"))
        badges.append(("required", "req") if attr.required else ("optional", "opt"))
        if attr.description:
            desc = _esc(attr.description)
        if attr.example is not None:
            example = _esc(attr.example)
    else:  # product
        if attr.datatype:
            badges.append((f"type: {attr.datatype}", "type"))
        if attr.derived:
            badges.append(("derived", "derived"))
            # Optionality is only declared by derived placeholders; required is
            # the default, so we don't badge it (noise). A concrete value is a
            # "constant" baked into the product.
            if not attr.required:
                badges.append(("optional", "opt"))
            value_muted = "derived at runtime"
        else:
            badges.append(("constant", "const"))
            value = _format_value(attr.value)
    return AttrVM(
        _esc(attr.name), badges, _rules_vm(attr.rules, rule_constraints),
        value, value_muted, desc, example,
    )


def _dim_vm(d, mode: str) -> DimVM:
    if isinstance(d, NodeRef):
        return DimVM("", None, [], ref=d.ref)
    if mode == "project":
        return DimVM(_esc("<dimension template>"), None, _rules_vm(d.rules))
    size = "unlimited" if d.size is None else str(d.size)
    return DimVM(_esc(d.name), (size, "dim"), [])


def _var_vm(v, mode: str) -> VarVM:
    if isinstance(v, NodeRef):
        return VarVM("", [], [], [], ref=v.ref)
    badges: list[tuple[str, str]] = []
    name = _esc(v.name) if v.name else _esc("<variable template>")
    if mode == "product":
        if v.datatype:
            badges.append((f"type: {v.datatype}", "type"))
        if v.dimensions:
            badges.append(("dims: " + ", ".join(v.dimensions), "dim"))
    # Surface the concrete long_name as a subheading (it still appears in the
    # attribute list below — we only lift a copy up to the header).
    long_name = next(
        (a.value for a in v.attributes if a.name == "long_name" and isinstance(a.value, str)),
        None,
    )
    return VarVM(
        name, badges, [_attr_vm(a, mode) for a in v.attributes], _rules_vm(v.rules),
        datatype=v.datatype, dims=list(v.dimensions or []),
        long_name=_esc(long_name) if long_name else None,
        required=v.required,
    )


def _group_vm(g, mode: str, name: str | None = None, anchor: str | None = None) -> GroupVM:
    if isinstance(g, NodeRef):
        return GroupVM("", [], [], [], [], [], ref=g.ref)
    nm = name if name is not None else (_esc(g.name) if g.name else _esc("<group template>"))
    return GroupVM(
        nm,
        [_attr_vm(a, mode) for a in g.attributes],
        [_var_vm(v, mode) for v in g.variables],
        [_dim_vm(d, mode) for d in g.dimensions],
        [_group_vm(sg, mode) for sg in g.groups],
        _rules_vm(g.rules),
        anchor=anchor,
    )


def _templates_vm(defs, mode: str) -> list[TemplateVM]:
    out: list[TemplateVM] = []
    for name, tmpl in defs.items():
        anchor = f"def-{name}"
        if isinstance(tmpl, GroupDoc):
            gvm = _group_vm(tmpl, mode, name=_esc(f"{name} template"), anchor=anchor)
            out.append(TemplateVM(_esc(name), anchor, "group", gvm, [], [], []))
        elif isinstance(tmpl, VariableDoc):
            struct = [
                ("name", "per variable", "the variable's name"),
                ("datatype", "per variable", "the variable's data type"),
                ("dimensions", "per variable", "the dimensions it spans"),
            ]
            out.append(TemplateVM(
                _esc(name), anchor, "variable", None, struct,
                [_attr_vm(a, mode) for a in tmpl.attributes], _rules_vm(tmpl.rules),
            ))
        elif isinstance(tmpl, DimensionDoc):
            struct = [
                ("name", "per dimension", "the dimension's name"),
                ("size", "per dimension", "the length — null means unlimited"),
            ]
            out.append(TemplateVM(
                _esc(name), anchor, "dimension", None, struct, [], _rules_vm(tmpl.rules)
            ))
    return out


def _doc_title(meta_attrs) -> str | None:
    """The dataset's display name from its meta — short_name, else a canonical /
    long name. Uses the concrete value (product) or the spec's example (project)."""
    def text(name: str) -> str | None:
        for a in meta_attrs:
            if a.name == name:
                if isinstance(a.value, str):
                    return a.value
                if isinstance(a.example, str):
                    return a.example
        return None

    return text("short_name") or text("canonical_name") or text("long_name")


def build_doc(doc: ProjectDoc | ProductDoc) -> DocVM:
    """Walk an IR root into the layout-neutral :class:`DocVM` view model."""
    ds = doc.dataset
    defs = getattr(doc, "defs", {}) or {}
    return DocVM(
        mode=doc.mode,
        title=_doc_title(ds.meta),
        meta=[_attr_vm(a, doc.mode) for a in ds.meta],
        attributes=[_attr_vm(a, doc.mode) for a in ds.attributes],
        dataset_rules=_rules_vm(ds.rules),
        dims=[_dim_vm(d, doc.mode) for d in ds.dimensions],
        vars=[_var_vm(v, doc.mode) for v in ds.variables],
        groups=[_group_vm(g, doc.mode) for g in ds.groups],
        templates=_templates_vm(defs, doc.mode),
        diagnostics=list(doc.diagnostics),
    )


# --------------------------------------------------------------------------- #
# Shared markup atoms.
# --------------------------------------------------------------------------- #


def _badges(badges: list[tuple[str, str]]) -> str:
    return "".join(f'<span class="badge {k}">{_esc(t)}</span>' for t, k in badges)


def _rules_block(rules: list[RuleVM]) -> str:
    if not rules:
        return ""
    items = []
    for r in rules:
        members = ""
        if r.members:
            members = '<div class="members">' + " ".join(
                f'<span class="chip">{m}</span>' for m in r.members
            ) + "</div>"
        items.append(f"<li>{r.text}{members}</li>")
    return f'<div class="rules"><ul>{"".join(items)}</ul></div>'


def _ref_line(label: str, ref: str) -> str:
    """A NodeRef redirect rendered as a plain link (no table/box around it), so
    variable / dimension / group redirects all read the same in project mode."""
    return f'<div class="ref">↳ {label} → <a href="#def-{ref}">{ref}</a></div>'


def _scope_label(name: str) -> str:
    """The model name for a "Rules for …" heading — drop a trailing " template"
    (a group template's display name) so it reads "Rules for Group", not "… Group template"."""
    return name[: -len(" template")] if name.endswith(" template") else name


def _model_rules(scope: str, rules: list[RuleVM]) -> str:
    """Model-bound validator rules (declared on the Variable / Group / Dimension
    model itself) wrapped in a labelled callout, so they read as rules *about the
    model* rather than a stray bullet list floating beside the attribute table."""
    if not rules:
        return ""
    return (
        f'<div class="mrules"><div class="mrules-head">Rules for {scope}</div>'
        f"{_rules_block(rules)}</div>"
    )


# --------------------------------------------------------------------------- #
# Tables.
# --------------------------------------------------------------------------- #


def _type_cell(a: AttrVM) -> str:
    for text, kind in a.badges:
        if kind == "type":
            return f'<code>{_esc(text.removeprefix("type: "))}</code>'
    return ""


def _kind_cell(a: AttrVM) -> str:
    return _badges([b for b in a.badges if b[1] != "type"])


def _details_cell(a: AttrVM) -> str:
    out = []
    if a.desc:
        out.append(f'<div class="desc">{a.desc}</div>')
    if a.example is not None:
        out.append(f'<div class="ex">e.g. <code>{a.example}</code></div>')
    if a.value is not None:
        out.append(f"<code>{a.value}</code>")
    if a.value_muted:
        out.append(f'<span class="muted">{a.value_muted}</span>')
    out.append(_rules_block(a.rules))
    return "".join(o for o in out if o) or "—"


def _attr_rows(attrs: list[AttrVM]) -> str:
    return "".join(
        f'<tr><td class="name">{a.name}</td><td>{_type_cell(a)}</td>'
        f'<td>{_kind_cell(a)}</td><td class="details">{_details_cell(a)}</td></tr>'
        for a in attrs
    )


def _attr_table(attrs: list[AttrVM]) -> str:
    return (
        "<table><thead><tr><th>Name</th><th>Type</th><th>Kind</th>"
        f"<th>Details &amp; rules</th></tr></thead><tbody>{_attr_rows(attrs)}</tbody></table>"
    )


def _dim_table(dims: list[DimVM]) -> str:
    # A NodeRef redirect (project mode) is a plain link, not a table row — so we
    # don't render Name/Size/Rules column headers over a single "see template" line.
    refs = [d for d in dims if d.ref]
    concrete = [d for d in dims if not d.ref]
    out = ""
    if concrete:
        rows = "".join(
            f'<tr><td class="name">{d.name}</td>'
            f'<td>{_esc(d.badge[0]) if d.badge else ""}</td>'
            f'<td class="details">{_rules_block(d.rules) or "—"}</td></tr>'
            for d in concrete
        )
        out += (
            "<table><thead><tr><th>Name</th><th>Size</th><th>Rules</th></tr>"
            f"</thead><tbody>{rows}</tbody></table>"
        )
    out += "".join(_ref_line("dimension template", d.ref) for d in refs)
    return out


def _struct_table(struct: list[tuple[str, str, str]]) -> str:
    rows = "".join(
        f'<tr><td class="name">{n}</td><td><span class="badge varies">{m}</span></td>'
        f'<td class="details">{d}</td></tr>'
        for n, m, d in struct
    )
    return (
        "<table><thead><tr><th>Field</th><th>Value</th><th>Meaning</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


# Each variable is a collapsed <details> whose body is its own headered table;
# every table shares one fixed colgroup, so columns stay aligned across all
# variables (the "flat grid" feel) even though each carries its own header.
_FLAT_COLS = (
    '<colgroup><col style="width:24%"><col style="width:13%">'
    '<col style="width:20%"><col></colgroup>'
)


def _flat_attr_table(attrs: list[AttrVM]) -> str:
    return (
        f'<table class="vt">{_FLAT_COLS}<thead><tr><th>Name</th><th>Type</th>'
        f'<th>Kind</th><th>Details &amp; rules</th></tr></thead>'
        f"<tbody>{_attr_rows(attrs)}</tbody></table>"
    )


def _var_head_extra(v: VarVM) -> str:
    bits = []
    if v.datatype:
        bits.append(f'<span class="badge type">{_esc(v.datatype)}</span>')
    if v.dims:
        bits.append(f'<span class="badge dim">dims: {_esc(", ".join(v.dims))}</span>')
    # Whether a conforming dataset must contain this variable; absent (None) on
    # the project template, so only concrete product variables get a badge.
    if v.required is not None:
        text, kind = ("required", "req") if v.required else ("optional", "opt")
        bits.append(f'<span class="badge {kind}">{text}</span>')
    return " ".join(bits)


def _attr_count(v: VarVM) -> str:
    n = len(v.attrs)
    return f'<span class="cnt">{n} attr{"" if n == 1 else "s"}</span>'


def _variable(v: VarVM) -> str:
    body = _model_rules(v.name, v.rules) + (_flat_attr_table(v.attrs) if v.attrs else "")
    sub = f'<span class="vsub">{v.long_name}</span>' if v.long_name else ""
    # Badges sit on the name's row so they follow the variable name directly,
    # rather than after the (possibly wider) subtitle below it.
    name_row = f'<span class="vrow"><span class="vname">{v.name}</span>{_var_head_extra(v)}</span>'
    title = f'<span class="vtitle">{name_row}{sub}</span>'
    return (
        '<details class="grp"><summary><span class="arr"></span>'
        f'{title}{_attr_count(v)}</summary>'
        f'<div class="pad">{body}</div></details>'
    )


def _variables(vars_: list[VarVM]) -> str:
    # Only the real variable rows live in the bordered .flatlist box; a NodeRef
    # redirect (project mode) is a plain link, matching dimensions / groups.
    refs = [v for v in vars_ if v.ref]
    concrete = [v for v in vars_ if not v.ref]
    out = ""
    if concrete:
        out += '<div class="flatlist">' + "".join(_variable(v) for v in concrete) + "</div>"
    out += "".join(_ref_line("variable template", v.ref) for v in refs)
    return out


def _group_counts(g: GroupVM) -> str:
    def n(items, word):
        return f'{len(items)} {word}{"" if len(items) == 1 else "s"}' if items else ""
    bits = [n(g.vars, "var"), n(g.dims, "dim"), n(g.attrs, "attr"), n(g.groups, "group")]
    bits = [b for b in bits if b]
    return f'<span class="cnt">{", ".join(bits)}</span>' if bits else ""


def _group(g: GroupVM) -> str:
    if g.ref:
        return _ref_line("recursive group", g.ref)
    parts = [_model_rules(_scope_label(g.name), g.rules)]
    if g.attrs:
        parts.append("<h3>Attributes</h3>" + _attr_table(g.attrs))
    if g.dims:
        parts.append("<h3>Dimensions</h3>" + _dim_table(g.dims))
    if g.vars:
        parts.append("<h3>Variables</h3>" + _variables(g.vars))
    if g.groups:
        parts.append("<h3>Groups</h3>" + "".join(_group(sg) for sg in g.groups))
    body = "".join(p for p in parts if p)
    anchor = f' id="{g.anchor}"' if g.anchor else ""
    return (
        f'<details class="group" open{anchor}><summary><span class="arr"></span>'
        f'<span class="gtag">group</span><span class="gname">{g.name}</span>'
        f'{_group_counts(g)}</summary><div class="gbody">{body}</div></details>'
    )


def _template(t: TemplateVM) -> str:
    if t.group:
        return _group(t.group)
    inner = f"<h3>{t.name} template — structure</h3>" + _struct_table(t.struct)
    if t.attrs:
        inner += "<h3>Attributes</h3>" + _attr_table(t.attrs)
    inner += _model_rules(t.name, t.rules)
    return f'<div id="{t.anchor}">{inner}</div>'


# --------------------------------------------------------------------------- #
# Document.
# --------------------------------------------------------------------------- #

_CSS = """
* { box-sizing: border-box; }
body { font: 14px/1.45 system-ui, sans-serif; margin: 0; background: #f7f8fa; color: #1a1d21; }
header { background: #fff; border-bottom: 1px solid #e2e5e9; padding: 14px 28px; position: sticky; top: 0; z-index: 3; display: flex; align-items: center; gap: 10px; }
header h1 { margin: 0; font-size: 19px; }
header .brand { margin-left: auto; font: 800 13px ui-monospace, monospace; letter-spacing: .22em; text-transform: uppercase; color: #b3b9c0; }
.mode { font: 700 10px ui-monospace, monospace; letter-spacing: .14em; text-transform: uppercase; color: #2f855a; }
main { max-width: 1120px; margin: 0 auto; padding: 20px 28px 100px; }
h2 { font-size: 15px; margin: 32px 0 8px; }
h3 { font-size: 12px; color: #6a727b; margin: 16px 0 6px; font-family: ui-monospace, monospace; text-transform: uppercase; letter-spacing: .05em; }
.count { font-size: 12px; color: #9aa1a9; font-weight: 400; }
table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e2e5e9; border-radius: 8px; overflow: hidden; }
th { text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: #8a929b; padding: 8px 12px; border-bottom: 2px solid #e2e5e9; background: #fafbfc; }
td { padding: 9px 12px; border-bottom: 1px solid #eef0f2; vertical-align: top; }
tr:last-child td { border-bottom: 0; }
tbody tr:nth-child(even) td { background: #fcfcfd; }
td.name { font-family: ui-monospace, monospace; font-weight: 600; white-space: nowrap; }
td.name a { color: #2b6cb0; text-decoration: none; } td.name a:hover { text-decoration: underline; }
td.num { text-align: right; color: #6a727b; font-variant-numeric: tabular-nums; }
.details .muted { color: #9aa1a9; font-style: italic; } .details .desc { margin: 0 0 4px; } .details .ex { font-size: 12px; color: #6a727b; }
.rules ul { margin: 4px 0 0; padding-left: 16px; } .rules li { margin: 2px 0; font-size: 13px; }
.ref { font-size: 13px; color: #6a727b; padding: 6px 0; } .ref a { color: #2b6cb0; }

.badge { font-size: 11px; padding: 2px 8px; border-radius: 4px; background: #edf0f3; color: #4a5158; white-space: nowrap; }
.badge.type { background: #c9d8fb; color: #1f3d80; border: 1px solid #9db4ed; font-family: ui-monospace, monospace; }
.badge.req { background: #fdeaea; color: #b03535; } .badge.opt { background: #eef1f4; color: #6a727b; border: 1px solid #c8ced6; }
.badge.derived { background: #fff4e0; color: #9a6700; border: 1px solid #e8cf9a; } .badge.dim { background: #cfe9d2; color: #246138; border: 1px solid #a6d4ac; }
.badge.const { background: #e6f1f4; color: #2c5d6e; border: 1px solid #b3d2dc; }
.badge.varies { background: #eef0ff; color: #5560a0; font-style: italic; }
code { background: #f0f2f4; padding: 1px 5px; border-radius: 3px; font-size: .9em; font-family: ui-monospace, monospace; }
.chip { font-size: 12px; background: #f0f2f4; border: 1px solid #e2e5e9; border-radius: 4px; padding: 1px 7px; }
.members { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }

.flatlist { border: 1px solid #e2e5e9; border-radius: 8px; overflow: hidden; background: #fff; }
details.grp + details.grp { border-top: 1px solid #e2e5e9; }
details.grp > summary { display: flex; align-items: center; gap: 8px; padding: 9px 12px; cursor: pointer; user-select: none; background: #eef2f6; font-family: ui-monospace, monospace; }
details.grp > summary:hover { background: #e6ebf1; }
details.grp > summary::-webkit-details-marker { display: none; }
details.grp > summary::marker { content: ""; }
.arr::before { content: "▸"; display: inline-block; width: 14px; color: #6a727b; }
details[open] > summary .arr::before { content: "▾"; }
.vtitle { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.vrow { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.vname { font-weight: 700; }
.vsub { font: 400 12px/1.3 system-ui, sans-serif; color: #8a929b; }
.cnt { margin-left: auto; font: 400 12px system-ui, sans-serif; color: #6a727b; }
details.grp .pad { padding: 0; }
details.grp .pad > .rules { padding: 8px 12px; }
table.vt { border: 0; border-radius: 0; table-layout: fixed; }
table.vt td { word-break: break-word; }
table.vt td.name { padding-left: 22px; }

details.group { border: 1px solid #d2dae3; border-radius: 8px; margin: 12px 0; background: #fff; }
details.group > summary { display: flex; align-items: center; gap: 8px; padding: 10px 14px; cursor: pointer; user-select: none; background: #e3e9f1; border-radius: 7px 7px 0 0; font-family: ui-monospace, monospace; font-size: 14px; }
details.group:not([open]) > summary { border-radius: 7px; }
details.group > summary:hover { background: #d9e1ec; }
details.group > summary::-webkit-details-marker { display: none; }
details.group > summary::marker { content: ""; }
.gtag { font: 700 9px system-ui, sans-serif; letter-spacing: .12em; text-transform: uppercase; color: #5560a0; background: #eef0ff; padding: 2px 6px; border-radius: 4px; }
.gname { font-weight: 700; }
.gbody { padding: 2px 14px 12px; }
.gbody > h3:first-child { margin-top: 12px; }
details.group details.group { border-left: 3px solid #b9c6da; }

.mrules { border: 1px solid #d9e0ea; background: #f4f7fb; border-radius: 8px; padding: 8px 12px; margin: 12px 0; }
.mrules-head { font: 700 14px system-ui, sans-serif; color: #4a5573; margin-bottom: 4px; }
.mrules .rules { margin: 0; } .mrules .rules ul { margin: 2px 0 0; }
"""


def _header(vm: DocVM, title: str | None) -> str:
    name = _esc(title or vm.title or "autodoc")
    return (
        f'<header><h1>{name}</h1><span class="mode">{vm.mode}</span>'
        '<span class="brand">vocal</span></header><main>'
    )


def _document(vm: DocVM, title: str | None) -> str:
    body = [_header(vm, title)]
    if vm.meta:
        body.append(f'<h2>Meta <span class="count">{len(vm.meta)}</span></h2>{_attr_table(vm.meta)}')
    if vm.attributes:
        body.append(f'<h2>Global attributes <span class="count">{len(vm.attributes)}</span></h2>{_attr_table(vm.attributes)}')
    if vm.dataset_rules:
        body.append(f'<h2>Dataset rules <span class="count">{len(vm.dataset_rules)}</span></h2>{_rules_block(vm.dataset_rules)}')
    if vm.dims:
        body.append(f'<h2>Dimensions <span class="count">{len(vm.dims)}</span></h2>{_dim_table(vm.dims)}')
    if vm.vars:
        body.append(f'<h2>Variables <span class="count">{len(vm.vars)}</span></h2>{_variables(vm.vars)}')
    if vm.groups:
        body.append(f'<h2>Groups <span class="count">{len(vm.groups)}</span></h2>' + "".join(_group(g) for g in vm.groups))
    if vm.templates:
        body.append("<h2>Templates</h2>" + "".join(_template(t) for t in vm.templates))
    if vm.diagnostics:
        body.append(
            f'<h2>Diagnostics <span class="count">{len(vm.diagnostics)}</span></h2>'
            '<div class="rules"><ul>' + "".join(f"<li>{_esc(d)}</li>" for d in vm.diagnostics) + "</ul></div>"
        )
    body.append("</main>")
    page_title = title or vm.title or vm.mode
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>autodoc — {page_title}</title><style>{_CSS}</style></head>"
        f"<body>{''.join(body)}</body></html>"
    )


def render(doc: ProjectDoc | ProductDoc, title: str | None = None) -> str:
    """Render an autodoc IR root to a self-contained HTML document string.

    ``title`` overrides the heading — used for projects, whose name is the
    package, not a value carried in the IR. Products fall back to the dataset's
    ``short_name`` from the meta section.
    """
    return _document(build_doc(doc), title)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render an autodoc IR to HTML.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--project", metavar="PATH", help="path to a vocal project package")
    group.add_argument("--product", metavar="PATH", help="path to a product-pack JSON")
    parser.add_argument(
        "--out", default="autodoc.html", help="output HTML path (default: autodoc.html)"
    )
    parser.add_argument("--open", action="store_true", help="open the result in a browser")
    args = parser.parse_args(argv)

    from vocal.autodoc import document_product, document_project

    title = None
    if args.project:
        from vocal.utils import import_project

        doc: ProjectDoc | ProductDoc = document_project(import_project(args.project).Dataset)
        # A project's name is its package, not anything carried in the IR.
        title = Path(args.project.rstrip("/")).name
    else:
        doc = document_product(args.product)

    out = Path(args.out)
    out.write_text(render(doc, title), encoding="utf-8")
    print(f"wrote {out} ({doc.mode})", file=sys.stderr)
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
