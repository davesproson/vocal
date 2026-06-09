"""Extract custom ``vocal`` validator rules off a model class (project mode).

A vocal standard's rules live partly in custom validators (``is_exact`` /
``is_in`` / ``in_vocabulary`` / the model-level structural validators) whose
human-readable meaning is carried on each validator's ``.description``. Those
rules do **not** survive into ``model_json_schema()``, so autodoc reads them by
class-static introspection: vocal attaches ``.description`` / ``.binding`` (and,
for ``in_vocabulary``, ``.vocabulary``) to the *raw* validator function so the
metadata survives pydantic's class construction and is reachable via
``cls.__dict__[name].__func__``. No model instance is synthesised.

This slice handles ``Attribute``-bound validators: each is routed to the
``AttributeDoc`` for the attribute it validates. ``Model``-bound (structural)
validators are routed to container nodes in a later slice; they are recognised
here but ignored by :func:`attribute_rules`.
"""

from __future__ import annotations

from typing import Any, Iterator

from pydantic import BaseModel

from vocal.validation import Attribute

from .ir import RuleDoc


def iter_validators(model: type[BaseModel]) -> Iterator[Any]:
    """Yield the raw validator functions declared directly on ``model``.

    A vocal validator is reachable as ``cls.__dict__[name].__func__`` and
    carries a ``binding`` attribute. Plain pydantic validators (e.g. the
    placeholder substitutor) lack that metadata and are skipped. Only the
    class's own ``__dict__`` is consulted, so a validator is documented on the
    class that declares it — never on a subclass that merely inherits it.
    """
    for member in vars(model).values():
        func = getattr(member, "__func__", None)
        if func is None:
            continue
        if not hasattr(func, "binding"):
            continue
        yield func


def rule_doc(func: Any) -> RuleDoc:
    """Build a :class:`RuleDoc` from a raw validator function.

    A controlled-vocabulary validator (``in_vocabulary``) carries the
    ``vocabulary`` it checks against; its enumerable members are listed when
    ``members()`` returns a list, and omitted (``None``) when the vocabulary is
    not enumerable, in which case the rule's ``description`` carries the prose.
    """
    members: list[str] | None = None
    vocabulary = getattr(func, "vocabulary", None)
    if vocabulary is not None:
        members = vocabulary.members()
    return RuleDoc(description=func.description, members=members)


def attribute_rules(model: type[BaseModel] | None) -> dict[str, list[RuleDoc]]:
    """Map attribute name -> the rules its ``Attribute``-bound validators impose.

    Returns an empty mapping when ``model`` is ``None`` or carries no
    attribute-bound validators.
    """
    rules: dict[str, list[RuleDoc]] = {}
    if model is None:
        return rules
    for func in iter_validators(model):
        binding = func.binding
        if isinstance(binding, Attribute):
            rules.setdefault(binding.name, []).append(rule_doc(func))
    return rules
