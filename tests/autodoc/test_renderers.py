"""Structural tests for the autodoc renderers and their registry.

The renderers are pure ``IR root -> str`` transforms, so these tests build a
tiny hermetic IR (a synthetic project model and a product dict, in the
``test_walkers`` style) and assert *structural* invariants of the rendered
output — that the right sections, names, badges, links and rule phrasings make
it through — rather than snapshotting exact bytes, which would lock the HTML
layout the renderer is meant to keep iterating on.
"""

from typing import Optional

from pydantic import BaseModel

from vocal.autodoc import document_product, document_project
from vocal.autodoc.renderers import RENDERERS, Renderer, get_renderer
from vocal.autodoc.renderers.html import render
from vocal.field import Field
from vocal.mixins import (
    VocalAttributesMixin,
    VocalDatasetMixin,
    VocalDimensionMixin,
    VocalVariableMixin,
)

import pytest


# ---------------------------------------------------------------------------
# Tiny hermetic IR — one global attribute carrying a range constraint, plus a
# variable model (so project mode emits a NodeRef redirect to a template def).
# ---------------------------------------------------------------------------


class _GlobalAttributes(BaseModel, VocalAttributesMixin):
    title: str = Field(description="A brief title", example="My dataset")
    revision: int = Field(description="Revision number", ge=0, example=1)


class _VariableAttributes(BaseModel, VocalAttributesMixin):
    long_name: str = Field(description="A long name", example="Air temperature")


class _VariableMeta(BaseModel):
    name: str
    datatype: str


class _Variable(BaseModel, VocalVariableMixin):
    meta: _VariableMeta
    dimensions: list[str]
    attributes: _VariableAttributes


class _Dimension(BaseModel, VocalDimensionMixin):
    name: str
    size: Optional[int]


class _DatasetMeta(BaseModel):
    file_pattern: str = Field(
        description="Canonical filename pattern", example="thing_{date}.nc"
    )
    short_name: Optional[str] = Field(
        description="A short name", example="thing", default=None
    )


class _Dataset(BaseModel, VocalDatasetMixin):
    meta: _DatasetMeta
    attributes: _GlobalAttributes
    dimensions: list[_Dimension]
    variables: list[_Variable]


_PRODUCT = {
    "meta": {"file_pattern": "thing_{date}.nc", "short_name": "thing"},
    "attributes": {
        "title": "My dataset",
        "altitude": "<float32: derived_from_file>",
    },
    "dimensions": [{"name": "time", "size": None}],
    "variables": [
        {
            "meta": {"name": "temperature", "datatype": "<float32>", "required": True},
            "dimensions": ["time"],
            "attributes": {"long_name": "Air temperature"},
        }
    ],
}


class TestRenderProject:
    def test_is_html_document_with_title_and_mode(self) -> None:
        out = render(document_project(_Dataset), "myproj")
        assert out.startswith("<!doctype html>")
        assert "myproj" in out  # title override
        assert "project" in out  # mode banner

    def test_global_attribute_and_required_badge_appear(self) -> None:
        out = render(document_project(_Dataset), "myproj")
        assert "title" in out
        assert "required" in out  # title is a required global attribute

    def test_range_constraint_is_phrased_as_a_rule(self) -> None:
        # The ge=0 constraint on `revision` should read as its rule sentence,
        # mirroring vocal's own constraint descriptions.
        out = render(document_project(_Dataset), "myproj")
        assert "at least 0" in out

    def test_variable_slot_renders_as_a_template_redirect(self) -> None:
        # Project mode redirects the variable slot to a template def, rendered as
        # an in-page anchor link rather than an inlined variable.
        out = render(document_project(_Dataset), "myproj")
        assert 'href="#def-' in out


class TestRenderProduct:
    def test_concrete_variable_and_value_appear(self) -> None:
        out = render(document_product(_PRODUCT))
        assert "temperature" in out  # inlined concrete variable
        assert "My dataset" in out  # concrete attribute value
        assert "product" in out  # mode banner

    def test_derived_placeholder_is_marked(self) -> None:
        out = render(document_product(_PRODUCT))
        assert "derived at runtime" in out

    def test_falls_back_to_short_name_without_title(self) -> None:
        out = render(document_product(_PRODUCT))
        assert "thing" in out  # short_name from meta

    def test_satisfies_standards_section_appears(self) -> None:
        out = render(
            document_product(_PRODUCT, satisfies_standards=["MYSTD-2.3+"])
        )
        assert "Satisfies standards" in out
        assert "MYSTD-2.3+" in out

    def test_no_satisfies_standards_section_when_absent(self) -> None:
        out = render(document_product(_PRODUCT))
        assert "Satisfies standards" not in out


class TestRegistry:
    def test_html_is_registered(self) -> None:
        r = get_renderer("html")
        assert isinstance(r, Renderer)
        assert r.extension == "html"
        assert callable(r.render)

    def test_unknown_format_raises_listing_available(self) -> None:
        with pytest.raises(ValueError) as exc:
            get_renderer("does-not-exist")
        msg = str(exc.value)
        assert "does-not-exist" in msg
        assert "html" in msg  # names the available formats

    def test_every_registered_renderer_has_a_callable_and_extension(self) -> None:
        for name, r in RENDERERS.items():
            assert callable(r.render), name
            assert r.extension and "." not in r.extension, name
