"""Integration tests for the autodoc walkers (slice 1: global attributes).

Fixtures are tiny, hermetic, synthetic model trees and product JSON, following
the inline-model style in ``tests/test_validation.py`` — no dependency on an
externally-installed project.
"""

from typing import Optional

from pydantic import BaseModel

from vocal.autodoc import (
    ProductDoc,
    ProjectDoc,
    document_product,
    document_project,
)
from vocal.field import Field
from vocal.mixins import (
    VocalAttributesMixin,
    VocalDatasetMixin,
    VocalDimensionMixin,
    VocalVariableMixin,
)
from vocal.validation import variable_has_dimensions


# ---------------------------------------------------------------------------
# Synthetic project: a Dataset whose `attributes` field is an attributes model
# with one required and one optional global attribute, plus a variable model
# (with its own attributes + a structural rule) and a dimension model.
# ---------------------------------------------------------------------------


class _GlobalAttributes(BaseModel, VocalAttributesMixin):
    title: str = Field(description="A brief title", example="My dataset")
    comment: Optional[str] = Field(
        description="An optional comment", example="a note", default=None
    )
    revision: int = Field(description="Revision number", ge=0, example=1)


class _VariableAttributes(BaseModel, VocalAttributesMixin):
    long_name: str = Field(description="A long name", example="Air temperature")
    units: str = Field(description="The units", example="K")


class _VariableMeta(BaseModel):
    name: str
    datatype: str


class _Variable(BaseModel, VocalVariableMixin):
    meta: _VariableMeta
    dimensions: list[str]
    attributes: _VariableAttributes

    _v_dims = variable_has_dimensions("time", ["time"])


class _Dimension(BaseModel, VocalDimensionMixin):
    name: str
    size: Optional[int]


class _Dataset(BaseModel, VocalDatasetMixin):
    attributes: _GlobalAttributes
    dimensions: list[_Dimension]
    variables: list[_Variable]


def _attr(doc, name):
    return next(a for a in doc.dataset.attributes if a.name == name)


class TestDocumentProject:
    def test_returns_project_doc(self) -> None:
        doc = document_project(_Dataset)
        assert isinstance(doc, ProjectDoc)
        assert doc.mode == "project"

    def test_documents_every_global_attribute(self) -> None:
        doc = document_project(_Dataset)
        assert {a.name for a in doc.dataset.attributes} == {
            "title",
            "comment",
            "revision",
        }

    def test_required_attribute_fields(self) -> None:
        title = _attr(document_project(_Dataset), "title")
        assert title.description == "A brief title"
        assert title.example == "My dataset"
        assert title.required is True

    def test_optional_attribute_is_not_required(self) -> None:
        comment = _attr(document_project(_Dataset), "comment")
        assert comment.required is False

    def test_attribute_constraints_are_normalized(self) -> None:
        from vocal.autodoc import ConstraintDoc

        revision = _attr(document_project(_Dataset), "revision")
        assert ConstraintDoc(kind="type", detail={"type": "integer"}) in (
            revision.constraints
        )
        assert ConstraintDoc(kind="range", detail={"ge": 0}) in revision.constraints

    def test_concrete_fields_absent_in_project_mode(self) -> None:
        title = _attr(document_project(_Dataset), "title")
        assert title.value is None
        assert title.derived is None
        assert title.datatype is None

    def test_keys_on_canonical_field_name_not_mixin(self) -> None:
        # A model that is not a VocalDataset but does carry an `attributes`
        # field is still walked — the walk keys on the field name.
        class PlainDataset(BaseModel):
            attributes: _GlobalAttributes

        doc = document_project(PlainDataset)
        assert {a.name for a in doc.dataset.attributes} == {
            "title",
            "comment",
            "revision",
        }

    def test_roundtrips_through_json(self) -> None:
        doc = document_project(_Dataset)
        assert ProjectDoc.model_validate_json(doc.model_dump_json()) == doc


# ---------------------------------------------------------------------------
# Synthetic product: raw JSON with concrete and derived global attributes.
# ---------------------------------------------------------------------------

_PRODUCT = {
    "meta": {"file_pattern": "thing_{date}.nc"},
    "attributes": {
        "title": "My dataset",
        "flight_number": "<str: derived_from_file>",
        "altitude": "<float32: derived_from_file>",
        "comment": "<str: derived_from_file optional>",
    },
    "dimensions": [
        {"name": "time", "size": None},
        {"name": "bins", "size": 512},
    ],
    "variables": [
        {
            "meta": {"name": "temperature", "datatype": "<float32>"},
            "dimensions": ["time"],
            "attributes": {
                "long_name": "Air temperature",
                "units": "K",
            },
        },
        {
            "meta": {"name": "spectrum", "datatype": "<int16>"},
            "dimensions": ["time", "bins"],
            "attributes": {"long_name": "A spectrum", "units": "1"},
        },
    ],
}


class TestDocumentProduct:
    def test_returns_product_doc(self) -> None:
        doc = document_product(_PRODUCT)
        assert isinstance(doc, ProductDoc)
        assert doc.mode == "product"

    def test_documents_every_attribute(self) -> None:
        doc = document_product(_PRODUCT)
        assert {a.name for a in doc.dataset.attributes} == {
            "title",
            "flight_number",
            "altitude",
            "comment",
        }

    def test_concrete_value_passes_through(self) -> None:
        title = _attr(document_product(_PRODUCT), "title")
        assert title.value == "My dataset"
        assert title.derived is False
        assert title.datatype is None

    def test_derived_placeholder_recovers_datatype(self) -> None:
        altitude = _attr(document_product(_PRODUCT), "altitude")
        assert altitude.derived is True
        assert altitude.datatype == "float32"
        assert altitude.value is None

    def test_rule_bearing_fields_absent_in_product_mode(self) -> None:
        title = _attr(document_product(_PRODUCT), "title")
        assert title.description is None
        assert title.example is None
        assert title.constraints is None

    def test_loads_from_path(self, tmp_path) -> None:
        import json

        path = tmp_path / "product.json"
        path.write_text(json.dumps(_PRODUCT))
        doc = document_product(path)
        assert {a.name for a in doc.dataset.attributes} == {
            "title",
            "flight_number",
            "altitude",
            "comment",
        }

    def test_roundtrips_through_json(self) -> None:
        doc = document_product(_PRODUCT)
        assert ProductDoc.model_validate_json(doc.model_dump_json()) == doc


# ---------------------------------------------------------------------------
# Variables + dimensions (slice 5): project templates vs. product concretes.
# ---------------------------------------------------------------------------


class TestProjectVariablesAndDimensions:
    def test_variables_hold_exactly_one_template(self) -> None:
        doc = document_project(_Dataset)
        assert len(doc.dataset.variables) == 1

    def test_template_has_no_concrete_fields(self) -> None:
        (template,) = document_project(_Dataset).dataset.variables
        assert template.name is None
        assert template.datatype is None
        assert template.dimensions is None

    def test_template_attributes_reuse_attribute_walk(self) -> None:
        (template,) = document_project(_Dataset).dataset.variables
        names = {a.name for a in template.attributes}
        assert names == {"long_name", "units"}
        long_name = next(a for a in template.attributes if a.name == "long_name")
        # Same rule-bearing AttributeDoc shape as the global attributes.
        assert long_name.description == "A long name"
        assert long_name.example == "Air temperature"
        assert long_name.required is True

    def test_template_carries_variable_model_rules(self) -> None:
        (template,) = document_project(_Dataset).dataset.variables
        assert template.rules is not None
        assert any("dimensions" in r.description for r in template.rules)

    def test_dimensions_hold_exactly_one_template(self) -> None:
        doc = document_project(_Dataset)
        assert len(doc.dataset.dimensions) == 1
        (template,) = doc.dataset.dimensions
        assert template.name is None
        assert template.size is None

    def test_roundtrips_through_json(self) -> None:
        doc = document_project(_Dataset)
        assert ProjectDoc.model_validate_json(doc.model_dump_json()) == doc


class TestProductVariablesAndDimensions:
    def _var(self, doc, name):
        return next(v for v in doc.dataset.variables if v.name == name)

    def test_documents_every_concrete_variable(self) -> None:
        doc = document_product(_PRODUCT)
        assert {v.name for v in doc.dataset.variables} == {
            "temperature",
            "spectrum",
        }

    def test_variable_meta_and_dimensions(self) -> None:
        temperature = self._var(document_product(_PRODUCT), "temperature")
        assert temperature.datatype == "float32"
        assert temperature.dimensions == ["time"]

    def test_variable_attributes_are_concrete(self) -> None:
        temperature = self._var(document_product(_PRODUCT), "temperature")
        long_name = next(
            a for a in temperature.attributes if a.name == "long_name"
        )
        assert long_name.value == "Air temperature"
        assert long_name.derived is False
        # Concrete attributes carry no rule-bearing fields.
        assert long_name.description is None

    def test_documents_every_concrete_dimension(self) -> None:
        doc = document_product(_PRODUCT)
        dims = {d.name: d.size for d in doc.dataset.dimensions}
        assert dims == {"time": None, "bins": 512}

    def test_roundtrips_through_json(self) -> None:
        doc = document_product(_PRODUCT)
        assert ProductDoc.model_validate_json(doc.model_dump_json()) == doc
