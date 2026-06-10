"""Integration tests for the autodoc walkers (slice 1: global attributes).

Fixtures are tiny, hermetic, synthetic model trees and product JSON, following
the inline-model style in ``tests/test_validation.py`` — no dependency on an
externally-installed project.
"""

from typing import Optional

from pydantic import BaseModel

from vocal.autodoc import (
    ConstraintDoc,
    DimensionDoc,
    GroupDoc,
    NodeRef,
    ProductDoc,
    ProjectDoc,
    VariableDoc,
    document_product,
    document_project,
)
from vocal.field import Field
from vocal.mixins import (
    VocalAttributesMixin,
    VocalDatasetMixin,
    VocalDimensionMixin,
    VocalGroupMixin,
    VocalVariableMixin,
)
from vocal.validation import Attribute, variable_has_dimensions, vocal_validator


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


class _GroupAttributes(BaseModel, VocalAttributesMixin):
    group_title: str = Field(description="The group title", example="A group")


class _GroupMeta(BaseModel):
    name: str


class Group(BaseModel, VocalGroupMixin):
    meta: _GroupMeta
    attributes: _GroupAttributes
    dimensions: list[_Dimension]
    variables: list[_Variable]
    groups: Optional[list["Group"]] = None


Group.model_rebuild()


class _Reference(BaseModel):
    title: str = Field(description="The reference title", example="A paper")
    doi: Optional[str] = Field(description="The DOI", example="10.1/2", default=None)


class _DatasetMeta(BaseModel):
    file_pattern: str = Field(
        description="Canonical filename pattern", example="thing_{date}.nc"
    )
    short_name: Optional[str] = Field(
        description="A short name", example="thing", default=None
    )
    description: Optional[str] = Field(
        description="A description of the dataset", default=None
    )
    references: Optional[list[_Reference]] = Field(
        description="References for the dataset", default=None
    )


class _Dataset(BaseModel, VocalDatasetMixin):
    meta: _DatasetMeta
    attributes: _GlobalAttributes
    dimensions: list[_Dimension]
    variables: list[_Variable]
    groups: Optional[list[Group]] = None


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
    "meta": {
        "file_pattern": "thing_{date}.nc",
        "short_name": "thing",
        "long_name": "A Thing Dataset",
        "description": "A concrete product describing some thing.",
        "references": [
            {"title": "A paper", "doi": "10.1/2"},
            {"title": "A website", "web": "https://example.com"},
        ],
    },
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
    "groups": [
        {
            "meta": {"name": "navigation"},
            "attributes": {"group_title": "Navigation data"},
            "dimensions": [{"name": "sps", "size": 32}],
            "variables": [
                {
                    "meta": {"name": "latitude", "datatype": "<float64>"},
                    "dimensions": ["time"],
                    "attributes": {
                        "long_name": "Latitude",
                        "units": "degrees_north",
                    },
                }
            ],
            "groups": [
                {
                    "meta": {"name": "raw"},
                    "attributes": {"group_title": "Raw navigation"},
                    "dimensions": [],
                    "variables": [],
                    "groups": [],
                }
            ],
        }
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

    def test_concrete_value_is_required_with_no_constraints(self) -> None:
        # A concrete value declares no optionality, so it keeps required=True and
        # carries no constraints (the renderer gates its badge on `derived`).
        title = _attr(document_product(_PRODUCT), "title")
        assert title.required is True
        assert title.constraints is None

    def test_optional_placeholder_is_not_required(self) -> None:
        comment = _attr(document_product(_PRODUCT), "comment")
        assert comment.derived is True
        assert comment.required is False

    def test_non_optional_placeholder_is_required(self) -> None:
        flight_number = _attr(document_product(_PRODUCT), "flight_number")
        assert flight_number.derived is True
        assert flight_number.required is True

    def test_placeholder_regex_surfaces_as_pattern_constraint(self) -> None:
        spec = {
            "attributes": {
                "fn": "<str: derived_from_file optional,regex=[a-z][0-9]{3}>"
            }
        }
        fn = _attr(document_product(spec), "fn")
        assert fn.required is False
        assert fn.constraints == [
            ConstraintDoc(kind="pattern", detail={"pattern": "[a-z][0-9]{3}"})
        ]

    def test_placeholder_length_surfaces_as_length_constraint(self) -> None:
        spec = {
            "attributes": {
                "tags": "<Array[str]: derived_from_file min_len=1,max_len=5>"
            }
        }
        tags = _attr(document_product(spec), "tags")
        assert tags.constraints == [
            ConstraintDoc(kind="length", detail={"min_length": 1, "max_length": 5})
        ]

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
    def test_variables_slot_is_a_ref_to_the_template(self) -> None:
        doc = document_project(_Dataset)
        (ref,) = doc.dataset.variables
        assert isinstance(ref, NodeRef)
        assert ref.ref == "_Variable"

    def test_variable_template_lives_in_defs(self) -> None:
        template = document_project(_Dataset).defs["_Variable"]
        assert isinstance(template, VariableDoc)
        assert template.name is None
        assert template.datatype is None
        assert template.dimensions is None

    def test_template_attributes_reuse_attribute_walk(self) -> None:
        template = document_project(_Dataset).defs["_Variable"]
        names = {a.name for a in template.attributes}
        assert names == {"long_name", "units"}
        long_name = next(a for a in template.attributes if a.name == "long_name")
        # Same rule-bearing AttributeDoc shape as the global attributes.
        assert long_name.description == "A long name"
        assert long_name.example == "Air temperature"
        assert long_name.required is True

    def test_template_carries_variable_model_rules(self) -> None:
        template = document_project(_Dataset).defs["_Variable"]
        assert template.rules is not None
        assert any("dimensions" in r.description for r in template.rules)

    def test_dimensions_slot_is_a_ref_to_the_template(self) -> None:
        doc = document_project(_Dataset)
        (ref,) = doc.dataset.dimensions
        assert isinstance(ref, NodeRef)
        assert ref.ref == "_Dimension"
        template = doc.defs["_Dimension"]
        assert isinstance(template, DimensionDoc)
        assert template.name is None
        assert template.size is None

    def test_variable_template_is_shared_not_duplicated(self) -> None:
        # The dataset and the group both use the same `_Variable` model, so it is
        # documented once in `defs` and both slots reference that single entry.
        doc = document_project(_Dataset)
        (dataset_ref,) = doc.dataset.variables
        (group_ref,) = doc.defs["Group"].variables
        assert dataset_ref.ref == group_ref.ref == "_Variable"
        assert isinstance(doc.defs["_Variable"], VariableDoc)

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


# ---------------------------------------------------------------------------
# Meta section (slice 7): project field specs vs. product concrete values.
# ---------------------------------------------------------------------------


def _meta(doc, name):
    return next(m for m in doc.dataset.meta if m.name == name)


class TestProjectMeta:
    def test_documents_every_meta_field(self) -> None:
        doc = document_project(_Dataset)
        assert {m.name for m in doc.dataset.meta} == {
            "file_pattern",
            "short_name",
            "description",
            "references",
        }

    def test_meta_field_specs(self) -> None:
        file_pattern = _meta(document_project(_Dataset), "file_pattern")
        assert file_pattern.description == "Canonical filename pattern"
        assert file_pattern.example == "thing_{date}.nc"
        assert file_pattern.required is True

    def test_optional_meta_field_is_not_required(self) -> None:
        short_name = _meta(document_project(_Dataset), "short_name")
        assert short_name.required is False

    def test_references_field_is_documented(self) -> None:
        references = _meta(document_project(_Dataset), "references")
        assert references.description == "References for the dataset"
        # Project mode carries the spec, not a concrete value.
        assert references.value is None

    def test_concrete_fields_absent_in_project_mode(self) -> None:
        file_pattern = _meta(document_project(_Dataset), "file_pattern")
        assert file_pattern.value is None
        assert file_pattern.derived is None

    def test_dataset_without_meta_has_empty_meta(self) -> None:
        class NoMeta(BaseModel, VocalDatasetMixin):
            attributes: _GlobalAttributes
            variables: list[_Variable]

        assert document_project(NoMeta).dataset.meta == []

    def test_roundtrips_through_json(self) -> None:
        doc = document_project(_Dataset)
        assert ProjectDoc.model_validate_json(doc.model_dump_json()) == doc


class TestProductMeta:
    def test_documents_every_meta_field(self) -> None:
        doc = document_product(_PRODUCT)
        assert {m.name for m in doc.dataset.meta} == {
            "file_pattern",
            "short_name",
            "long_name",
            "description",
            "references",
        }

    def test_concrete_scalar_values(self) -> None:
        doc = document_product(_PRODUCT)
        assert _meta(doc, "file_pattern").value == "thing_{date}.nc"
        assert _meta(doc, "short_name").value == "thing"
        assert _meta(doc, "long_name").value == "A Thing Dataset"
        assert _meta(doc, "description").value.startswith("A concrete product")

    def test_references_value_passes_through(self) -> None:
        references = _meta(document_product(_PRODUCT), "references")
        assert references.value == [
            {"title": "A paper", "doi": "10.1/2"},
            {"title": "A website", "web": "https://example.com"},
        ]
        assert references.derived is False

    def test_rule_bearing_fields_absent_in_product_mode(self) -> None:
        file_pattern = _meta(document_product(_PRODUCT), "file_pattern")
        assert file_pattern.description is None
        assert file_pattern.constraints is None

    def test_roundtrips_through_json(self) -> None:
        doc = document_product(_PRODUCT)
        assert ProductDoc.model_validate_json(doc.model_dump_json()) == doc


# ---------------------------------------------------------------------------
# Diagnostics / doc-lint (slice 8): documentation gaps surfaced non-fatally on
# the project IR's `diagnostics` list while the walk completes normally.
# ---------------------------------------------------------------------------


@vocal_validator(description="", bound=Attribute("title"))
def _undescribed_rule(cls, value):  # pragma: no cover - never invoked
    return value


class TestProjectDiagnostics:
    def test_well_formed_project_has_no_diagnostics(self) -> None:
        assert document_project(_Dataset).diagnostics == []

    def test_undescribed_validator_is_flagged(self) -> None:
        class _BadAttrs(_GlobalAttributes):
            _v_blank = _undescribed_rule

        class _DatasetWithGap(BaseModel, VocalDatasetMixin):
            meta: _DatasetMeta
            attributes: _BadAttrs
            dimensions: list[_Dimension]
            variables: list[_Variable]

        doc = document_project(_DatasetWithGap)
        # Non-fatal: the IR is still produced in full.
        assert isinstance(doc, ProjectDoc)
        assert {a.name for a in doc.dataset.attributes} >= {"title"}
        # ... and the gap is surfaced.
        assert any(
            "_BadAttrs" in d and "empty description" in d for d in doc.diagnostics
        )

    def test_field_name_mixin_mismatch_is_flagged(self) -> None:
        class _PlainAttrs(BaseModel):  # forgot VocalAttributesMixin
            title: str = Field(description="A title", example="x")

        class _DatasetWithMismatch(BaseModel, VocalDatasetMixin):
            attributes: _PlainAttrs

        doc = document_project(_DatasetWithMismatch)
        assert isinstance(doc, ProjectDoc)
        assert any(
            "_PlainAttrs" in d and "VocalAttributesMixin" in d
            for d in doc.diagnostics
        )


# ---------------------------------------------------------------------------
# Groups + recursion (slice 6): project NodeRef/defs vs. product inlining.
# ---------------------------------------------------------------------------


class TestProjectGroups:
    def test_dataset_groups_slot_is_noderef(self) -> None:
        doc = document_project(_Dataset)
        (ref,) = doc.dataset.groups
        assert isinstance(ref, NodeRef)
        assert ref.ref == "Group"

    def test_defs_registers_group_variable_and_dimension_templates(self) -> None:
        doc = document_project(_Dataset)
        assert set(doc.defs) == {"_Variable", "_Dimension", "Group"}
        assert isinstance(doc.defs["Group"], GroupDoc)

    def test_group_template_reuses_attribute_and_variable_walks(self) -> None:
        template = document_project(_Dataset).defs["Group"]
        # Template, so no concrete name.
        assert template.name is None
        assert {a.name for a in template.attributes} == {"group_title"}
        group_title = template.attributes[0]
        assert group_title.description == "The group title"
        assert group_title.required is True
        # References the shared variable/dimension templates rather than
        # re-expanding them in place.
        (var_ref,) = template.variables
        (dim_ref,) = template.dimensions
        assert isinstance(var_ref, NodeRef) and var_ref.ref == "_Variable"
        assert isinstance(dim_ref, NodeRef) and dim_ref.ref == "_Dimension"

    def test_group_recursion_is_noderef_back_to_template(self) -> None:
        template = document_project(_Dataset).defs["Group"]
        (ref,) = template.groups
        assert isinstance(ref, NodeRef)
        assert ref.ref == "Group"

    def test_dataset_without_groups_has_no_group_def(self) -> None:
        class NoGroups(BaseModel, VocalDatasetMixin):
            attributes: _GlobalAttributes
            variables: list[_Variable]

        doc = document_project(NoGroups)
        assert doc.dataset.groups == []
        # The variable template still registers; only the group def is absent.
        assert "Group" not in doc.defs

    def test_roundtrips_through_json(self) -> None:
        doc = document_project(_Dataset)
        assert ProjectDoc.model_validate_json(doc.model_dump_json()) == doc


class TestProductGroups:
    def _group(self, doc, name):
        return next(g for g in doc.dataset.groups if g.name == name)

    def test_groups_are_inlined_as_groupdoc(self) -> None:
        doc = document_product(_PRODUCT)
        nav = self._group(doc, "navigation")
        assert isinstance(nav, GroupDoc)

    def test_group_attributes_and_variables_are_concrete(self) -> None:
        nav = self._group(document_product(_PRODUCT), "navigation")
        assert {a.name for a in nav.attributes} == {"group_title"}
        assert nav.attributes[0].value == "Navigation data"
        assert {v.name for v in nav.variables} == {"latitude"}
        latitude = nav.variables[0]
        assert latitude.datatype == "float64"
        assert {d.name: d.size for d in nav.dimensions} == {"sps": 32}

    def test_nested_group_is_inlined(self) -> None:
        nav = self._group(document_product(_PRODUCT), "navigation")
        (raw,) = nav.groups
        assert isinstance(raw, GroupDoc)
        assert raw.name == "raw"
        assert raw.groups == []

    def test_defs_is_unused_in_product_mode(self) -> None:
        doc = document_product(_PRODUCT)
        assert "defs" not in doc.model_dump()

    def test_roundtrips_through_json(self) -> None:
        doc = document_product(_PRODUCT)
        assert ProductDoc.model_validate_json(doc.model_dump_json()) == doc
