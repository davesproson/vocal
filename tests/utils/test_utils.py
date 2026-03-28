"""
Tests for vocal/utils/__init__.py — focused on dataset_from_partial_yaml.

A minimal set of pydantic models is defined here as test doubles. They use
permissive dict[str, Any] attribute fields so tests can focus on the template
merging behaviour rather than attribute validation constraints.
"""
from pathlib import Path
from typing import Any, Optional

import pytest
import yaml
from pydantic import BaseModel

from vocal.utils import dataset_from_partial_yaml


# ---------------------------------------------------------------------------
# Minimal pydantic test-double models
# ---------------------------------------------------------------------------


class VariableMeta(BaseModel):
    name: str
    datatype: str
    required: bool = True


class Variable(BaseModel):
    meta: VariableMeta
    dimensions: list[str] = []
    attributes: dict[str, Any] = {}


class Dimension(BaseModel):
    name: str
    size: Optional[int] = None


class DatasetMeta(BaseModel):
    file_pattern: str


class Dataset(BaseModel):
    meta: DatasetMeta
    attributes: dict[str, Any] = {}
    dimensions: list[Dimension] = []
    variables: list[Variable] = []
    groups: list[Any] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, content: dict[str, Any]) -> str:
    path = str(tmp_path / "definition.yaml")
    with open(path, "w") as f:
        yaml.dump(content, f)
    return path


MINIMAL_YAML: dict[str, Any] = {
    "meta": {"file_pattern": "test"},
    "attributes": {"title": "My Product"},
    "dimensions": [{"name": "time", "size": None}],
    "variables": [
        {
            "meta": {"name": "time", "datatype": "<float64>"},
            "dimensions": ["time"],
            "attributes": {"units": "seconds since 1970-01-01"},
        }
    ],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDatasetFromPartialYaml:
    def test_returns_model_instance(self, tmp_path: Path) -> None:
        yamlfile = _write_yaml(tmp_path, MINIMAL_YAML)
        result = dataset_from_partial_yaml(yamlfile, {}, {}, {}, Dataset)
        assert isinstance(result, Dataset)

    def test_no_model_raises(self, tmp_path: Path) -> None:
        yamlfile = _write_yaml(tmp_path, MINIMAL_YAML)
        with pytest.raises(ValueError):
            dataset_from_partial_yaml(yamlfile, {}, {}, {}, None)  # type: ignore[arg-type]


class TestGlobalsTemplateMerging:
    def test_yaml_global_attributes_present(self, tmp_path: Path) -> None:
        yamlfile = _write_yaml(tmp_path, MINIMAL_YAML)
        result = dataset_from_partial_yaml(yamlfile, {}, {}, {}, Dataset)
        assert result.attributes["title"] == "My Product"

    def test_globals_template_default_applied(self, tmp_path: Path) -> None:
        yamlfile = _write_yaml(tmp_path, MINIMAL_YAML)
        globals_template = {"institution": "Default Institution"}
        result = dataset_from_partial_yaml(yamlfile, {}, globals_template, {}, Dataset)
        assert result.attributes["institution"] == "Default Institution"

    def test_yaml_overrides_globals_template(self, tmp_path: Path) -> None:
        content = {**MINIMAL_YAML, "attributes": {"title": "YAML Title"}}
        yamlfile = _write_yaml(tmp_path, content)
        globals_template = {"title": "Template Title"}
        result = dataset_from_partial_yaml(yamlfile, {}, globals_template, {}, Dataset)
        assert result.attributes["title"] == "YAML Title"

    def test_globals_template_key_absent_from_yaml_still_present(self, tmp_path: Path) -> None:
        yamlfile = _write_yaml(tmp_path, MINIMAL_YAML)
        globals_template = {"title": "Default Title", "institution": "Default Inst"}
        result = dataset_from_partial_yaml(yamlfile, {}, globals_template, {}, Dataset)
        # YAML has its own title — template's title is overridden
        assert result.attributes["title"] == "My Product"
        # YAML has no institution — template's value is kept
        assert result.attributes["institution"] == "Default Inst"


class TestVariableTemplateMerging:
    def test_yaml_variable_attributes_present(self, tmp_path: Path) -> None:
        yamlfile = _write_yaml(tmp_path, MINIMAL_YAML)
        result = dataset_from_partial_yaml(yamlfile, {}, {}, {}, Dataset)
        time_var = next(v for v in result.variables if v.meta.name == "time")
        assert time_var.attributes["units"] == "seconds since 1970-01-01"

    def test_variable_template_default_applied(self, tmp_path: Path) -> None:
        yamlfile = _write_yaml(tmp_path, MINIMAL_YAML)
        variable_template = {"frequency": 1}
        result = dataset_from_partial_yaml(yamlfile, variable_template, {}, {}, Dataset)
        time_var = next(v for v in result.variables if v.meta.name == "time")
        assert time_var.attributes["frequency"] == 1

    def test_yaml_overrides_variable_template(self, tmp_path: Path) -> None:
        yamlfile = _write_yaml(tmp_path, MINIMAL_YAML)
        variable_template = {"units": "default_units"}
        result = dataset_from_partial_yaml(yamlfile, variable_template, {}, {}, Dataset)
        time_var = next(v for v in result.variables if v.meta.name == "time")
        assert time_var.attributes["units"] == "seconds since 1970-01-01"

    def test_variable_template_applied_to_all_variables(self, tmp_path: Path) -> None:
        content: dict[str, Any] = {
            "meta": {"file_pattern": "test"},
            "attributes": {},
            "dimensions": [],
            "variables": [
                {
                    "meta": {"name": "v1", "datatype": "<float32>"},
                    "dimensions": [],
                    "attributes": {},
                },
                {
                    "meta": {"name": "v2", "datatype": "<float32>"},
                    "dimensions": [],
                    "attributes": {},
                },
            ],
        }
        yamlfile = _write_yaml(tmp_path, content)
        variable_template = {"_FillValue": -9999.0}
        result = dataset_from_partial_yaml(yamlfile, variable_template, {}, {}, Dataset)
        for var in result.variables:
            assert var.attributes["_FillValue"] == -9999.0


class TestGroupTemplateMerging:
    def _yaml_with_group(self, group_attrs: dict[str, Any]) -> dict[str, Any]:
        return {
            "meta": {"file_pattern": "test"},
            "attributes": {},
            "dimensions": [],
            "variables": [],
            "groups": [
                {
                    "meta": {"name": "raw_data"},
                    "attributes": group_attrs,
                    "variables": [],
                }
            ],
        }

    def test_group_template_default_applied(self, tmp_path: Path) -> None:
        yamlfile = _write_yaml(tmp_path, self._yaml_with_group({}))
        group_template = {"comment": "Default group comment"}
        result = dataset_from_partial_yaml(yamlfile, {}, {}, group_template, Dataset)
        assert result.groups[0]["attributes"]["comment"] == "Default group comment"

    def test_yaml_overrides_group_template(self, tmp_path: Path) -> None:
        yamlfile = _write_yaml(tmp_path, self._yaml_with_group({"comment": "YAML comment"}))
        group_template = {"comment": "Template comment"}
        result = dataset_from_partial_yaml(yamlfile, {}, {}, group_template, Dataset)
        assert result.groups[0]["attributes"]["comment"] == "YAML comment"

    def test_group_template_does_not_leak_into_globals(self, tmp_path: Path) -> None:
        yamlfile = _write_yaml(tmp_path, self._yaml_with_group({}))
        group_template = {"group_only_key": "group_value"}
        result = dataset_from_partial_yaml(yamlfile, {}, {}, group_template, Dataset)
        assert "group_only_key" not in result.attributes


class TestConstructMode:
    def test_construct_false_returns_validated_model(self, tmp_path: Path) -> None:
        yamlfile = _write_yaml(tmp_path, MINIMAL_YAML)
        result = dataset_from_partial_yaml(yamlfile, {}, {}, {}, Dataset, construct=False)
        assert isinstance(result, Dataset)

    def test_construct_true_returns_model_instance(self, tmp_path: Path) -> None:
        yamlfile = _write_yaml(tmp_path, MINIMAL_YAML)
        result = dataset_from_partial_yaml(yamlfile, {}, {}, {}, Dataset, construct=True)
        assert isinstance(result, Dataset)

    def test_construct_true_skips_validation(self, tmp_path: Path) -> None:
        # model_construct does not run validators — a required field with a
        # strict validator can be bypassed
        class StrictModel(BaseModel):
            meta: DatasetMeta
            attributes: dict[str, Any] = {}
            dimensions: list[Dimension] = []
            variables: list[Variable] = []
            groups: list[Any] = []

            @classmethod
            def model_validate(cls, obj: Any, **kwargs: Any) -> "StrictModel":
                raise AssertionError("model_validate should not be called in construct mode")

        yamlfile = _write_yaml(tmp_path, MINIMAL_YAML)
        # construct=True should use model_construct, not model_validate
        result = dataset_from_partial_yaml(yamlfile, {}, {}, {}, StrictModel, construct=True)
        assert isinstance(result, StrictModel)
