import inspect
import os

from vocal.utils import import_project
from vocal.validation import Validator


class DatasetUtilsMixin:
    def _get_vocal_project(self):
        dataset_file = inspect.getfile(self.__class__)
        project_dir = os.path.dirname(os.path.dirname(dataset_file))
        return import_project(project_dir)

    def _get_filecodec(self):
        return self._get_vocal_project().filecodec

    def get_filename(self, **kwargs):
        filecodec = self._get_filecodec()

        format_kwargs = {}
        for key, value in filecodec.items():
            try:
                if value["factory"] is None:
                    value["factory"] = lambda x: x
                format_kwargs[key] = value["factory"](kwargs[key])
            except KeyError:
                pass

        missing_key = ""
        try:
            return self.meta.file_pattern.format(**format_kwargs)
        except KeyError as e:
            missing_key = e

        raise KeyError(
            (
                f"File pattern for dataset {self.meta.short_name} requires "
                f"missing key: {missing_key}"
            )
        )

    @property
    def regex(self):
        return self.meta.file_pattern.format(
            **{i: j["regex"] for i, j in self._get_filecodec().items()}
        )


class VocalValidatorsMixin:
    @property
    def validators(self) -> list[Validator]:
        return [
            getattr(self, i)
            for i in [j for j in dir(self) if j.startswith("_validate")]
        ]
