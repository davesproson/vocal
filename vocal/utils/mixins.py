from vocal.validation import Validator


class DatasetUtilsMixin:
    """Filename helpers for a dataset model.

    The ``filecodec`` that expands a product's templated ``file_pattern`` no
    longer lives in the project — it moved to the pack's ``pack.yaml`` — so it is
    passed in explicitly by the caller (which sources it from the pack /
    definitions side) rather than imported from the project package.
    """

    def get_filename(self, filecodec, **kwargs):
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

    def regex(self, filecodec):
        return self.meta.file_pattern.format(
            **{i: j["regex"] for i, j in filecodec.items()}
        )


class VocalValidatorsMixin:
    @property
    def validators(self) -> list[Validator]:
        return [
            getattr(self, i)
            for i in [j for j in dir(self) if j.startswith("_validate")]
        ]
