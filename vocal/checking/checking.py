import json

from dataclasses import dataclass

from vocal.netcdf import NetCDFReader

from .errors import NotCheckedError
from .checks import Check, CheckError, CheckWarning, CheckComment
from .core import compare_container


@dataclass
class CheckReport:
    """
    A class to hold the context of a check report, which may be for a product,
    project, or definition.
    """

    checks: list[Check]

    @property
    def passing(self) -> bool:
        """
        Returns True if all checks have passed, False if any have failed, or
        raises a NotCheckedError if no checks have been carried out
        """
        if not self.checks:
            raise NotCheckedError("Checks have not been performed")

        return all([i.passed for i in self.checks])

    @property
    def warnings(self) -> list[CheckWarning]:
        """
        Returns a list of CheckWarnings for checks which have warning on them, or
        raises a NotCheckedError if no checks have been carried out
        """
        if not self.checks:
            raise NotCheckedError("Checks have not been performed")

        return [i.warning for i in self.checks if i.has_warning and i.warning]

    @property
    def errors(self) -> list[CheckError]:
        """
        Returns a list of CheckErrors for failed checks. Raises a NotCheckedError
        if no checks have been carried out.
        """
        if not self.checks:
            raise NotCheckedError("Checks have not been performed")

        return [i.error for i in self.checks if not i.passed]  # type: ignore

    @property
    def comments(self) -> list[CheckComment]:
        """
        Returns a list of CheckComments for checks which have comments on them, or
        raises a NotCheckedError if no checks have been carried out
        """
        if not self.checks:
            raise NotCheckedError("Checks have not been performed")

        return [i.comment for i in self.checks if i.has_comment and i.comment]


@dataclass
class ProductChecker:
    """
    A class providing methods to check a file against a product definition
    """

    definition: str

    def load_definition(self) -> dict:
        """
        Load the product definition, and return it as a dict.
        """
        with open(self.definition, "r") as f:
            product_def = json.load(f)
        return product_def

    def check(self, target_file: str) -> CheckReport:
        """
        Check a target file against the instance's product specification

        Args:
            target_file: the path of the file to check
        """

        product_def = self.load_definition()
        netcdf_rep = NetCDFReader(target_file).dict

        return CheckReport(checks=compare_container(product_def, netcdf_rep))
