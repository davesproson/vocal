import os
import requests

from typing import Protocol
import xml.etree.ElementTree as ET

from vocal.utils import cache_dir


class Vocabulary(Protocol):
    def __contains__(self, word: str) -> bool: ...


def ensure_cache_dir() -> str:
    """
    Return the path to the cache directory, creating it if it does not exist.
    """
    vocab_cache = os.path.join(cache_dir(), "vocabs")
    os.makedirs(vocab_cache, exist_ok=True)
    return vocab_cache


class ListVocabulary:
    def __init__(self, name: str, items: list[str]) -> None:
        self.name = name
        self.items = items

    def __contains__(self, word: str) -> bool:
        return word in self.items

    def __str__(self) -> str:
        return self.name


class CFStandardNames:
    def __init__(self, version: int = 88, allow_alias: bool = True) -> None:
        """
        Create a new instance of the CF Standard Names vocabulary.

        Kwargs:
            version (int): The version of the CF Standard Names vocabulary to use.
            allow_alias (bool): Whether to allow aliases in the vocabulary.
        """
        self.version = version
        self.allow_alias = allow_alias
        self.tree: ET.ElementTree | None = None
        self._load()

    def __str__(self) -> str:
        """
        Return the string representation of the CF Standard Names vocabulary.
        """
        return f"CF Standard Names v{self.version}"

    def _cached_filename(self) -> str:
        """
        Return the filename of the cached CF Standard Names vocabulary file.
        """
        return os.path.join(
            ensure_cache_dir(), f"cf_standard_names_v{self.version}.xml"
        )

    def _load_from_cache(self) -> None:
        """
        Load the CF Standard Names vocabulary from the cache.
        """
        filename = self._cached_filename()
        if filename is None:
            raise FileNotFoundError("No cached CF Standard Names vocabulary found.")

        self.tree = ET.parse(filename)

    def _load_from_remote(self) -> None:
        """
        Load the CF Standard Names vocabulary from the web.
        """
        url = f"https://cfconventions.org/Data/cf-standard-names/{self.version}/src/cf-standard-name-table.xml"
        response = requests.get(url)
        response.raise_for_status()
        with open(self._cached_filename(), "wb") as f:
            f.write(response.content)
        self._load_from_cache()

    def _load(self) -> None:
        """
        Load the CF Standard Names vocabulary from the cache or the web.
        """
        try:
            self._load_from_cache()
        except FileNotFoundError:
            self._load_from_remote()

    def __contains__(self, word: str) -> bool:
        """
        Return whether the CF Standard Names vocabulary includes the given word.

        Args:
            word (str): The word to check.

        Returns:
            bool: Whether the CF Standard Names vocabulary includes the word.
        """
        if self.tree is None:
            return False

        root = self.tree.getroot()
        if root is None:
            return False

        for entry in root.findall("entry"):
            if isinstance(entry, ET.Element):
                std_name = entry.get("id")
                if std_name == word:
                    return True

        if not self.allow_alias:
            return False

        for alias in root.findall("alias"):
            if isinstance(alias, ET.Element):
                std_name = alias.get("id")
                if std_name == word:
                    return True

        return False

    def canonical_units(self, word: str) -> str | None:
        """
        Return the canonical units for the given word.

        Args:
            word (str): The word to check.

        Returns:
            str: The canonical units for the given word.
        """
        if self.tree is None:
            return ""

        root = self.tree.getroot()
        if root is None:
            return None

        for entry in root.findall("entry"):
            if isinstance(entry, ET.Element):
                std_name = entry.get("id")
                if std_name == word:
                    units = entry.find("canonical_units")
                    if units is not None:
                        return getattr(units, "text", None)

        return None
