# Vocal

Vocal is a tool for managing netCDF data product standards and associated data product specifications.
It is intended to be used with datasets following the [Climate-Forecast Conventions](https://cfconventions.org/),
but may also be used non cf-compliant datasets.

## Dependencies

*Vocal* requires the [udunits2](https://www.unidata.ucar.edu/software/udunits/) C library to be
installed on your system. On Debian/Ubuntu-based systems this can be installed with:

    sudo apt install libudunits2-dev

On macOS with Homebrew:

    brew install udunits

## Installation

### With uv (recommended)

The recommended way to install *vocal* is with [uv](https://docs.astral.sh/uv/):

    uv tool install git+https://github.com/FAAM-146/vocal.git

This makes the `vocal` command available globally. Alternatively, to use *vocal* in a project:

    uv add git+https://github.com/FAAM-146/vocal.git

### With pip

*Vocal* can also be installed with pip:

    pip install git+https://github.com/FAAM-146/vocal.git

Note that if using pip directly, it is **strongly** recommended that you use a python environment
manager such as [Virtualenv](https://pypi.org/project/virtualenv/).

Once installed, the `vocal` command should be available in your `PATH`:

    $ vocal

     Usage: vocal [OPTIONS] COMMAND [ARGS]...

     Compliance checking and metadata management.

    ╭─ Commands ────────────────────────────────────────────────────────────────╮
    │ build      Create an example data file from a definition.                 │
    │ check      Check a netCDF file against standard and product definitions.  │
    │ fetch      Fetch a vocal project or pack and register it.                 │
    │ init       Initialise a vocal project.                                    │
    │ register   Register a vocal project or pack globally.                     │
    │ release    Produce a pack with a manifest, v{Y}/, and latest/.           │
    │ web        Launch a web-based checker GUI.                                │
    ╰───────────────────────────────────────────────────────────────────────────╯

## Vocal projects

*Vocal* uses *vocal projects* to define standards for netCDF data. *Vocal* projects are comprised of
[pydantic](https://docs.pydantic.dev/) model definitions, and associated validators. *Vocal*
then provides a mapping from netCDF data to these models, allowing the power of pydantic to
be used for compliance checking.

Typically as a data provider you will be provided with a *vocal* project to use to check your
data for compliance.

### Obtaining a vocal project

The simplest way to obtain a *vocal* project is with the `fetch` command:

    $ vocal fetch <url>

where `<url>` is the URL of the git repository containing the project. For private repositories or
repositories hosted outside of GitHub, pass the `--git` flag to use git directly:

    $ vocal fetch --git <url>

### Registering a vocal project

Fetching a project registers it automatically. To register a project (or pack) you already have
on disk, point `register` at it:

    $ vocal register <path>

`register` auto-detects the kind of resource from its marker file — a `conventions.yaml` at the
path is a project, a `manifest.json` is a pack — and registers it under the correct key. There is
no conventions-string flag: a project's identity (its name and version, e.g. `MYSTD-1.0`) comes
from its `conventions.yaml`. Pass `-f`/`--force` to overwrite an existing registration.

### Creating a new vocal project

To create a new *vocal* project, type `vocal init -n <NAME>`, where `<NAME>` is the standard's
name (e.g. `MYSTD`). By default the project is scaffolded in the current directory; pass
`-d <directory>` to scaffold it elsewhere, and `--major` / `--minor` to set the standard's
version (defaulting to `1` / `0`). This writes a `conventions.yaml` recording the standard's
identity and module layout, plus an importable Python package named after the standard
(lower-cased, or overridden with `-p`/`--project-directory`):

    ./conventions.yaml
    ./mystd/__init__.py
    ./mystd/defaults.py
    ./mystd/models/__init__.py
    ./mystd/models/dimension.py
    ./mystd/models/variable.py
    ./mystd/models/group.py
    ./mystd/models/dataset.py
    ./mystd/attributes/__init__.py
    ./mystd/attributes/global_attributes.py
    ./mystd/attributes/group_attributes.py
    ./mystd/attributes/variable_attributes.py

The `models` directory contains the pydantic models which define the dataset,
groups, dimensions and variables. The `attributes` directory contains the pydantic models
for the attributes associated with the dataset (globals), groups and variables.

Product definitions are conventionally kept in a `definitions` directory alongside the project
(see [Specifying data products](#specifying-data-products)), though this location can be
overridden at runtime.

## Specifying data products

Data product definitions are specified in YAML files, typically in the `definitions` directory.

An simple example of a product definition may be

    meta:
        file_pattern: "example_data.nc"
        short_name: "example_data"
        description: "An example data product"
        references:
            - ["Reference 1", "https://example.com"]
            - ["Reference 2", "https://example.com"]
    attributes:
        Conventions: "CF-1.8"
        title: "Example data"
        comment: <str: derived_from_file optional>
    dimensions:
        - name: time
          size: null # null indicates unlimited dimension
        - name: height
          size: 32
    variables:
        - meta:
            name: "example_variable"
            datatype: "<float32>"
            required: true
        attributes:
            long_name: "Example variable"
            units: "m"
            comment: <str: derived_from_file optional>
        dimensions:
            - time
            - height

This definition specifies a single required variable, `example_variable`, with dimensions `time` and `height`. Attributes may be literal values, or may be a placeholder indicating
that the value may change between files. In this case, the `comment` attribute is derived from the file. A typical attribute placeholder is `<str: derived_from_file optional>`, which indicates that the attribute is a string, and that it is optional. Array-valued attributes are also supported, for example `<Array[int8]: derived_from_file optional>` indicates that the attribute is an array of 8-bit integers, and is optional.

### Versioning data product definitions

The 'working' copy of a data product definition is typically stored in the `definitions` directory. However, it is possible that a data product definition may change over time. For example, a new version of a standard may be released, or a data product may be updated to include new variables. In this case, it is useful to be able to track the changes between versions of a data product definition.

To create a versioned release of a set of data product definitions, use the `vocal release` command:

    $ vocal release -p <project_path> -v <version> -u <pack_repo_url> -o <output_dir>

This produces a **pack**: a self-describing, independently releasable catalogue of product definitions. The command writes a `v<version>/` directory containing the versioned product definitions, plus a byte-identical `latest/` directory holding a copy of the most recent release. Each product definition is a JSON file intended to be used with the `check` command, and each release directory carries a `manifest.json` recording the pack's identity and the standard it requires. Additionally a `dataset_schema.json` file is created, which is a JSON Schema representation of the pydantic model for the dataset, minus any validators.

The `-u`/`--url` value is the pack's **GitHub repository URL** — the repository you will publish the pack from. It is recorded in every `manifest.json` and is the identity consumers fetch the pack by (see [Packs](#packs)). On the first release in a fresh output directory `--url` is required; on subsequent releases it falls back to the URL recorded in `<output>/latest/manifest.json`, and supplying a different URL is a deliberate, explicit operation.

Publishing a pack is a normal git workflow: commit the `v<version>/` and `latest/` tree and cut a GitHub release from the repository. `vocal release` only produces the files locally; it does not create the GitHub release for you.

## Packs

A **pack** is a versioned, self-describing catalogue of product definitions, produced with `vocal release` (see [Versioning data product definitions](#versioning-data-product-definitions)). Where a *project* defines the standard, a *pack* holds the concrete product definitions authored against that standard, and is published and consumed independently of it.

### Hosting a pack

Packs are hosted on **GitHub**, exactly like projects. A pack repository is a multi-version monorepo: it keeps every release's `v{Y}/` directory plus a `latest/` copy, so a single repository carries the full version history. To publish, commit the tree produced by `vocal release` and cut a GitHub release from the repository — the release's source archive then contains every version.

A pack's identity is its GitHub repository URL, recorded by `vocal release --url` into every `manifest.json`. There is no separate static-hosting URL; the repository *is* the pack.

### Fetching a pack

Obtain a pack with the same `fetch` command used for projects:

    $ vocal fetch <pack-repo-url>

By default this downloads the pack repository's **latest GitHub release** and registers it. For private repositories, non-GitHub hosts, or repositories with no published release, clone the repository directly with `--git`:

    $ vocal fetch --git <pack-repo-url>

`vocal fetch` auto-detects whether a URL points at a project or a pack by inspecting the downloaded tree (a `conventions.yaml` at the root is a project; a `latest/manifest.json` is a pack), so you never have to tell it which kind of resource you are fetching. If a fetched repository is neither, the command reports a clear error.

A single fetch registers **every** version (`v{Y}`) the pack contains — you can then validate files authored against any historical version without re-fetching. The `latest/` directory is a hosting artifact only and is not registered separately; "latest" is simply the highest registered version.

Fetching a pack that is already registered is gated to avoid silently clobbering what you have:

- a plain `vocal fetch <pack-repo-url>` on an already-registered pack reports that it is already fetched and hints at `--update` / `--force`;
- `vocal fetch --update <pack-repo-url>` picks up newly released versions and refreshes existing ones. Update is **additive** — it never removes a version you have already registered;
- `vocal fetch --force <pack-repo-url>` re-installs every version in the latest release regardless of what is registered, repairing a corrupted or partial install.

### How packs are used at check time

When you check a file, *vocal* routes it to the right pack using two global attributes on the file: `vocal_definitions_url` (the pack's GitHub repository URL) and `vocal_definitions_version` (the `v{Y}` release the file was authored against). The pack must already be fetched; if the named URL or version is not registered, `vocal check` reports a `PackMissing` error hinting at the `vocal fetch <pack-repo-url>` you need to run.

`vocal_definitions_version` is **optional**. When it is present, the file is checked against that exact registered version. When it is absent (but `vocal_definitions_url` is present), *vocal* falls back to the **highest registered version** for that pack.

> **Latest-version caveat.** With `vocal_definitions_version` omitted, a file is validated against the locally newest registered version, which may differ from the version it was actually authored against. The version attribute is the precise pin; its absence means "latest". For reproducible checks, pin the version explicitly.

## Checking data products

*Vocal* can be used to check netCDF files against *vocal* projects and data product definitions. To do this, use the `check` command:

    $ vocal check <file> -p <project_name> -d <definition>

This will check the file against the project and definition specified. If the file is valid, the command will return with exit code 0. If the file is invalid, the command will return with exit code 1. When checking against a product definition, all of the checks will be printed to the console. You can limit the output to warnings and errors only by using the `-w` flag, to errors only by using the `-e` flag, or to no output by using the `-q` flag. Comments are hidden by default; pass `-c`/`--comments` to show them. Use `--no-color` to disable coloured output.

For example,

    $ vocal check <file> -p <project_name> -d <definition> -e

will check the file against the project and definition specified, and will only print errors to the console.

A file can also be checked only against a project, without a data product definition:

    $ vocal check <file> -p <project_name>

For example, to check a data file against a project standard:

    $ vocal check example_data.nc -p example_project

    Checking example_data.nc against example_project standard... OK!

Any errors will be printed to the console, indicating where in the file the error occurred, the reason for the error, and potentially the validator that failed.

    $ vocal check example_data.nc -p example_project

    Checking example_data.nc against example_project standard... ERROR!
    ✗ root -> groups -> instrument_group_1 -> attributes -> instrument_name: field required

If you omit `-p`, *vocal* resolves the project automatically from the file's `Conventions`
attribute, matching it against the registered projects. When the file also carries the
`vocal_definitions_url` (and optionally `vocal_definitions_version`) attributes, the matching
product definition is resolved from the corresponding registered pack as well, so neither `-p`
nor `-d` is needed (see [Packs](#packs)):

    $ vocal check example_data.nc

    Checking example_data.nc against MYSTD-1 standard... OK!

If no registered project matches the file's conventions, or the named pack/version is not
fetched, `check` reports a typed error explaining what to fetch or register.

## Checking data products via the web interface

*Vocal* includes a web-based checker GUI that can be launched with the `web` command:

    $ vocal web

    INFO:     Started server process [12345]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://127.0.0.1:8088 (Press CTRL+C to quit)

The host and port can be configured with the `--host` and `--port` options.

## Creating example data

*Vocal* can be used to create example data files from *vocal* projects and data product definitions. To do this, use the `build` command:

    $ vocal build -p <project_name> -d <definition> -o <output_file>

This will create a netCDF file with sinusoidal data for each variable in the data product definition.
