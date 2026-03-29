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
    │ fetch      Fetch a vocal project from a git repository.                   │
    │ init       Initialise a vocal project.                                    │
    │ register   Register a vocal project globally.                             │
    │ release    Create versioned JSON product specifications.                  │
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

Once fetched, a project can be registered globally so that *vocal* can automatically discover
and use it when checking files:

    $ vocal register <project_path> -c <conventions_string>

The conventions string identifies which files the project applies to, e.g. `"MYSTD-[].[]"`.

### Creating a new vocal project

To create a new *vocal* project, simply type `vocal init -d <project_name>`. This will create a
directory named `project_name` with the following structure:

    ./models
    ./models/dimension.py
    ./models/group.py
    ./models/dataset.py
    ./models/__init__.py
    ./models/variable.py
    ./attributes
    ./attributes/variable_attributes.py
    ./attributes/global_attributes.py
    ./attributes/group_attributes.py
    ./attributes/__init__.py
    ./definitions
    ./defaults.py

The models directory contains the pydantic models which define the dataset,
groups, dimensions and variables. The attributes directory contains the pydantic models
for the attributes associated with the dataset (globals), groups and variables.

The definitions directory is the standard location for the working copies of
definitions of individual data products, though this location can be overridden at runtime.

For more information on building *vocal* projects, see the section on [vocal projects](#vocal-projects-details).

## Specifying data products

Data product definitions are specified in YAML files, typically in the `definitions` directory.

An simple example of a product definition may be

    meta:
        file_pattern: "example_data.nc"
        canonical_name: "example_data"
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
            data_type: "<float32>"
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

To create a versioned copy of a data product definition, use the `vocal release` command.

    $ vocal release <project_name> -v <version> -o <output_dir>

This will create a directory named `<output_dir>/<version>` containing the versioned data product definition, as well as a `latest` directory containing a copy of the latest versions. The versioned data product definition is a JSON file, and is intended to be used with the `check` command. Additionally a `dataset_schema.json` file is created, which is a JSON Schema representation of the pydantic model for the dataset, minus any validators.

## Checking data products

*Vocal* can be used to check netCDF files against *vocal* projects and data product definitions. To do this, use the `check` command:

    $ vocal check <file> -p <project_name> -d <definition>

This will check the file against the project and definition specified. If the file is valid, the command will return with exit code 0. If the file is invalid, the command will return with exit code 1. When checking against a product definition, all of the checks will be printed to the console. You can limit the output to warnings and errors only by using the `-w` flag, to errors only by using the `-e` flag, or to no output by using the `-q` flag.

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

If a registered project is found that matches the file's conventions string, the project and any
matching product definitions will be used automatically without needing to pass `-p` or `-d`:

    $ vocal check example_data.nc

    ✔ Found 1 registered project(s) for conventions MYSTD-1.0: example_project
    ✔ Found matching definition: example_product.json

    Checking example_data.nc against example_project standard... OK!

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
