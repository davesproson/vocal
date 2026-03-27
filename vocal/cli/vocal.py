#!/usr/bin/env python

import os
import sys
import typer

from vocal.application import build, check, fetch, init, register, release, web

app = typer.Typer(
    name="vocal",
    help="Compliance checking and metadata management.",
    no_args_is_help=True,
)

app.command("build")(build.command)
app.command("check")(check.command)
app.command("fetch")(fetch.command)
app.command("init")(init.command)
app.command("register")(register.command)
app.command("release")(release.command)
app.command("web")(web.command)


def main() -> None:
    debug = os.environ.get("VOCAL_DEBUG", "false").lower() == "true"
    try:
        app()
    except Exception as e:
        if debug:
            raise
        typer.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
