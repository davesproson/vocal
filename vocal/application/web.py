"""Launch a web-based checker GUI."""

import time
import threading
import webbrowser

import typer
import uvicorn


def command(
    host: str = typer.Option("127.0.0.1", "--host", help="The host to bind to."),
    port: int = typer.Option(8088, "--port", help="The port to bind to."),
) -> None:
    """Launch a web-based checker GUI."""

    def _start_browser_in_thread(host: str, port: int) -> None:
        time.sleep(1)
        webbrowser.open_new_tab(f"http://{host}:{port}")

    threading.Thread(target=_start_browser_in_thread, args=(host, port)).start()
    uvicorn.run(
        "vocal.web:app", host=host, port=port, log_level="info", reload=True
    )
