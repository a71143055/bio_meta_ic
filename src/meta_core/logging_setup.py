from rich.console import Console
from rich.theme import Theme

def setup_logging():
    theme = Theme({"info": "cyan", "warning": "yellow", "error": "red"})
    console = Console(theme=theme)
    return console
