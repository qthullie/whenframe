"""Find the moment you are looking for in your own videos, on your own machine."""

from .encoder import Encoder
from .library import Library, Settings
from .search import Hit, SearchConfig, Searcher

__all__ = ["Encoder", "Hit", "Library", "SearchConfig", "Searcher", "Settings"]
__version__ = "0.1.0"
