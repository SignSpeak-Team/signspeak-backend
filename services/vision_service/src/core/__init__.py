"""Core module for SignSpeak Vision Service."""

# Lazy imports to avoid loading heavy dependencies when only partial imports needed
__all__ = ["SignPredictor", "get_predictor", "WordBuffer", "WordDetection"]


def __getattr__(name):
    """Lazy loading of submodules."""
    if name in ("SignPredictor", "get_predictor"):
        from core.predictor import SignPredictor, get_predictor

        return SignPredictor if name == "SignPredictor" else get_predictor
    elif name in ("WordBuffer", "WordDetection"):
        from core.word_buffer import WordBuffer, WordDetection

        return WordBuffer if name == "WordBuffer" else WordDetection
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
