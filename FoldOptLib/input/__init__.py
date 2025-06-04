from .input_data_checker import CheckInputData

# Import InputDataProcessor lazily so that optional heavy dependencies used in
# this module do not interfere with lightweight parts of the package such as the
# tests for ``CheckInputData``.  This avoids importing ``helper._helper`` at
# package import time.
try:
    from .input_data_processor import InputDataProcessor
except Exception:  # pragma: no cover - optional dependency may be missing
    InputDataProcessor = None
