"""Training utilities for the Incident Commander OpenEnv environment.

This package is installed via the ``[training]`` extras — it pulls in heavy
deps (torch, transformers, unsloth, trl) that the core env server does not
need. Keep this package GPU-only; do not import it from ``server/`` or
``inference.py``.
"""
