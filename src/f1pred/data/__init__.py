"""Stratul de date: încărcare offline din cache-ul FastF1 + asamblarea tabelului lung."""

from .build_dataset import get_results_long, build_results_long

__all__ = ["get_results_long", "build_results_long"]
