"""YOLO11-DAL manuscript-faithful reconstruction package."""

from .model import VARIANTS, build_model, variant_summary

__all__ = ["VARIANTS", "build_model", "variant_summary"]
