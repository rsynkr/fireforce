"""Application state assembled once at startup."""

from __future__ import annotations

import logging

from app.config import Settings
from app.services.inference import InferenceService
from app.services.model_registry import ModelRegistry
from app.services.sample_catalog import SampleCatalog, SampleCatalogError, load_catalog

logger = logging.getLogger(__name__)


class AppState:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.catalog: SampleCatalog | None = None
        self.catalog_error: str | None = None

        try:
            self.catalog = load_catalog(
                str(settings.manifest_path), str(settings.data_root)
            )
        except SampleCatalogError as exc:
            self.catalog_error = str(exc)
            logger.warning("sample catalog unavailable: %s", exc)

        self.registry = ModelRegistry(
            str(settings.checkpoint_dir), device=settings.device
        )
        for model_id, status in self.registry.statuses().items():
            if not status.ready:
                logger.warning("%s unavailable: %s", model_id, status.error)

        self.inference = (
            InferenceService(self.catalog, self.registry)
            if self.catalog is not None
            else None
        )
