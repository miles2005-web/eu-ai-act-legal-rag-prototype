"""EU AI Act product-regulation reference data foundation."""

from src.assessment.product_regulation.catalog import (
    AmbiguousAnnexIInstrumentAliasError,
    AnnexIInstrumentAliasNotFoundError,
    AnnexIInstrumentCatalog,
    AnnexIInstrumentCatalogError,
    AnnexIInstrumentNotFoundError,
    DEFAULT_ANNEX_I_CATALOG_PATH,
    InvalidProductRegulationFactsError,
    load_annex_i_instrument_catalog,
    normalize_annex_i_alias,
    validate_product_regulation_facts,
)
from src.assessment.product_regulation.models import (
    AnnexIInstrument,
    AnnexIInstrumentType,
    AnnexISection,
)

__all__ = [
    "AmbiguousAnnexIInstrumentAliasError",
    "AnnexIInstrument",
    "AnnexIInstrumentAliasNotFoundError",
    "AnnexIInstrumentCatalog",
    "AnnexIInstrumentCatalogError",
    "AnnexIInstrumentNotFoundError",
    "AnnexIInstrumentType",
    "AnnexISection",
    "DEFAULT_ANNEX_I_CATALOG_PATH",
    "InvalidProductRegulationFactsError",
    "load_annex_i_instrument_catalog",
    "normalize_annex_i_alias",
    "validate_product_regulation_facts",
]
