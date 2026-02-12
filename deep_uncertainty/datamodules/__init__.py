"""
Datamodule exports.

Some datamodules depend on optional third-party packages (e.g. `wget`, COCO tooling,
transformers, etc.). To keep tabular workflows lightweight, we only require the
tabular datamodule to import unconditionally and make the rest best-effort.
"""

from .tabular_datamodule import TabularDataModule

try:
    from .mnist_datamodule import MNISTDataModule
except Exception:  # pragma: no cover
    MNISTDataModule = None  # type: ignore

try:
    from .coco_people_datamodule import COCOPeopleDataModule
except Exception:  # pragma: no cover
    COCOPeopleDataModule = None  # type: ignore

try:
    from .reviews_datamodule import ReviewsDataModule
except Exception:  # pragma: no cover
    ReviewsDataModule = None  # type: ignore

try:
    from .bible_datamodule import BibleDataModule
except Exception:  # pragma: no cover
    BibleDataModule = None  # type: ignore
