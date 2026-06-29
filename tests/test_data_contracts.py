from __future__ import annotations

from ripgnn.data.contracts import ArtifactBundle, validate_bundle
from ripgnn.data.synthetic import build_edges, build_region_meta, build_snapshots


def test_demo_bundle_satisfies_contracts() -> None:
    bundle = ArtifactBundle(
        snapshots=build_snapshots(),
        edges=build_edges(),
        region_meta=build_region_meta(),
    )
    validate_bundle(bundle)

