"""Scientific nomenclature for the existing SWIFT-C clustering methods.

These are labels for documentation, not new algorithm implementations.
"""

from __future__ import annotations

METHOD_FAMILIES = {
    "Description reference": (
        ("input", "K-means on description embeddings"),
        ("spherical_input", "Spherical K-means on descriptions"),
        ("spectral_input", "Spectral clustering on descriptions"),
        ("eac_classic", "Evidence accumulation on descriptions"),
        ("dec_input", "Deep Embedded Clustering on descriptions"),
    ),
    "Discrete constraint-based supervision": (
        ("discrete_mustlink_kmeans", "K-means with text-identity must-link guidance"),
    ),
    "Representation-based integration": (
        ("labels_only", "K-means on pooled weak-label embeddings"),
        ("fusion", "K-means on fused description and weak-label embeddings"),
        ("spherical_fusion", "Spherical K-means on fused embeddings"),
        ("hdbscan_fusion", "HDBSCAN on fused embeddings"),
    ),
    "Graph-based integration": (
        ("spectral_fusion", "Spectral clustering on fused embeddings"),
        ("eac_multiview", "Multiview evidence accumulation"),
        ("eac_uniform", "Uniform evidence-guided label refinement"),
        ("eac_adaptive", "Adaptive evidence-guided label refinement"),
    ),
    "Latent representation integration": (
        ("dec_fusion", "Deep Embedded Clustering on fused embeddings"),
    ),
}


def format_method_catalogue():
    lines = []
    for family, methods in METHOD_FAMILIES.items():
        lines.append(family)
        lines.extend(f"  {method:26s} {description}" for method, description in methods)
        lines.append("")
    lines.append("The discrete method is implemented by the separate swiftc discrete command.")
    return "\n".join(lines)
