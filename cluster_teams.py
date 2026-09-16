"""
Cluster players into 2 teams using SigLIP embeddings (from
siglip_embeddings_colab.ipynb). Runs entirely on CPU, cheap.

No scikit-learn or umap-learn dependency - pure numpy k-means. (Some
Windows machines, especially managed/hostel ones, block the compiled DLLs
those libraries ship with via an "Application Control" security policy -
this version sidesteps that entirely.)

Usage:
    python cluster_teams.py --embeddings embeddings.parquet --out team_assignments.csv
    python cluster_teams.py --embeddings embeddings.parquet --out team_assignments.csv --manifest crops/manifest.csv
        (passing --manifest lets you check clustering accuracy against the
        known-true labels from the placeholder crop generator; skip it once
        you're using real crops with no ground truth to check against)
"""

import argparse

import numpy as np
import pandas as pd


def kmeans(X, n_clusters, seed, n_init=10, max_iter=100):
    """Minimal numpy k-means. Runs n_init random restarts, keeps the one
    with lowest inertia (sum of squared distances to assigned centroid),
    same idea as sklearn's KMeans(n_init=...) without the dependency."""
    rng = np.random.default_rng(seed)
    best_labels, best_inertia = None, np.inf

    for _ in range(n_init):
        centroid_idx = rng.choice(len(X), size=n_clusters, replace=False)
        centroids = X[centroid_idx].copy()

        for _ in range(max_iter):
            dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            labels = dists.argmin(axis=1)
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
                for k in range(n_clusters)
            ])
            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids

        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = dists.argmin(axis=1)
        inertia = ((X - centroids[labels]) ** 2).sum()

        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels

    return best_labels


def cluster(embeddings_path, n_clusters, seed):
    df = pd.read_parquet(embeddings_path)
    track_ids = df["track_id"].values
    X = df.drop(columns=["track_id"]).values.astype(np.float64)

    # Standardize so no single dimension dominates the distance calc.
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    labels = kmeans(X, n_clusters, seed)
    return pd.DataFrame({"track_id": track_ids, "predicted_team_id": labels})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", required=True, help="embeddings.parquet from the Colab notebook")
    ap.add_argument("--out", default="team_assignments.csv")
    ap.add_argument("--n-clusters", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", default=None, help="optional manifest.csv with true_team_id, to check accuracy")
    args = ap.parse_args()

    result = cluster(args.embeddings, args.n_clusters, args.seed)
    result.to_csv(args.out, index=False)
    print(f"Wrote {len(result)} team assignments -> {args.out}")
    print(result.to_string(index=False))

    if args.manifest:
        manifest = pd.read_csv(args.manifest)[["track_id", "true_team_id"]]
        merged = result.merge(manifest, on="track_id")
        match_a = (merged["predicted_team_id"] == merged["true_team_id"]).mean()
        match_b = (merged["predicted_team_id"] == (1 - merged["true_team_id"])).mean()
        accuracy = max(match_a, match_b)
        print(f"\nAccuracy vs. true_team_id (label-swap invariant): {accuracy:.1%}")