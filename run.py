from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ordering_similarity import treePenalizedPathLength


def read_distance_file(filepath: str) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """
    Read a pairwise dissimilarity file.

    Parameters
    ----------
    filepath : str
        Path to a whitespace-delimited file with three columns:
        query identifier, target identifier, and dissimilarity value.

    Returns
    -------
    id_list : list of str
        Sorted list of unique identifiers found in the file.

    distances : dict of dict of float
        Nested dictionary of pairwise dissimilarity values, where
        ``distances[a][b]`` gives the dissimilarity between identifiers
        ``a`` and ``b``. The dictionary is symmetric and includes
        diagonal values of 0.0.
    """
    distances: Dict[str, Dict[str, float]] = {}
    id_set = set()

    with open(filepath) as f:
        next(f)  
        for line in f:
            parts = line.strip().split()

            if not parts:
                continue
            if len(parts) != 3:
                raise ValueError(
                    f"Expected 3 columns, got {len(parts)}: {line.strip()}"
                )

            query_id, target_id, value = parts
            value = float(value)

            id_set.add(query_id)
            id_set.add(target_id)

            distances.setdefault(query_id, {})[target_id] = value
            distances.setdefault(target_id, {})[query_id] = value

    id_list = sorted(id_set)

    for id_ in id_list:
        distances.setdefault(id_, {})[id_] = 0.0

    return id_list, distances


def order_similarity(
    id_list: List[str],
    distances: Dict[str, Dict[str, float]],
) -> List[Tuple[int, str]]:
    """
    Order identifiers by similarity using tree-penalized path length.

    Parameters
    ----------
    id_list : list of str
        List of identifiers to order.

    distances : dict of dict of float
        Nested dictionary containing pairwise dissimilarity values, where
        ``distances[a][b]`` gives the dissimilarity between identifiers
        ``a`` and ``b``.

    Returns
    -------
    ids_ordered : list of tuple
        List of tuples of the form ``(index, identifier)``, where the
        index is the position in the similarity-based ordering.
    """
    n_ids = len(id_list)
    dist = np.zeros((n_ids, n_ids), dtype=float)

    for i, id1 in enumerate(id_list):
        row = distances[id1]
        for j, id2 in enumerate(id_list):
            dist[i, j] = row[id2]

    disc_order = treePenalizedPathLength(dist, 10, 39873)
    ids_ordered = [(idx, id_list[order]) for idx, order in enumerate(disc_order)]

    return ids_ordered


def build_heatmap_matrix(
    distances: Dict[str, Dict[str, float]],
    ids_ordered: List[Tuple[int, str]],
) -> Tuple[List[List[float]], List[str]]:
    """
    Construct a pairwise dissimilarity matrix ordered by similarity.

    Parameters
    ----------
    distances : dict of dict of float
        Nested dictionary containing pairwise dissimilarity values between
        identifiers, where ``distances[a][b]`` gives the dissimilarity
        between identifiers ``a`` and ``b``.

    ids_ordered : list of tuple
        List of tuples representing the similarity-based ordering of
        identifiers, where each tuple is of the form ``(index, identifier)``.

    Returns
    -------
    matrix : list of list of float
        Square matrix of pairwise dissimilarity values arranged according
        to the order specified in ``ids_ordered``.

    ordered_ids : list of str
        Identifiers corresponding to the rows and columns of the matrix.
    """
    ordered_ids = [id_ for _, id_ in ids_ordered]
    matrix = [
        [distances[row_id][col_id] for col_id in ordered_ids]
        for row_id in ordered_ids
    ]
    return matrix, ordered_ids


def plot_heatmap(
    data_matrix: List[List[float]],
    row_labels: List[str],
    column_labels: List[str] | None = None,
) -> None:
    """
    Plot a heatmap of pairwise dissimilarity values.

    Parameters
    ----------
    data_matrix : list of list of float
        Two-dimensional matrix of dissimilarity values to display.

    row_labels : list of str
        Labels for heatmap rows.

    column_labels : list of str, optional
        Labels for heatmap columns. If not provided, ``row_labels`` are
        used for both rows and columns.

    Returns
    -------
    None
        Displays the heatmap.
    """
    data = np.array(data_matrix, dtype=float)

    if column_labels is None:
        column_labels = row_labels

    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad(color="lightgray")

    masked_data = np.ma.masked_invalid(data)

    _, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(masked_data, cmap=cmap, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(column_labels, rotation=90)
    ax.set_yticklabels(row_labels)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Dis-similarity")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    id_list, distance_dict = read_distance_file(
        "zernike_scores_full_atom_prediction.txt"
    )
    ids_ordered = order_similarity(id_list, distance_dict)
    heatmap_matrix, ordered_ids = build_heatmap_matrix(distance_dict, ids_ordered)
    plot_heatmap(heatmap_matrix, ordered_ids)