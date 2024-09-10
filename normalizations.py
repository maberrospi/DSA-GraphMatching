import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import norm_params


def instance_normalization(feature_matrix, norm_type="zscore"):
    assert norm_type in ["zscore", "minmax"]

    ft_matrix = feature_matrix.copy()

    if norm_type == "zscore":
        z_score = StandardScaler()
        ft_matrix = z_score.fit_transform(ft_matrix)
        # This was used when coords were not included in zscore norm
        # ft_matrix = np.concatenate(
        #     (
        #         ft_matrix[:, :2],
        #         z_score.fit_transform(ft_matrix[:, 2:]),
        #     ),
        #     axis=1,
        # )
    elif norm_type == "minmax":
        min_max = MinMaxScaler()
        ft_matrix = min_max.fit_transform(ft_matrix)

    return ft_matrix


def distribution_normalization(feature_matrix, norm_type="zscore", fts=1):
    assert norm_type in ["zscore", "minmax"]
    assert fts in [1, 2, 3, 4]

    # fts has four ids:
    # 1: Radius and LM fts
    # 2: Coordinates and LM fts
    # 3: Radius and Coordinates
    # 4: Only LM features

    ft_matrix = feature_matrix.copy()

    new_min = norm_params.distr_min
    new_max = norm_params.distr_max
    new_mean = norm_params.distr_mean
    new_std = norm_params.distr_std

    if norm_type == "zscore":

        # The zcore depends on what features are used
        # Since our normalization parameter matrix includes all fts

        match fts:
            case 1:
                # Not Coords and Radius
                ft_matrix = (ft_matrix - new_mean[:, 2:]) / new_std[:, 2:]
            case 2:
                # Coords and NOT Radius
                ft_matrix = np.concatenate(
                    (
                        (ft_matrix[:, :2] - new_mean[:, :2]) / new_std[:, :2],
                        (ft_matrix[:, 2:] - new_mean[:, 3:]) / new_std[:, 3:],
                    ),
                    axis=1,
                )
            case 3:
                # Coords and Radius
                ft_matrix = (ft_matrix - new_mean) / new_std
            case 4:
                # NOT coords NOT radius
                ft_matrix = (ft_matrix - new_mean[:, 3:]) / new_std[:, 3:]

    elif norm_type == "minmax":
        match fts:
            case 1:
                # Not Coords and Radius
                ft_matrix = (ft_matrix - new_min[:, 2:]) / (
                    new_max[:, 2:] - new_min[:, 2:]
                )
            case 2:
                # Coords and NOT Radius
                ft_matrix = np.concatenate(
                    (
                        (ft_matrix[:, :2] - new_min[:, :2])
                        / (new_max[:, :2] - new_min[:, :2]),
                        (ft_matrix[:, 2:] - new_min[:, 3:])
                        / (new_max[:, 3:] - new_min[:, 3:]),
                    ),
                    axis=1,
                )
            case 3:
                # Coords and Radius
                ft_matrix = (ft_matrix - new_min) / (new_max - new_min)
            case 4:
                # NOT coord NOT Radius
                ft_matrix = (ft_matrix - new_min[:, 3:]) / (
                    new_max[:, 3:] - new_min[:, 3:]
                )

    return ft_matrix
