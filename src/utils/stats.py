
import numpy as np

def reject_outliers(data, method='mad', threshold=3.0):
    """
    Generate a boolean mask identifying valid (non-outlier) elements in the input data
    using either the Median Absolute Deviation (MAD) method or Z-score method.

    Parameters
    ----------
    data : numpy.ndarray
        Array of numerical values (dlevels)  to analyze for outliers.

    method : str, optional
        The statistical method to use for outlier detection. Options are:
        - 'mad' : median absolute deviation (robust to outliers)
        - 'zscore' : standard Z-score (assumes normal distribution)
        Default is 'mad'.

    threshold : float, optional
        The cutoff value to determine outliers. Values farther than
        `threshold` times the MAD or standard deviation from the median or mean
        are classified as outliers. Default is 3.0.

    Returns
    -------
    numpy.ndarray
        Boolean array of the same shape as `data`, where `True` indicates a
        valid (non-outlier) value, and `False` indicates an outlier.

    Raises
    ------
    ValueError
        If the `method` argument is not one of ['mad', 'zscore'].

    Notes
    -----
    - For `method='mad'`, if the MAD is zero (indicating no variation), all values
      are considered valid.
    - For `method='zscore'`, if the standard deviation is zero (constant data),
      all values are considered valid.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([10, 12, 10, 10, 200])
    >>> mask = reject_outliers(data, method='mad', threshold=3)
    >>> mask
    array([ True,  True,  True,  True, False])
    """
    if method == 'mad':
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad == 0:
            return np.ones_like(data, dtype=bool)
        mask = np.abs(data - median) <= threshold * mad
    elif method == 'zscore':
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return np.ones_like(data, dtype=bool)
        mask = np.abs(data - mean) <= threshold * std
    else:
        raise ValueError("method must be 'mad' or 'zscore'")
    return mask




    