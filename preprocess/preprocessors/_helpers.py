"""
Contains specific modules that help in preprocessing the data,
such as common calcium trace data transformations, necessary for the
base and specific preprocessors.
"""

from preprocess._pkg import torch, np, pd, interp1d, NEURON_LABELS, logger, NUM_NEURONS


###############################################################################################
# TODO: Encapsulate smoothing functions in OOP style class.
def gaussian_kernel_smooth(x, t, sigma):
    """Causal Gaussian smoothing for a multidimensional time series.

    Parameters:
        x (ndarray): The input time series to be smoothed (time, neurons).
        t (ndarray): The time vector (in seconds) corresponding to the input time series.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        x_smooth (ndarray): The smoothed time series.

    Steps:
        1. Check if the input is a torch.Tensor and convert to numpy if necessary.
        2. Reshape the input if it is 1-dimensional.
        3. Initialize the smoothed time series array.
        4. Compute the Gaussian weights for each time point.
        5. Apply the Gaussian smoothing to each time point and feature.
        6. Convert the smoothed time series back to torch.Tensor if the input was a tensor.
    """
    istensor = isinstance(x, torch.Tensor)
    if istensor:
        x = x.cpu().numpy()
    dim = x.ndim
    if dim == 1:
        x = x.reshape(-1, 1)
    # Apply one-sided exponential decay
    x_smooth = np.zeros_like(x, dtype=np.float32)
    alpha = 1 / (2 * sigma**2)
    # TODO: Vectorize this instead of using a loop.
    for i in range(x.shape[0]):  # temporal dimension
        weights = np.exp(-alpha * np.arange(i, -1, -1) ** 2)
        weights /= weights.sum()
        for j in range(x.shape[1]):  # feature dimension
            x_smooth[i, j] = np.dot(weights, x[: i + 1, j])
    if dim == 1:
        x_smooth = x_smooth.squeeze(-1)
    if istensor:
        x_smooth = torch.from_numpy(x_smooth)
    return x_smooth


def moving_average_smooth(x, t, window_size):
    """Causal moving average smoothing filter for a multidimensional time series.

    Parameters:
        x (ndarray): The input time series to be smoothed.
        t (ndarray): The time vector (in seconds) corresponding to the input time series.
        window_size (int): The size of the moving average window. Must be an odd number.

    Returns:
        x_smooth (ndarray): The smoothed time series.

    Steps:
        1. Ensure window_size is odd for symmetry.
        2. Check for correct dimensions and convert to torch.Tensor if necessary.
        3. Initialize the smoothed time series array.
        4. Apply the moving average smoothing to each time point and feature.
        5. Convert the smoothed time series back to numpy.ndarray if the input was a numpy array.
    """
    # Ensure window_size is odd for symmetry
    if window_size % 2 == 0:
        window_size += 1
    # Check for correct dimensions
    isnumpy = isinstance(x, np.ndarray)
    if isnumpy:
        x = torch.from_numpy(x)
    dim = x.ndim
    if dim == 1:
        x = x.unsqueeze(-1)
    x_smooth = torch.zeros_like(x)
    # TODO: Vectorize this instead of using a loop.
    for i in range(x.shape[1]):  # feature dimension
        for j in range(x.shape[0]):  # temporal dimension
            start = max(j - window_size // 2, 0)
            end = j + 1
            window = x[start:end, i]
            x_smooth[j, i] = window.mean()
    if dim == 1:
        x_smooth = x_smooth.squeeze(-1)
    if isnumpy:
        x_smooth = x_smooth.cpu().numpy()
    return x_smooth


def exponential_kernel_smooth(x, t, alpha):
    """Exponential kernel smoothing for a multidimensional time series.
    This method is already causal by its definition.

    Parameters:
        x (ndarray): The input time series to be smoothed (time, neurons).
        t (ndarray): The time vector (in seconds) corresponding to the input time series.
        alpha (float): The smoothing factor, 0 < alpha < 1. A higher value of alpha will
                       result in less smoothing (more weight is given to the current value),
                       while a lower value of alpha will result in more smoothing
                       (more weight is given to the previous smoothed values).

    Returns:
        x_smooth (ndarray): The smoothed time series.

    Steps:
        1. Check if the input is a torch.Tensor and convert to numpy if necessary.
        2. Reshape the input if it is 1-dimensional.
        3. Initialize the smoothed time series array.
        4. Apply the exponential smoothing to each time point and feature.
        5. Convert the smoothed time series back to torch.Tensor if the input was a tensor.
    """
    istensor = isinstance(x, torch.Tensor)
    if istensor:
        x = x.cpu().numpy()
    dim = x.ndim
    if dim == 1:
        x = x.reshape(-1, 1)
    x_smooth = np.zeros_like(x, dtype=np.float32)
    x_smooth[0] = x[0]
    # TODO: Vectorize this smoothing operation
    for i in range(1, x.shape[0]):
        x_smooth[i] = alpha * x[i] + (1 - alpha) * x_smooth[i - 1]

    if dim == 1:
        x_smooth = x_smooth.squeeze(-1)
    if istensor:
        x_smooth = torch.from_numpy(x_smooth)
    return x_smooth


def smooth_data_preprocess(calcium_data, time_in_seconds, smooth_method, **kwargs):
    """Smooths the provided calcium data using the specified smoothing method.

    Parameters:
        calcium_data (np.ndarray): Original calcium data with shape (time, neurons).
        time_in_seconds (np.ndarray): Time vector with shape (time, 1).
        smooth_method (str): The method used to smooth the data. Options are "gaussian", "moving", "exponential".

    Returns:
        smooth_ca_data (np.ndarray): Calcium data that is smoothed.

    Steps:
        1. Check if the smooth_method is None, and if so, return the original calcium data.
        2. If the smooth_method is "gaussian", apply Gaussian kernel smoothing.
        3. If the smooth_method is "moving", apply moving average smoothing.
        4. If the smooth_method is "exponential", apply exponential kernel smoothing.
        5. Raise a TypeError if the smooth_method is not recognized.
    """
    if smooth_method is None:
        smooth_ca_data = calcium_data
    elif str(smooth_method).lower() == "gaussian":
        smooth_ca_data = gaussian_kernel_smooth(
            calcium_data, time_in_seconds, sigma=kwargs.get("sigma", 5)
        )
    elif str(smooth_method).lower() == "moving":
        smooth_ca_data = moving_average_smooth(
            calcium_data, time_in_seconds, window_size=kwargs.get("window_size", 15)
        )
    elif str(smooth_method).lower() == "exponential":
        smooth_ca_data = exponential_kernel_smooth(
            calcium_data, time_in_seconds, alpha=kwargs.get("alpha", 0.5)
        )
    elif str(smooth_method).lower() == "none":
        smooth_ca_data = calcium_data
    else:
        raise TypeError(
            "See `preprocess/config.py` for viable smooth methods."
        )
    return smooth_ca_data


###############################################################################################


def reshape_calcium_data(worm_dataset):
    """Reorganizes calcium data into a standard organized matrix with shape (max_timesteps, NUM_NEURONS).
    Also creates neuron masks and mappings of neuron labels to indices in the data.
    Converts the data to torch tensors.

    Parameters:
        worm_dataset (dict): Dataset for a single worm that includes calcium data and other information.

    Returns:
        dict: The modified worm dataset with restructured calcium data.

    Steps:
        1. Initialize the CalciumDataReshaper with the provided worm dataset.
        2. Return the reshaped worm dataset.
    """
    reshaper = CalciumDataReshaper(worm_dataset)
    return reshaper.worm_dataset


def interpolate_data(time, data, target_dt, method="linear"):
    """
    Interpolate data using scipy's interp1d or np.interp.

    This function interpolates the given data to the desired time intervals.

    Parameters:
        time (numpy.ndarray): 1D array containing the time points corresponding to the data.
        data (numpy.ndarray): A 2D array containing the data to be interpolated, with shape (time, neurons).
        target_dt (float): The desired time interval between the interpolated data points.
        method (str, optional): The interpolation method to use. Default is 'linear'.

    Returns:
        numpy.ndarray, numpy.ndarray: Two arrays containing the interpolated time points and data.
    """
    # Check if correct interpolation method provided
    assert method in {
        None,
        "linear",
        "quadratic",
        "cubic",
    }, (
        "Invalid interpolation method. Choose from [None, 'linear', 'cubic', 'quadratic']."
    )
    assert time.shape[0] == data.shape[0], "Input temporal dimension mismatch."
    # If target_dt is None, return the original data
    if target_dt is None:
        return time, data
    # Ensure that time is a 1D array
    time = time.squeeze()
    # Create the target time vector, ensuring the range does not exceed the original data range
    target_time_np = np.arange(time.min(), time.max(), target_dt)
    num_neurons = data.shape[1]
    interpolated_data_np = np.zeros(
        (len(target_time_np), num_neurons), dtype=np.float32
    )
    # Use scipy's interpolation method
    # TODO: Vectorize this operation.
    if method == "linear":
        for i in range(num_neurons):
            interpolated_data_np[:, i] = np.interp(target_time_np, time, data[:, i])
    else:
        logger.info(
            "Warning: scipy.interplate.interp1d is deprecated. Best to choose method='linear'."
        )
        for i in range(num_neurons):
            interp = interp1d(
                x=time,
                y=data[:, i],
                kind=method,
                bounds_error=False,
                fill_value="extrapolate",
            )
            interpolated_data_np[:, i] = interp(target_time_np)
    # Reshape interpolated time vector to (time, 1)
    target_time_np = target_time_np.reshape(-1, 1)
    # Final check for shape consistency
    assert target_time_np.shape[0] == interpolated_data_np.shape[0], (
        "Output temporal dimension."
    )
    # Return the interpolated time and data
    return target_time_np, interpolated_data_np


def aggregate_data(time, data, target_dt):
    """
    Downsample data using aggregation.

    This function downsamples the data by averaging over intervals of size `target_dt`.

    Parameters:
        time (numpy.ndarray): 1D array containing the time points corresponding to the data.
        data (numpy.ndarray): A 2D array containing the data to be downsampled, with shape (time, neurons).
        target_dt (float): The desired time interval between the downsampled data points.

    Returns:
        numpy.ndarray, numpy.ndarray: Two arrays containing the downsampled time points and data.
    """
    assert time.shape[0] == data.shape[0], "Input temporal dimension."
    # If target_dt is None, return the original data
    if target_dt is None:
        return time, data
    # Ensure that time is a 1D array
    time = time.squeeze()
    # Compute the downsample rate
    original_dt = np.median(np.diff(time, axis=0)[1:]).item()
    interval_width = max(1, int(np.round(target_dt / original_dt)))
    num_intervals = len(time) // interval_width
    # Create the downsampled time array
    target_time_np = target_dt * np.arange(num_intervals)
    # Create the downsampled data array
    num_neurons = data.shape[1]
    downsampled_data = np.zeros((num_intervals, num_neurons), dtype=np.float32)
    # Downsample the data by averaging over intervals
    # TODO: Vectorize this operation.
    for i in range(num_neurons):
        reshaped_data = data[: num_intervals * interval_width, i].reshape(
            num_intervals, interval_width
        )
        downsampled_data[:, i] = reshaped_data.mean(axis=1)
    # Reshape downsampled time vector to (time, 1)
    target_time_np = target_time_np.reshape(-1, 1)
    # Final check for shape consistency
    assert target_time_np.shape[0] == downsampled_data.shape[0], (
        "Output temporal dimension mismatch."
    )
    # Return the interpolated data
    return target_time_np, downsampled_data


# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


# # # Class definitions # # #
# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
class CausalNormalizer:
    """
    A transform for causal normalization of time series data.

    This normalizer computes the mean and standard deviation up to each time point t,
    ensuring that the normalization at each time point is based solely on past
    and present data, maintaining the causal nature of the time series.

    Attributes:
        nan_fill_method (str): Method to fill NaN values. Options are 'ffill' (forward fill),
                               'bfill' (backward fill), and 'interpolate'. Default is 'interpolate'.
        cumulative_mean_ (np.ndarray): Cumulative mean up to each time point.
        cumulative_std_ (np.ndarray): Cumulative standard deviation up to each time point.

    Methods:
        fit(X, y=None):
            Compute the cumulative mean and standard deviation of the dataset X.
        transform(X):
            Perform causal normalization on the dataset X using the previously computed cumulative mean and standard deviation.
        fit_transform(X, y=None):
            Fit to data, then transform it.
    """

    def __init__(self, nan_fill_method="interpolate"):
        """
        Initialize the CausalNormalizer with a method to handle NaN values.

        Parameters:
            nan_fill_method (str): Method to fill NaN values. Options are 'ffill' (forward fill),
                                   'bfill' (backward fill), and 'interpolate'. Default is 'interpolate'.
        """
        self.cumulative_mean_ = None
        self.cumulative_std_ = None
        self.nan_fill_method = nan_fill_method

    def _handle_nans(self, X):
        """
        Handle NaN values in the dataset X based on the specified method.

        Parameters:
            X (array-like): The input data with potential NaN values.

        Returns:
            X_filled (array-like): The data with NaN values handled.
        """
        df = pd.DataFrame(X)
        if self.nan_fill_method == "ffill":
            df.fillna(method="ffill", inplace=True)
        elif self.nan_fill_method == "bfill":
            df.fillna(method="bfill", inplace=True)
        elif self.nan_fill_method == "interpolate":
            df.interpolate(method="linear", inplace=True)
        else:
            raise ValueError("Invalid NaN fill method specified.")
        return df.values

    def fit(self, X, y=None):
        """
        Compute the cumulative mean and standard deviation of the dataset X.
        Uses the two-pass algorithm: https://www.wikiwand.com/en/Algorithms_for_calculating_variance.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data.
            y (Ignored): Not used, present here for API consistency by convention.

        Returns:
            self : object
                Returns the instance itself.
        """
        X = self._handle_nans(X)
        T, D = X.shape

        count = np.arange(1, T + 1).reshape(-1, 1)
        cumulative_sum = np.cumsum(X, axis=0)
        cumulative_squares_sum = np.cumsum(X**2, axis=0)

        self.cumulative_mean_ = cumulative_sum / count
        cumulative_variance = (cumulative_squares_sum / count) - (
            self.cumulative_mean_**2
        )
        self.cumulative_std_ = np.sqrt(
            np.maximum(cumulative_variance, 1e-8)
        )  # Avoid sqrt(0)

        # # Avoid zero-division
        # self.cumulative_std_[self.cumulative_std_ == 0] = 1

        assert self.cumulative_mean_.shape == (T, D), (
            "cumulative_mean is not shape of input!"
        )
        assert self.cumulative_std_.shape == self.cumulative_mean_.shape, (
            "cumulative_std and cumulative_mean shapes don't match"
        )

        return self

    def transform(self, X):
        """
        Perform causal normalization on the dataset X using the
        previously computed cumulative mean and standard deviation.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data.

        Returns:
            X_transformed (array-like of shape (n_samples, n_features)): The transformed data.
        """
        if self.cumulative_mean_ is None or self.cumulative_std_ is None:
            raise RuntimeError("The transformer has not been fitted yet.")
        X_transformed = (X - self.cumulative_mean_) / self.cumulative_std_
        return X_transformed

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data to fit and transform.
            y (Ignored): Not used, present here for API consistency by convention.

        Returns:
            X_transformed (array-like of shape (n_samples, n_features)): The transformed data.
        """
        return self.fit(X).transform(X)


class CalciumDataReshaper:
    """
    Reshapes and organizes calcium imaging data for a single worm.

    This class takes a dataset for a single worm and reorganizes the calcium data into a standard
    matrix with shape (max_timesteps, NUM_NEURONS). It also creates neuron masks and mappings of
    neuron labels to indices in the data, and converts the data to torch tensors.

    Attributes:
        worm_dataset (dict): Dataset for a single worm that includes calcium data and other information.
        labeled_neuron_to_idx (dict): Mapping of labeled neurons to their indices.
        unlabeled_neuron_to_idx (dict): Mapping of unlabeled neurons to their indices.
        slot_to_labeled_neuron (dict): Mapping of slots to labeled neurons.
        slot_to_unlabeled_neuron (dict): Mapping of slots to unlabeled neurons.
        slot_to_neuron (dict): Mapping of slots to neurons.
        dtype (torch.dtype): Data type for the tensors.

    Methods:
        _init_neuron_data():
            Initializes attributes from keys that must already be present in the worm dataset.
        _reshape_data():
            Reshapes the calcium data and updates the worm dataset.
        _prepare_initial_data():
            Prepares initial data structures for reshaping.
        _init_empty_calcium_data():
            Initializes empty calcium data matrices.
        _tensor_time_data():
            Converts time data to torch tensors.
        _fill_labeled_neurons_data():
            Fills data for labeled neurons.
        _fill_calcium_data(idx, slot):
            Fills calcium data for a given neuron index and slot.
        _fill_unlabeled_neurons_data():
            Fills data for unlabeled neurons.
        _update_worm_dataset():
            Updates the worm dataset with reshaped data and mappings.
        _remove_old_mappings():
            Removes old mappings from the worm dataset.
    """

    def __init__(self, worm_dataset: dict):
        """
        Initialize the CalciumDataReshaper with the provided worm dataset.

        Parameters:
            worm_dataset (dict): Dataset for a single worm that includes calcium data and other information.

        NOTE:
            'idx' refers to the index of the neuron in the original dataset.
                0 < idx < N, where N is however many neurons were recorded.
            'slot' refers to the index of the neuron in the reshaped dataset.
                0 < slot < NUM_NEURONS, the number of neurons in hermaphrodite C. elegans.
        """
        self.worm_dataset = worm_dataset
        self.labeled_neuron_to_idx = dict()
        self.unlabeled_neuron_to_idx = dict()
        self.slot_to_labeled_neuron = dict()
        self.slot_to_unlabeled_neuron = dict()
        self.slot_to_neuron = dict()
        self.dtype = torch.half
        self._init_neuron_data()
        self._reshape_data()

    def _init_neuron_data(self):
        """Initializes attributes from keys that must already be present in the worm dataset
        at the time that the reshaper is invoked.
        Therefore, those keys specify what is required for a worm dataset to be valid.
        """
        # Post-processed data / absolutely necessary keys
        self.time_in_seconds = self.worm_dataset["time_in_seconds"]
        self.dt = self.worm_dataset["dt"]
        self.max_timesteps = self.worm_dataset["max_timesteps"]
        self.median_dt = self.worm_dataset["median_dt"]
        self.calcium_data = self.worm_dataset["calcium_data"]

        # Only CausalNormalizer tracks these
        self.original_cumulative_mean = self.worm_dataset.get("cumulative_mean", None)
        self.original_cumulative_std = self.worm_dataset.get("cumulative_std", None)
        # for future, clearer reference
        self.using_causalnormalizer = (
            self.original_cumulative_mean is not None
            and self.original_cumulative_std is not None
        )

        self.smooth_calcium_data = self.worm_dataset["smooth_calcium_data"]
        self.residual_calcium = self.worm_dataset["residual_calcium"]
        self.smooth_residual_calcium = self.worm_dataset["smooth_residual_calcium"]
        self.neuron_to_idx = self.worm_dataset["neuron_to_idx"]
        self.idx_to_neuron = self.worm_dataset["idx_to_neuron"]
        self.extra_info = self.worm_dataset.get("extra_info", dict())
        # Original data / optional keys that may be inferred
        self.original_time_in_seconds = self.worm_dataset.get(
            "original_time_in_seconds", self.worm_dataset["time_in_seconds"]
        )
        self.original_dt = self.worm_dataset.get("original_dt", self.worm_dataset["dt"])
        self.original_max_timesteps = self.worm_dataset.get(
            "original_max_timesteps", self.worm_dataset["max_timesteps"]
        )
        self.original_calcium_data = self.worm_dataset.get(
            "original_calcium_data", self.worm_dataset["calcium_data"]
        )
        self.original_median_dt = self.worm_dataset.get(
            "original_median_dt", self.worm_dataset["median_dt"]
        )
        self.original_smooth_calcium_data = self.worm_dataset.get(
            "original_smooth_calcium_data", self.worm_dataset["smooth_calcium_data"]
        )
        self.original_residual_calcium = self.worm_dataset.get(
            "original_residual_calcium", self.worm_dataset["residual_calcium"]
        )
        self.original_smooth_residual_calcium = self.worm_dataset.get(
            "original_smooth_residual_calcium",
            self.worm_dataset["smooth_residual_calcium"],
        )

    def _reshape_data(self):
        """
        Reshapes the calcium data and updates the worm dataset.
        This method performs the following steps:
            1. Prepares the initial data with `_prepare_initial_data()`.
            2. Fills the labeled neurons data with `_fill_labeled_neurons_data()`.
            3. Fills the unlabeled neurons data with `_fill_unlabeled_neurons_data()`.
            4. Updates the worm dataset with reshaped data with `_update_worm_dataset()`.
            5. Removes mappings with 'idx' no longer needed with `_remove_old_mappings()`.
        """
        self._prepare_initial_data()
        self._fill_labeled_neurons_data()
        self._fill_unlabeled_neurons_data()
        self._update_worm_dataset()
        self._remove_old_mappings()

    def _prepare_initial_data(self):
        """
        Prepares initial data structures for reshaping.
        Step 1 of reshape_data.
        """
        assert len(self.idx_to_neuron) == self.calcium_data.shape[1], (
            "Number of neurons in calcium data matrix does not match number of recorded neurons."
        )
        self.labeled_neurons_mask = torch.zeros(NUM_NEURONS, dtype=torch.bool)
        self.unlabeled_neurons_mask = torch.zeros(NUM_NEURONS, dtype=torch.bool)
        self._init_empty_calcium_data()
        self._tensor_time_data()

    def _init_empty_calcium_data(self):
        """
        Initializes empty calcium data matrices to be used for creating matrices
        of a fixed size (to avoid issueswhen e.g. some neurons weren't measured)

        Happens in prepare_inital_data -- the first step of reshaping.
        """
        # Resampled data
        self.standard_calcium_data = torch.zeros(
            self.max_timesteps, NUM_NEURONS, dtype=self.dtype
        )
        self.standard_residual_calcium = torch.zeros(
            self.max_timesteps, NUM_NEURONS, dtype=self.dtype
        )
        self.standard_smooth_calcium_data = torch.zeros(
            self.max_timesteps, NUM_NEURONS, dtype=self.dtype
        )
        self.standard_residual_smooth_calcium = torch.zeros(
            self.max_timesteps, NUM_NEURONS, dtype=self.dtype
        )
        
        if self.using_causalnormalizer:
            self.standard_cumulative_mean = torch.zeros(
                self.max_timesteps, NUM_NEURONS, dtype=self.dtype
            )
            self.standard_cumulative_std = torch.zeros(
                self.max_timesteps, NUM_NEURONS, dtype=self.dtype
            )
        
        # Raw data
        self.standard_original_calcium_data = torch.zeros(
            self.original_max_timesteps, NUM_NEURONS, dtype=self.dtype
        )
        self.standard_original_smooth_calcium_data = torch.zeros(
            self.original_max_timesteps, NUM_NEURONS, dtype=self.dtype
        )
        self.standard_original_residual_calcium = torch.zeros(
            self.original_max_timesteps, NUM_NEURONS, dtype=self.dtype
        )
        self.standard_original_smooth_residual_calcium = torch.zeros(
            self.original_max_timesteps, NUM_NEURONS, dtype=self.dtype
        )

    def _tensor_time_data(self):
        """
        Converts time data to torch tensors.
        """
        # Resampled data
        self.time_in_seconds = (
            self.time_in_seconds - self.time_in_seconds[0]
        )  # start at 0.0 seconds
        self.dt = np.diff(self.time_in_seconds, axis=0, prepend=0.0)
        self.median_dt = np.median(self.dt[1:]).item()
        self.time_in_seconds = torch.from_numpy(self.time_in_seconds).to(self.dtype)
        if self.time_in_seconds.ndim == 1:
            self.time_in_seconds = self.time_in_seconds.unsqueeze(-1)
        self.dt = torch.from_numpy(self.dt).to(self.dtype)
        if self.dt.ndim == 1:
            self.dt = self.dt.unsqueeze(-1)
        # Raw data
        self.original_time_in_seconds = (
            self.original_time_in_seconds - self.original_time_in_seconds[0]
        )  # start at 0.0 seconds
        self.original_dt = np.diff(self.original_time_in_seconds, axis=0, prepend=0.0)
        self.original_median_dt = np.median(self.original_dt[1:]).item()
        self.original_time_in_seconds = torch.from_numpy(
            self.original_time_in_seconds
        ).to(self.dtype)
        if self.original_time_in_seconds.ndim == 1:
            self.original_time_in_seconds = self.time_in_seconds.unsqueeze(-1)
        self.original_dt = torch.from_numpy(self.original_dt).to(self.dtype)
        if self.original_dt.ndim == 1:
            self.original_dt = self.original_dt.unsqueeze(-1)

    def _fill_labeled_neurons_data(self):
        """
        Fills data for labeled neurons.
        Step 2 of reshape_data.
        """
        # slot is the index of the neuron as we have mapped them in NEURON_LABELS
        for slot, neuron in enumerate(NEURON_LABELS):
            if neuron in self.neuron_to_idx:  # labeled neuron
                idx = self.neuron_to_idx[neuron]
                self.labeled_neuron_to_idx[neuron] = idx
                self._fill_calcium_data(idx, slot)
                self.labeled_neurons_mask[slot] = True
                self.slot_to_labeled_neuron[slot] = neuron

    def _fill_calcium_data(self, idx, slot):
        """
        Fills calcium data for a given neuron index and slot.
        Called as part of fill_labeled_neurons_data for each standard neuron.

        Parameters:
            idx (int): Index of the neuron in the original dataset.
            slot (int): Slot in the reshaped dataset.
        """
        self.standard_calcium_data[:, slot] = torch.from_numpy(
            self.calcium_data[:, idx]
        ).to(self.dtype)
        self.standard_residual_calcium[:, slot] = torch.from_numpy(
            self.residual_calcium[:, idx]
        ).to(self.dtype)
        self.standard_smooth_calcium_data[:, slot] = torch.from_numpy(
            self.smooth_calcium_data[:, idx]
        ).to(self.dtype)
        self.standard_residual_smooth_calcium[:, slot] = torch.from_numpy(
            self.smooth_residual_calcium[:, idx]
        ).to(self.dtype)
        # Raw data
        self.standard_original_calcium_data[:, slot] = torch.from_numpy(
            self.original_calcium_data[:, idx]
        ).to(self.dtype)
        self.standard_original_smooth_calcium_data[:, slot] = torch.from_numpy(
            self.original_smooth_calcium_data[:, idx]
        ).to(self.dtype)
        self.standard_original_residual_calcium[:, slot] = torch.from_numpy(
            self.original_residual_calcium[:, idx]
        ).to(self.dtype)
        self.standard_original_smooth_residual_calcium[:, slot] = torch.from_numpy(
            self.original_smooth_residual_calcium[:, idx]
        ).to(self.dtype)

        if self.using_causalnormalizer:
            self.standard_cumulative_mean[:, slot] = torch.from_numpy(
                self.original_cumulative_mean[:, idx]
            ).to(self.dtype)
            self.standard_cumulative_std[:, slot] = torch.from_numpy(
                self.original_cumulative_std[:, idx]
            ).to(self.dtype)
            
    def _fill_unlabeled_neurons_data(self):
        """
        Fills data for unlabeled neurons.
        Step 3 of reshape_data.
        """
        free_slots = list(np.where(~self.labeled_neurons_mask)[0])
        for neuron in set(self.neuron_to_idx) - set(self.labeled_neuron_to_idx):
            self.unlabeled_neuron_to_idx[neuron] = self.neuron_to_idx[neuron]
            slot = np.random.choice(free_slots)
            free_slots.remove(slot)
            self.slot_to_unlabeled_neuron[slot] = neuron
            self._fill_calcium_data(self.neuron_to_idx[neuron], slot)
            self.unlabeled_neurons_mask[slot] = True

    def _update_worm_dataset(self):
        """
        Updates the worm dataset with reshaped data and mappings.
        Step 4 of reshape_data.
        """
        self.slot_to_neuron.update(self.slot_to_labeled_neuron)
        self.slot_to_neuron.update(self.slot_to_unlabeled_neuron)
        self.worm_dataset.update(
            {
                "calcium_data": self.standard_calcium_data,  # normalized, resampled
                "dt": self.dt,  # resampled (vector)
                "idx_to_labeled_neuron": {
                    v: k for k, v in self.labeled_neuron_to_idx.items()
                },
                "idx_to_unlabeled_neuron": {
                    v: k for k, v in self.unlabeled_neuron_to_idx.items()
                },
                "median_dt": self.median_dt,  # resampled (scalar)
                "labeled_neuron_to_idx": self.labeled_neuron_to_idx,
                "labeled_neuron_to_slot": {
                    v: k for k, v in self.slot_to_labeled_neuron.items()
                },
                "labeled_neurons_mask": self.labeled_neurons_mask,
                "neuron_to_slot": {v: k for k, v in self.slot_to_neuron.items()},
                "neurons_mask": self.labeled_neurons_mask | self.unlabeled_neurons_mask,
                "original_calcium_data": self.standard_original_calcium_data,  # original, normalized
                "original_dt": self.original_dt,  # original (vector)
                "original_median_dt": self.original_median_dt,  # original (scalar)
                "original_residual_calcium": self.standard_original_residual_calcium,  # original
                "original_smooth_calcium_data": self.standard_original_smooth_calcium_data,  # original, normalized, smoothed
                "original_smooth_residual_calcium": self.standard_original_smooth_residual_calcium,  # original, smoothed
                "original_time_in_seconds": self.original_time_in_seconds,  # original
                "residual_calcium": self.standard_residual_calcium,  # resampled
                "smooth_calcium_data": self.standard_smooth_calcium_data,  # normalized, smoothed, resampled
                "smooth_residual_calcium": self.standard_residual_smooth_calcium,  # smoothed, resampled
                "slot_to_labeled_neuron": self.slot_to_labeled_neuron,
                "slot_to_neuron": self.slot_to_neuron,
                "slot_to_unlabeled_neuron": self.slot_to_unlabeled_neuron,
                "time_in_seconds": self.time_in_seconds,  # resampled
                "unlabeled_neuron_to_idx": self.unlabeled_neuron_to_idx,
                "unlabeled_neuron_to_slot": {
                    v: k for k, v in self.slot_to_unlabeled_neuron.items()
                },
                "unlabeled_neurons_mask": self.unlabeled_neurons_mask,
                "extra_info": self.extra_info,
            }
        )

        if self.using_causalnormalizer:
            self.worm_dataset.update(
                {
                    "cumulative_mean": self.standard_cumulative_mean,
                    "cumulative_std": self.standard_cumulative_std,
                }
            )

    def _remove_old_mappings(self):
        """
        Removes old mappings from the worm dataset.
        Step 5 (final step) of reshape_data.
        """
        keys_to_delete = [key for key in self.worm_dataset if "idx" in key]
        for key in keys_to_delete:
            self.worm_dataset.pop(key, None)
