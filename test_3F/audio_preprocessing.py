import numpy as np
from scipy.signal import lfilter
         

def preemphasis(x, alpha):
    y = lfilter(np.array([1, -alpha]), np.array([1, 0]), x)                         
    return y


def segmentation(x, window_length, window_overlap=0, pad=True):
    """
    Parameters
    ----------
    x : (1D numpy array) Input data for segmentation
    window_length : int or array
    window_overlap : (int) optional. The default is 0.
    pad: (bool) optional. Default: True. Zero padding last segment.
        If window_length is an array containing window function then the window is applied on each segment.

    Returns
    -------
    segments : (numpy array) Segmented input data.
    num_seg : (int) number of segments
    """
    
    # Convert input data to numpy array
    try:
        x = np.array(x)
    except ValueError:
        print("unsupported input data type, must be an array")
    
    # Check if input data is one dimensional array
    if x.ndim != 1:
        if np.shape(x)[0]==1 or np.shape(x)[1]==1:
            x = np.ravel(x)
        else:
            raise ValueError("input data must be one dimensional array")

    # Check if window length is window
    if np.ndim(window_length) != 0:
        windowing = True
        window = window_length
        window_length = len(window_length)
    else:
        windowing = False
        window_length = int(window_length)

    window_overlap = int(window_overlap)
    
    # Check if the window overlap is not >= than the window length
    if window_overlap >= window_length:
        raise ValueError("window overlap must be < window length")

    #%% Number of segments    
    num_seg = (len(x)-window_overlap) // (window_length-window_overlap)
    
    #%% Pad last segment with zeros if necessary
    rest_samples = (len(x)-window_overlap) % (window_length-window_overlap)

    if rest_samples != 0 and pad:
        padding = np.zeros(int(window_length - rest_samples))
        x = np.append(x, padding)
        num_seg += 1
    
    #%% Indexes for segmentation process
    # Indexes of samples within a segment  (segments in columns)
    idx1 = np.tile(np.arange(0, window_length), (num_seg, 1)).T
    
    # Indexes of steps (segments in columns)
    seg_step = int(window_length-window_overlap)
    idx2 = np.tile(np.arange(0, num_seg * seg_step, seg_step), (window_length, 1))
    
    indexes = idx1 + idx2
    
    #%% Segmentation
    segments = x[indexes.astype(np.int32, copy=False)]
    
    #%% Weight each segment by window
    if windowing:
        windows = np.tile(window, (num_seg, 1)).T  # windows in columns
        segments *= windows  

    return segments, num_seg





