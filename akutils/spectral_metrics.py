# Define a function to calculate normalized difference indices and scale by 10000
def calc_normalized_index(b1, b2, nodata=-32768):
    # Import packages
    import numpy as np
    # Convert to float32 for math operations
    b1_f = b1.astype(np.float32)
    b2_f = b2.astype(np.float32)
    # Ignore divide-by-zero warnings during array math
    with np.errstate(divide='ignore', invalid='ignore'):
        denominator = b1_f + b2_f
        # Calculate index, defaulting to 0 where denominator is 0
        index = np.where(denominator == 0, 0, (b1_f - b2_f) / denominator) * 10000.0
    # Handle NaN values
    index = np.nan_to_num(index, nan=0.0, posinf=10000.0, neginf=-10000.0)
    # Clamp to limits and convert back to int16
    index_int = np.clip(index, -10000, 10000).astype(np.int16)
    # Apply nodata mask where either input band was missing
    mask = (b1 == nodata) | (b2 == nodata)
    index_int[mask] = nodata
    # Return value
    return index_int
