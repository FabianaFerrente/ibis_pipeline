

from datetime import datetime
from typing import Union

def time_to_seconds(time_str: str) -> float:
    """
    Convert a time string (HH:MM:SS or ISO datetime) into total seconds since midnight.

    Parameters
    ----------
    time_str : str
        Time string in "HH:MM:SS" or full ISO "YYYY-MM-DDTHH:MM:SS" format (with optional fractions).

    Returns
    -------
    float
        Total number of seconds since midnight.

    Examples
    --------
    >>> time_to_seconds("01:30:15")
    5415.0
    >>> time_to_seconds("2025-09-26T01:30:15.250")
    5415.25
    >>> time_to_seconds("23:59:59.999")
    86399.999
    """
    
    # 1. Attempt to parse as full ISO format (YYYY-MM-DDTHH:MM:SS)
    if 'T' in time_str:
        try:
            # datetime.fromisoformat handles the full ISO format, including date and time with fractions
            dt_object = datetime.fromisoformat(time_str)
            
            # Extract time components from the datetime object
            h = dt_object.hour
            m = dt_object.minute
            s = dt_object.second
            ms = dt_object.microsecond
            
            # Calculate total seconds since midnight
            return h * 3600 + m * 60 + s + ms / 1e6
        except ValueError:
            # If ISO parsing fails, continue to the next attempt
            pass 

    # 2. Attempt to parse as pure time format (HH:MM:SS)
    try:
        # First, try with fractions of a second (e.g., "01:30:15.250") using %f
        try:
            time_obj = datetime.strptime(time_str, "%H:%M:%S.%f").time()
        except ValueError:
            # If the first attempt fails, try without fractions (e.g., "01:30:15")
            time_obj = datetime.strptime(time_str, "%H:%M:%S").time()
            
        # Extract time components from the resulting time object
        h = time_obj.hour
        m = time_obj.minute
        s = time_obj.second
        ms = time_obj.microsecond
        
        # Calculate total seconds since midnight
        return h * 3600 + m * 60 + s + ms / 1e6

    except ValueError:
        # If all parsing attempts fail, raise a clear error
        raise ValueError(f"Unsupported time format: {time_str}. Expected: HH:MM:SS or YYYY-MM-DDTHH:MM:SS.")


  
def time_from_header(header):
  """Extract time in seconds from DATE-OBS header."""
  timestr = header.get('DATE-OBS', '00:00:00')[11:]  # Es. '15:32:11'
  h, m, s = map(float, timestr.split(':'))
  return h * 3600 + m * 60 + s