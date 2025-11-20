
import logging 
import os
import sys
#src_dir = '/Users/giovanna/IBIS/ibis_pipeline_python/ibis_pipeline_definitive/'
#sys.path.insert(0, src_dir)
from src.configs import configuration as config
#from configs import configuration as config
print(f"src_dir: {config.output_Dir}")
def logging_setup():
    """
    Setup logging for the ingestion pipeline.

    Parameters
    ----------
    config : object
        Configuration object containing the logsDir attribute.
    """
    from pathlib import Path
    from datetime import datetime

    def get_current_timestamp():
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Setup logging
    log_file = Path(config.logsDir) / f"{get_current_timestamp()}.log"
    print(f"Log file path: {log_file}")
    # Ensure the logs directory exists
    os.makedirs(config.logsDir, exist_ok=True)
    # Configure logging
    # Set up logging to file
    logging.basicConfig(
        filename=str(log_file),
        format="%(message)s",
        level=logging.INFO,
    )
    logging.info("Starting ingestion pipeline")
    
    
if __name__ == "__main__":
    logging_setup()
    
#  python3 -m  src.utils.logging_config da ibis_pipline_definitive 