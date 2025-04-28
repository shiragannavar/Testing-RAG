import logging
import sys

def setup_logging(level: int = logging.INFO):
    """
    Configure the root logger to output structured logs to stdout.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
        stream=sys.stdout,
    )