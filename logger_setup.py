import logging
from logging.handlers import RotatingFileHandler


def get_logger(name, log_file="app.log", console_level=logging.WARNING):
    """Return a logger that writes to the console and to a rotating file.

    - Console shows only console_level and above (default WARNING),
      so your menu stays clean.
    - The file records everything (DEBUG and above).
    - The file rotates at about 1 MB and keeps 3 old copies.
    """
    logger = logging.getLogger(name)

    # Already set up
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


if __name__ == "__main__":
    log = get_logger("test", "test.log")
    log.debug("Only in the file")
    log.info("Only in the file")
    log.warning("On screen and in the file")
    log.error("On screen and in the file")
