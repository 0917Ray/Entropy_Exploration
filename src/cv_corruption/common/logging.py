import logging


def configure(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s %(message)s")
