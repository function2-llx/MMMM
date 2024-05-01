#! /bin/env python

import importlib
import inspect
import logging

from mmmm.data.defs import PROCESSED_DATA_ROOT

from processors._base import Processor

logger: logging.Logger

def setup_logging():
    global logger

    from datetime import datetime
    logging.basicConfig(level=logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    now = datetime.now()
    log_dir = PROCESSED_DATA_ROOT / '.logs' / 'image' / now.strftime("%Y-%m-%d")
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / f'{now.strftime("%H:%M:%S")}.log')
    fh.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]  %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger = logging.getLogger('process')
    logger.addHandler(fh)
    logger.addHandler(ch)

def get_processors() -> dict[str, type[Processor]]:
    module = importlib.import_module('processors')
    return {
        processor_cls.name: processor_cls
        for processor_cls in vars(module).values()
        if inspect.isclass(processor_cls) and issubclass(processor_cls, Processor)
    }

def main():
    processors = {
        processor_cls.name: processor_cls
        for processor_cls in vars(importlib.import_module('processors')).values()
        if inspect.isclass(processor_cls) and issubclass(processor_cls, Processor)
    }
    all_datasets = list(processors.keys())
    from jsonargparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('datasets', nargs='*', type=str)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--exclude', nargs='*', type=str, default=[])
    parser.add_argument('--max_workers', type=int, default=24)
    parser.add_argument('--chunksize', type=int, default=1)
    parser.add_argument('--override', action='store_true')
    parser.add_argument('--limit', type=int | None, default=None)
    parser.add_argument('--empty_cache', action='store_true')
    parser.add_argument('--raise_error', action='store_true')
    args = parser.parse_args()
    if args.all:
        datasets = list(set(all_datasets) - set(args.exclude))
    else:
        datasets = args.datasets
    setup_logging()
    logger.info(datasets)
    for dataset in datasets:
        processor_cls = processors[dataset]
        processor = processor_cls(logger, max_workers=args.max_workers, chunksize=args.chunksize, override=args.override)
        logger.info(f'start processing {dataset}, max_workers={processor.max_workers}, chunksize={processor.chunksize}')
        processor.process(args.limit, args.empty_cache, args.raise_error)

if __name__ == '__main__':
    main()
