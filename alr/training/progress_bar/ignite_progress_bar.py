import ignite
from alr.training.progress_bar import create_progress_bar
import sys
from typing import Callable, Optional


# from https://github.com/BlackHC/BatchBALD/blob/master/src/ignite_progress_bar.py
class ProgressBar:
    def __init__(self, desc: Callable = None, log_interval: Optional[int] = 10):
        r"""
        Creates a smart progress bar (tqdm if in notebooks, text if in terminal).
        Use `alr.training.progress_bar.use_tqdm = True/False` to force TQDM (or force disable it).

        Args:
            desc (Callable, optional): takes an engine as input and returns a string
            log_interval (int, optional): log every `log_interval` iterations
        """
        self.log_interval = log_interval
        self.desc = desc
        self.progress_bar = None

    def attach(self, engine: ignite.engine.Engine):
        engine.add_event_handler(ignite.engine.Events.EPOCH_STARTED, self.on_start)
        engine.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, self.on_complete)
        engine.add_event_handler(ignite.engine.Events.ITERATION_COMPLETED, self.on_iteration_complete)

    def on_start(self, engine):
        dataloader = engine.state.dataloader
        self.progress_bar = create_progress_bar(
            len(dataloader)
        )
        if self.desc is not None:
            print(self.desc(engine), file=sys.stderr, end='')
        self.progress_bar.start()

    def on_complete(self, _):
        self.progress_bar.finish()

    def on_iteration_complete(self, engine):
        dataloader = engine.state.dataloader
        iters = (engine.state.iteration - 1) % len(dataloader) + 1
        if iters % self.log_interval == 0:
            self.progress_bar.update(
                self.log_interval
            )

    def log_message(self, msg: str):
        self.progress_bar.log_message(msg)
