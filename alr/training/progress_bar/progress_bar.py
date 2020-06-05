# adapted from https://github.com/BlackHC/progress_bar/blob/master/src/blackhc/progress_bar/progress_bar.py
import abc


class ProgressBar(abc.ABC):
    def __init__(self, length):
        self.length = length

    @abc.abstractmethod
    def start(self):
        pass

    @abc.abstractmethod
    def update(self, delta_processed=1):
        pass

    @abc.abstractmethod
    def finish(self):
        pass

    @abc.abstractmethod
    def log_message(self, msg: str):
        pass