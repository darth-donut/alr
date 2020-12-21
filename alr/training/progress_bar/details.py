# adapted from https://github.com/BlackHC/progress_bar/blob/master/src/blackhc/progress_bar/details.py
import time
import sys
import tqdm

from alr.training.progress_bar.progress_bar import ProgressBar


class TQDMProgressBar(ProgressBar):
    def __init__(self, length, tqdm_args=None):
        super().__init__(length)

        self.progress_bar = None
        self.tqdm_args = tqdm_args or {}

    def start(self):
        if self.progress_bar is not None:
            raise AssertionError("start can only be called once!")

        self.progress_bar = tqdm.tqdm(total=self.length, **self.tqdm_args)

    def update(self, delta_processed=1):
        self.progress_bar.update(delta_processed)

    def finish(self):
        self.progress_bar.close()
        self.progress_bar = None

    def log_message(self, msg: str):
        self.progress_bar.write(msg)


class LogFriendlyProgressBar(ProgressBar):
    num_sections = 10
    last_flush = 0

    def __init__(self, length):
        super().__init__(length)

        self.start_time = None
        self.last_time = None
        self.num_processed = 0
        self.num_finished_sections = 0

    def start(self):
        if self.start_time is not None:
            raise AssertionError("start can only be called once!")

        self.start_time = self.get_time()
        self.last_time = self.start_time

        self.print_header(self.length)

    @staticmethod
    def get_time():
        return time.time()

    def update(self, delta_processed=1):
        self.num_processed += delta_processed

        while (
            self.num_processed
            >= self.length * (self.num_finished_sections + 1) / self.num_sections
        ):
            self.num_finished_sections += 1
            cur_time = self.get_time()
            elapsed_time = cur_time - self.start_time

            expected_time = (
                elapsed_time * self.num_sections / self.num_finished_sections
            )
            remaining_time = expected_time - elapsed_time

            self.print_section(elapsed_time, remaining_time)

            self.last_time = cur_time

        if self.num_finished_sections == self.num_sections:
            total_time = self.last_time - self.start_time + 0.000001
            ips = self.length / total_time

            self.print_finish(ips, total_time)
            self.start_time = None

    def finish(self):
        remaining_elements = self.length - self.num_processed
        if remaining_elements > 0:
            self.update(remaining_elements)

    def log_message(self, msg: str):
        LogFriendlyProgressBar.print(msg, file=sys.stdout)

    @staticmethod
    def print_header(num_iterations):
        LogFriendlyProgressBar.print(f"{num_iterations} iterations:")

        LogFriendlyProgressBar.print(
            "|"
            + "|".join(
                f'{f"{int((index + 1) * 100 / LogFriendlyProgressBar.num_sections)}%":^11}'
                for index in range(LogFriendlyProgressBar.num_sections)
            )
            + "|"
        )
        LogFriendlyProgressBar.print("|", end="")

    @staticmethod
    def print_section(elapsed_time, remaining_time):
        elapsed_time = f"{int(elapsed_time)}s"
        remaining_time = f"{int(remaining_time)}s"
        LogFriendlyProgressBar.print(f"{elapsed_time:<5}<{remaining_time:>5}|", end="")

    @staticmethod
    def print_finish(ips, total_time):
        LogFriendlyProgressBar.print()
        if ips >= 0.1:
            LogFriendlyProgressBar.print(f"{ips:.2f}it/s total: {total_time:.2f}s")
        else:
            LogFriendlyProgressBar.print(f"total: {total_time:.2f}s {1 / ips:.2f}s/it")

    @staticmethod
    def print(text="", end="\n", file=sys.stderr):
        print(text, end=end, file=file)

        cur_time = LogFriendlyProgressBar.get_time()
        if cur_time - LogFriendlyProgressBar.last_flush > 2:
            sys.stdout.flush()
            sys.stderr.flush()
            LogFriendlyProgressBar.last_flush = cur_time
