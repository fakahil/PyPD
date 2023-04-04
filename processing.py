import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Executor
class Thread:
    """
    # Thread
    The threads holds the information on the function to execute in a thread or process.
    Provides an interface to the `future` object once submitted to an executer.
    """

    def __init__(self, func, args):
        self.function = func
        self.arguments = args
        self.future = None

    def submit(self, executor: Executor):
        """Start execution via executor"""
        if not self.is_submitted():
            self.future = executor.submit(self.function, self.arguments)
        return self

    def is_submitted(self) -> bool:
        return self.future is not None

    def is_done(self):
        return self.is_submitted() and self.future.done()

    def exception(self):
        if not self.is_done():
            return None
        return self.future.exception()

    def result(self):
        if not self.is_submitted():
            return None
        return self.future.result()

class MP:
    """
    ## MP Multi-Processing
    Class provides housekeeping / setup methods to reduce the programming overhead of
    spawning threads or processes.
    """

    #: Number of CPUs of the current machine
    NUM_CPUs = round(os.cpu_count() * 0.8)

    @staticmethod
    def threaded(func, args, workers=10, raise_exception=True):
        """
        Calls the given function in multiple threads for the set of given arguments
            Note that this does not spawn processes, but threads. Use this for non CPU
            CPU dependent tasks, i.e. I/O
        Method returns once all calls are done.

        ### Params
        - func: [Function] the function to call
        - args: [Iterable] the 'list' of arguments for each call
        - workers: [Integer] the number of concurrent threads to use
        - raise_exception: [Bool] Flag if an exception in a thread shall be raised or just logged

        ### Returns
        Results from all `Threads` as list
        """
        if len(args) == 1:
            return list(func(arg) for arg in args)

        with ThreadPoolExecutor(workers) as ex:
            threads = [Thread(func, arg).submit(ex) for arg in args]
        return MP.collect_results(threads, raise_exception)

    @staticmethod
    def simultaneous(func, args, workers=None, raise_exception=True):
        """
        Calls the given function in multiple processes for the set of given arguments
            Note that this does spawn processes, not threads. Use this for task that
            depend heavily on CPU and can be done in parallel.
        Method returns once all calls are done.

        ### Params
        - func: [Function] the function to call
        - args: [Iterable] the 'list' of arguments for each call
        - workers: [Integer] the number of concurrent threads to use (Default: NUM_CPUs)
        - raise_exception: [Bool] Flag if an exception in a thread shall be raised or just logged

        ### Returns
        Results from all `Threads` as list
        """
        if len(args) == 1:
            return list(func(arg) for arg in args)

        if workers is None:
            workers = MP.NUM_CPUs
        with ProcessPoolExecutor(workers) as ex:
            threads = [Thread(func, arg).submit(ex) for arg in args]
        return MP.collect_results(threads, raise_exception)

    @staticmethod
    def collect_results(threads: list, raise_exception: bool = True) -> list:
        """
        Takes a list of threads and waits for them to be executed. Collects results.

        ### Params
        - threads: [List<Thread>] a list of submitted threads
        - raise_exception: [Bool] Flag if an exception in a thread shall be raised or just logged

        ### Returns
        Results from all `Threads` as list
        """
        result = []
        while len(threads) > 0:
            for thread in threads:
                if not thread.is_submitted():
                    threads.remove(thread)
                if not thread.is_done():
                    continue

                if thread.exception() is not None:
                    MP.__exception_handling(threads, thread, raise_exception)
                else:
                    result.append(thread.result())
                threads.remove(thread)
        return result

   
