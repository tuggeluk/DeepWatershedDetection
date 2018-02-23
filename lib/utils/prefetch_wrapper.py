from multiprocessing import Process, Queue
import time


class PrefetchWrapper:
    q = Queue()
    done = False

    def __init__(self,fp, prefetch_len, *args):
        self.p = Process(target=self.execute_func, args=(fp,prefetch_len, args[0]))
        self.p.start()

    def get_item(self):
        return self.q.get()

    def kill(self):
        self.done = True
        return None

    def execute_func(self, fp, prefetch_len, *args):
        while not self.done:
            if self.q.qsize() < prefetch_len:
                print("loading data")
                self.q.put(fp(args[0]))
            else:
                print("sleeping")
                time.sleep(.5)
        return None