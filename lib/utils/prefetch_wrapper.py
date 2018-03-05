from multiprocessing import Process, Queue
import time


class PrefetchWrapper:
    q = None
    done = False

    def __init__(self,fp, prefetch_len, *args):
        self.q = Queue()
        self.p = Process(target=self.execute_func, args=(fp,prefetch_len, args[0],args[1],args[2]))
        self.p.start()

    def get_item(self):
        return self.q.get()

    def forward(self, *args):
        # dont care about incoming args
        return self.q.get()

    def kill(self):
        self.done = True
        return None

    def execute_func(self, fp, prefetch_len, *args):
        while not self.done:
            if self.q.qsize() < prefetch_len:
                self.q.put(fp(args[0],args[1],args[2]))
            else:
                time.sleep(.5)
        return None