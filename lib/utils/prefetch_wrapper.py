from multiprocessing import Process, Lock, Manager
import time
from utils.Fast_Queue import Fast_Queue as Queue


class PrefetchWrapper:
    q = None
    done = False
    nr_processes = 5
    index_lock = Lock()


    def __init__(self,fp, prefetch_len, *args):
        self.q = Queue(maxsize=prefetch_len)
        self.p = [Process(target=self.execute_func, args=(fp,prefetch_len,self.index_lock, args[0],args[1],args[2])) for x in range(self.nr_processes)]
        [prc.start() for prc in self.p]


    def get_item(self):
        return self.q.get()

    def forward(self, *args):
        # dont care about incoming args
        return self.q.get()

    def kill(self):
        self.done = True
        print("trying to join")
        [prc.join(5) for prc in self.p]
        print("joined")
        return None



    def execute_func(self, fp, prefetch_len,index_lock, *args):
        while not self.done:
            if self.q.qsize() < prefetch_len:
                blob = fp(args[0],args[1],args[2],index_lock)
                self.q.put(blob)
            else:
                time.sleep(1)
        return None