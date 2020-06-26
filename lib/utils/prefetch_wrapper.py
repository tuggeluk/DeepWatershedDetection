from multiprocessing import Process, Lock, Value
import time
from utils.Fast_Queue import Fast_Queue as Queue


class PrefetchWrapper:
    q = Queue(maxsize=7)
    done = Value('b', False)
    nr_processes = 5
    index_lock = None


    def __init__(self,fp, prefetch_len, *args):
        self.done.value = False
        self.index_lock = Lock()
        self.q.clear()
        self.p = [Process(target=self.execute_func, args=(fp,self.q,prefetch_len,self.index_lock, args[0],args[1],args[2])) for x in range(self.nr_processes)]
        [prc.start() for prc in self.p]



    def get_item(self):
        return self.q.get()

    def forward(self, *args):
        # dont care about incoming args
        return self.q.get()

    def kill(self):
        self.done.value = True
        print("trying to join")
        [prc.join(2) for prc in self.p]
        print("joined")
        self.q.clear()
        return None



    def execute_func(self, fp,q, prefetch_len,index_lock, *args):
        while not self.done.value:
            #print(self.done.value)
            if q.qsize() < prefetch_len:
                #print(q.qsize())
                blob = fp(args[0],args[1],args[2],index_lock)
                #print("trying to put")
                q.put(blob)
                #print("dit put")
            else:
                time.sleep(1)
        return None