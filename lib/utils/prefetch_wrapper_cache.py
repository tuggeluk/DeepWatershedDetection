from multiprocessing import Process, Lock, Manager,Value
import time
import pickle
import os
import shutil


class PrefetchWrapperCache:
    done = Value('b', False)
    chunk_ind = Value('i', 0)
    cache_checked = Value('b', False)
    nr_processes = 5
    index_lock = None
    wait_list = []

    active_chunks_dir = "../../active_prefetch_chunks"

    cache_dir = None
    config_fingerprint = None


    def __init__(self,fp, prefetch_len,prefetch_size, cache_dir, nr_proc, fingerprint,  *args):

        self.index_lock = Lock()
        self.nr_processes = nr_proc
        self.config_fingerprint = fingerprint
        self.cache_dir = cache_dir

        # clear active dict
        try:
            shutil.rmtree(self.active_chunks_dir)
        except:
            pass
        os.mkdir(self.active_chunks_dir)
        self.forward(args)
        #self.execute_func(fp, prefetch_len,prefetch_size,self.active_chunks_dir, cache_dir, self.index_lock, fingerprint, *args)
        self.p = [Process(target=self.execute_func, args=(fp, prefetch_len,prefetch_size,self.active_chunks_dir, cache_dir, self.index_lock, fingerprint, *args)) for x in range(self.nr_processes)]
        [prc.start() for prc in self.p]

    def forward(self, *args):
        if len(self.wait_list) == 0:
            # try to load next cache chunk until done
            chunk_loaded = False
            while not chunk_loaded:
                if "chunk exists":
                    self.wait_list = pickle.load("datachunk")
                    self.chunk_ind.value += 1
                    chunk_loaded = True

                else:
                    print("------------------------")
                    print("Waiting for pickle cache")
                    print("------------------------")
                    time.sleep(2)

        return self.wait_list.pop()

    def kill(self):
        self.done.value = True
        print("trying to join")
        [prc.join(2) for prc in self.p]
        print("joined")
        return None

    def execute_func(self, fp, prefetch_len,prefetch_size,active_chunks, cache_dir, index_lock,fingerprint, *args):
        # aquire lock
        index_lock.acquire()
        # if chache non-empty and cache_checked == false
        if not self.cache_checked.value:
            self.cache_checked.value = True
            if os.path.exists(cache_dir + "/" + fingerprint):
            #   move full cache
                chunks = os.listdir(cache_dir + "/" + fingerprint)
                for chunk in chunks:
                    shutil.move(cache_dir + "/" + fingerprint+"/"+chunk, active_chunks+"/"+chunk)
            #   fast forward batch index on data generator
                self.chunk_ind.value += len(chunks)
                nr_b = len(chunks) * prefetch_size
                fp(ff=True, nr=nr_b, batch_size=args[0].batch_size)
            else:
                os.makedirs(cache_dir + "/" + fingerprint)

        # release lock
        index_lock.release()

        while not self.done.value:
            chunk_l = []
            # build batch
            for i in range(prefetch_size):
                chunk_l.append(fp(args[0], args[1], args[2], index_lock))

            index_lock.acquire()
            chunk_nr = self.chunk_ind.value
            self.chunk_ind.value += 1
            index_lock.release()

            pickle.dump(chunk_l, open(active_chunks +"/"+ str(chunk_nr)+"_prefetch.p", "wb"))

            if len(os.listdir(active_chunks)) > prefetch_len:
                print("prefetch full -------------- taking a break")
                time.sleep(4)

        return None



