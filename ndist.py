import torch
import torch.distributed as dist
import time

import redis

import logging 

log = logging.getLogger("ndist.py")
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s', datefmt='%m-%d %H:%M:%S')

TIMESTAMP='timestamp'

class AsyncNCCLDist:

    
    def __init__(self, master=False):
           # instance variable unique to each instance

        self.group_mapping = {}
        self.key_count = {}
        self.stream_seq = 1
        self.master = master
        self.initialize('localhost', 6379, 0)


    def initialize(self, host, port, db):

        log.info("Initializing...")
        self.db = redis.Redis(host=host, port=port, db=db)
        if self.master:
            self.db.flushdb()
            self.db.set("MASTER", 'go')
        else:
            while not self.db.exists("MASTER"):
                time.sleep(0.001)
        log.info("Redis client inited.")
        self.db.set(TIMESTAMP, '0')


    def init_process_group(self, backend, rank, world_size):
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        #dist.broadcast(tensor=tensor, group=group, src=rank)

    def new_group(self, rank_list):

        group = dist.new_group(rank_list)
        #print("group"+str(group))
        self.group_mapping[id(group)] = rank_list
        return group

    def tagged_new_group(self, rank_list, tag):

        group = dist.new_group(rank_list)
        #print("group"+str(id(group)))
        self.group_mapping[id(group)] = rank_list
        self.tag_mapping[id(group)] = tag
        return group


    def broadcast(self, tensor, group, src):

        rank_list = self.group_mapping[id(group)]
        if id(group) in self.tag_mapping:
            ## if the group is tagged, then identifiy a unique thread with this tag
            uni_key = "tag"+str(self.tage_mapping[id(group)])+str(rank_list.sort()+"src"+str(src))
        else:
            key = 'tensor'+str(tensor.size())+str(rank_list.sort())
            if key in self.key_count:
                self.key_count[key] += 1
            else:
                self.key_count[key] = 0
            uni_key = key+str(self.key_count[key])

        # if no key exists then assign a key in redis db
        self.db.setnx(uni_key, 0)
        # thread safe increase by 1, return current key
        ready_count = self.db.incr(uni_key)
        log.debug("ready_count for %s: %s", uni_key, ready_count)

        stamp = 0 

        # if ready_count equals to the total ranks in this operation
        if ready_count >= len(rank_list):        
            # get an unique logical clock 
            stamp = self.db.incr(TIMESTAMP)
            # publish the ready message with timestamp
            log.debug('publish %s', stamp)
            self.db.set(uni_key+'stamp', stamp)
            # clean the key
            self.db.delete(uni_key)
            log.debug('Publisher go for %s with logical stamp %s', uni_key, stamp)

        else:

            while not self.db.exists(uni_key+'stamp'):
                #print("stuch unikey stamp")
                time.sleep(0.001)
                # message = sub.get_message()
                # if message:
            stamp = int(self.db.get(uni_key+'stamp'))
            log.debug('Subscriber go for %s with logical stamp %s', uni_key, stamp)
        

        while True:
            if stamp == self.stream_seq:
                break
            else:
                #print("stuck stamp")
                time.sleep(0.001)

        log.debug("ndist broadcast go with logical stamp %d", stamp)
        rt = dist.broadcast(tensor=tensor, group=group, src=src)

        #torch.cuda.synchronize()
        log.debug("ndist broadcast finish with logical stamp %d", stamp)
        self.stream_seq+=1

        return rt
