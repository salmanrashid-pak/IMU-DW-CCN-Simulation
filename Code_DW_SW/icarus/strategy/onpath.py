"""Implementations of all on-path strategies"""
from __future__ import division
import random
import numpy as np
import operator


import networkx as nx
import matplotlib.pyplot as plt

from icarus.registry import register_strategy
from icarus.util import inheritdoc, path_links
from collections import defaultdict

from .base import Strategy

__all__ = [
       'Partition',
       'Edge',
       'LeaveCopyEverywhere',
       'LeaveCopyEverywhere1',
       'MP_LeaveCopyEverywhere',
       'LeaveCopyDown',
       'ProbCache',
       'CacheLessForMore',
       'RandomBernoulli',
       'RandomChoice',
       'OptimizedCache',
       'TestCache', 
           ]


@register_strategy('PARTITION')
class Partition(Strategy):
    """Partition caching strategy.

    In this strategy the network is divided into as many partitions as the number
    of caching nodes and each receiver is statically mapped to one and only one
    caching node. When a request is issued it is forwarded to the cache mapped
    to the receiver. In case of a miss the request is routed to the source and
    then returned to cache, which will store it and forward it back to the
    receiver.

    This requires median cache placement, which optimizes the placement of
    caches for this strategy.

    This strategy is normally used with a small number of caching nodes. This
    is the the behaviour normally adopted by Network CDN (NCDN). Google Global
    Cache (GGC) operates this way.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller):
        super(Partition, self).__init__(view, controller)
        if 'cache_assignment' not in self.view.topology().graph:
            raise ValueError('The topology does not have cache assignment '
                             'information. Have you used the optimal median '
                             'cache assignment?')
        self.cache_assignment = self.view.topology().graph['cache_assignment']

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        source = self.view.content_source(content)
        self.controller.start_session(time, receiver, content, log)
        cache = self.cache_assignment[receiver]
        self.controller.forward_request_path(receiver, cache)
        if not self.controller.get_content(cache):
            self.controller.forward_request_path(cache, source)
            self.controller.get_content(source)
            self.controller.forward_content_path(source, cache)
            self.controller.put_content(cache)
        self.controller.forward_content_path(cache, receiver)
        self.controller.end_session()


@register_strategy('EDGE')
class Edge(Strategy):
    """Edge caching strategy.

    In this strategy only a cache at the edge is looked up before forwarding
    a content request to the original source.

    In practice, this is like an LCE but it only queries the first cache it
    finds in the path. It is assumed to be used with a topology where each
    PoP has a cache but it simulates a case where the cache is actually further
    down the access network and it is not looked up for transit traffic passing
    through the PoP but only for PoP-originated requests.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller):
        super(Edge, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        edge_cache = None
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                edge_cache = v
                if self.controller.get_content(v):
                    serving_node = v
                else:
                    # Cache miss, get content from source
                    self.controller.forward_request_path(v, source)
                    self.controller.get_content(source)
                    serving_node = source
                break
        else:
            # No caches on the path at all, get it from source
            self.controller.get_content(v)
            serving_node = v

        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        self.controller.forward_content_path(serving_node, receiver, path)
        if serving_node == source:
            self.controller.put_content(edge_cache)
        self.controller.end_session()


@register_strategy('LCE')
class LeaveCopyEverywhere(Strategy):
    """Leave Copy Everywhere (LCE) strategy.

    In this strategy a copy of a content is replicated at any cache on the
    path between serving node and receiver.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, use_ego_betw=False, **kwargs):
        super(LeaveCopyEverywhere, self).__init__(view, controller)
        
        self.topology = view.topology()
        if use_ego_betw:
            self.betw = dict((v, nx.betweenness_centrality(nx.ego_graph(self.topology, v))[v])
                             for v in self.topology.nodes_iter())            	                    
        else:
            self.betw = nx.betweenness_centrality(self.topology)
        
        #print("Betweenness Centrality: ", self.betw[v])

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
       
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                path1 = self.view.shortest_path(v, source)
                self.controller.put_content_LUN(v,self.betw[v],len(path1))
              
                #if self.controller.remove_content(v) == None:
                   #print("Fuck")
                
                #print "\n"
                #print "-------------------------------------------------------------------"
        self.controller.end_session()

###FOR MP LCE 
@register_strategy('MPLCE')
class MP_LeaveCopyEverywhere(Strategy):
    """Leave Copy Everywhere (LCE) strategy.

    In this strategy a copy of a content is replicated at any cache on the
    path between serving node and receiver.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, use_ego_betw=False, **kwargs):
        super(MP_LeaveCopyEverywhere, self).__init__(view, controller)
        
        self.topology = view.topology()
        
    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):    
    
                      
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)

        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)            
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
       
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                path1 = self.view.shortest_path(v, source)
                print ("TIME :",time)
                print("Content :",content)
                self.controller.put_content_MP(v,time)            
              
        self.controller.end_session()




###Orignal LCE 
@register_strategy('LCE1')
class LeaveCopyEverywhere1(Strategy):
    """Leave Copy Everywhere (LCE) strategy.

    In this strategy a copy of a content is replicated at any cache on the
    path between serving node and receiver.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, use_ego_betw=False, **kwargs):
        super(LeaveCopyEverywhere1, self).__init__(view, controller)
        
        self.topology = view.topology()
        
    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):    
    
        #print ("TIME :",time)              
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)

        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)            
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
       
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                path1 = self.view.shortest_path(v, source)
                self.controller.put_content(v)            
              
        self.controller.end_session()





@register_strategy('LCD')
class LeaveCopyDown(Strategy):
    """Leave Copy Down (LCD) strategy.

    According to this strategy, one copy of a content is replicated only in
    the caching node you hop away from the serving node in the direction of
    the receiver. This strategy is described in [2]_.

    Rereferences
    ------------
    ..[1] N. Laoutaris, H. Che, i. Stavrakakis, The LCD interconnection of LRU
          caches and its analysis.
          Available: http://cs-people.bu.edu/nlaout/analysis_PEVA.pdf
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(LeaveCopyDown, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        # Leave a copy of the content only in the cache one level down the hit
        # caching node
        copied = False
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if not copied and v != receiver and self.view.has_cache(v):
                self.controller.put_content(v)
                copied = True
        self.controller.end_session()


@register_strategy('PROB_CACHE')
class ProbCache(Strategy):
    """ProbCache strategy [3]_

    This strategy caches content objects probabilistically on a path with a
    probability depending on various factors, including distance from source
    and destination and caching space available on the path.

    This strategy was originally proposed in [2]_ and extended in [3]_. This
    class implements the extended version described in [3]_. In the extended
    version of ProbCache the :math`x/c` factor of the ProbCache equation is
    raised to the power of :math`c`.

    References
    ----------
    ..[2] I. Psaras, W. Chai, G. Pavlou, Probabilistic In-Network Caching for
          Information-Centric Networks, in Proc. of ACM SIGCOMM ICN '12
          Available: http://www.ee.ucl.ac.uk/~uceeips/prob-cache-icn-sigcomm12.pdf
    ..[3] I. Psaras, W. Chai, G. Pavlou, In-Network Cache Management and
          Resource Allocation for Information-Centric Networks, IEEE
          Transactions on Parallel and Distributed Systems, 22 May 2014
          Available: http://doi.ieeecomputersociety.org/10.1109/TPDS.2013.304
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, t_tw=10):
        super(ProbCache, self).__init__(view, controller)
        self.t_tw = t_tw
        self.cache_size = view.cache_nodes(size=True)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for hop in range(1, len(path)):
            u = path[hop - 1]
            v = path[hop]
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        c = len([v for v in path if self.view.has_cache(v)])
        x = 0.0
        for hop in range(1, len(path)):
            u = path[hop - 1]
            v = path[hop]
            N = sum([self.cache_size[n] for n in path[hop - 1:]
                     if n in self.cache_size])
            if v in self.cache_size:
                x += 1
            self.controller.forward_content_hop(u, v)
            if v != receiver and v in self.cache_size:
                # The (x/c) factor raised to the power of "c" according to the
                # extended version of ProbCache published in IEEE TPDS
                prob_cache = float(N) / (self.t_tw * self.cache_size[v]) * (x / c) ** c
                if random.random() < prob_cache:
                    self.controller.put_content(v)
        self.controller.end_session()


@register_strategy('CL4M')
class CacheLessForMore(Strategy):
    """Cache less for more strategy [4]_.

    This strategy caches items only once in the delivery path, precisely in the
    node with the greatest betweenness centrality (i.e., that is traversed by
    the greatest number of shortest paths). If the argument *use_ego_betw* is
    set to *True* then the betweenness centrality of the ego-network is used
    instead.

    References
    ----------
    ..[4] W. Chai, D. He, I. Psaras, G. Pavlou, Cache Less for More in
          Information-centric Networks, in IFIP NETWORKING '12
          Available: http://www.ee.ucl.ac.uk/~uceeips/centrality-networking12.pdf
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, use_ego_betw=False, **kwargs):
        super(CacheLessForMore, self).__init__(view, controller)
        topology = view.topology()
        if use_ego_betw:
            self.betw = dict((v, nx.betweenness_centrality(nx.ego_graph(topology, v))[v])
                             for v in topology.nodes_iter())
        else:
            self.betw = nx.betweenness_centrality(topology)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        # No cache hits, get content from source
        else:
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        # get the cache with maximum betweenness centrality
        # if there are more than one cache with max betw then pick the one
        # closer to the receiver
        max_betw = -1
        designated_cache = None
        for v in path[1:]:
            if self.view.has_cache(v):
                if self.betw[v] >= max_betw:
                    max_betw = self.betw[v]
                    designated_cache = v
        # Forward content
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if v == designated_cache:
                self.controller.put_content(v)
        self.controller.end_session()


@register_strategy('RAND_BERNOULLI')
class RandomBernoulli(Strategy):
    """Bernoulli random cache insertion.

    In this strategy, a content is randomly inserted in a cache on the path
    from serving node to receiver with probability *p*.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, p=0.2, **kwargs):
        super(RandomBernoulli, self).__init__(view, controller)
        self.p = p

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if v != receiver and self.view.has_cache(v):
                if random.random() < self.p:
                    self.controller.put_content(v)
        self.controller.end_session()

@register_strategy('RAND_CHOICE')
class RandomChoice(Strategy):
    """Random choice strategy

    This strategy stores the served content exactly in one single cache on the
    path from serving node to receiver selected randomly.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(RandomChoice, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        caches = [v for v in path[1:-1] if self.view.has_cache(v)]
        designated_cache = random.choice(caches) if len(caches) > 0 else None
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if v == designated_cache:
                self.controller.put_content(v)
        self.controller.end_session()


@register_strategy('OPTIMIZED_CACHE')
class OptimizedCache(Strategy):
   
    @inheritdoc(Strategy)
    def __init__(self, view, controller, t_tw=10):
        super(OptimizedCache, self).__init__(view, controller)
        self.t_tw = t_tw

        ####code from Optimized Caching"     
        self.cache_size = view.cache_nodes(size=True)
        self.content_count={}
        self.content_hopeCount={}
        self.content_accesedTime = {}
        self.content_cgp = {}
        self.content_accesedTime.setdefault(0,0)
        self.content_hopeCount.setdefault(0,0)
        self.defCGP = 0
        self.gama = 0.7
        self.defPinit = 0.8
        self.maxFrequency = 1;
        self.maxDvu=1;
        self.defCGP = 0;

        

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):

        #print("-------------------------------------------------------")
        #print("Content Keys : ",self.content_count.keys())
       
        ####code from Optimized Caching"     
        self.content_accesedTime[content] = time
        if content in self.content_count.keys():
            self.content_count[content]=self.content_count[content]+1
            #print("increment in array : ",self.content_count)
        else:
            self.content_count[content]=1
            #print("1 in array : ",self.content_count[content])
            #print("content_count.get %d:",self.content_count.get)
        ####code from Optimized Caching"    

        #print("Same content request count :",self.content_count[content])
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)

        
        #print("Content",content)  
        #print("Receiver",receiver)
        #print("Source/Server",source)
        #print("Receiver to Source/Server: Path",path)

        
        
        self.controller.start_session(time, receiver, content, log)
        for hop in range(1, len(path)):
            u = path[hop - 1]
            v = path[hop]
            #print("U",u)
            #print("V",v)
            self.controller.forward_request_hop(u, v)            
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v 
                    #print("serving within path ",serving_node)                    
                    break
                #print("has cache within path ",v) 
        else:       
            self.controller.get_content(v)
            serving_node = v        
            #print("Content find from source/Server",serving_node)
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        #print("Path : ",path)   
        c = len([v for v in path if self.view.has_cache(v)])
        #print("Number of nodes the request traversed from requestor to source:c:",c)
        d = len([v for v in path]) ##This line add from Optimize_Caching
        #print("Distance between request and source node in number of hop count:d:",d)

        #print("--------------------loop start------------------------------------") 
          
        
        x = 0.0
        for hop in range(1, len(path)):
            u = path[hop - 1]
            v = path[hop]
            #print("U---",u)
            #print("V---",v)


            #print(" Cache Size",self.cache_size)
            #print("    ")             
            #self.content_hopeCount[content]= self.content_hopeCount[content]+1
            
            if v in self.cache_size:
                x += 1            
            self.controller.forward_content_hop(u, v)
            self.content_hopeCount[content]= x
            
            #print("Number of nodes the request traversed from requestor to source:x:",x)
            if self.view.has_cache(v) and self.controller.get_content(v):
                #print("Hit node",v)             
                self.maxFrequency = self.content_count[ max(self.content_count, key=self.content_count.get)]
                #print("maxFrequency :",self.maxFrequency)
                self.maxDvu = self.content_hopeCount[ max(self.content_hopeCount, key=self.content_hopeCount.get)]
                #print("maxDvu :",self.maxDvu)
                defCGPOld = (self.content_count[content] / self.maxFrequency)*(d/self.maxDvu) * (x/d) *(1/self.maxFrequency)
                #print("defCGPOld through formula in if ",defCGPOld)
                self.defCGP = defCGPOld + ( (1 - defCGPOld) * self.defPinit )
                #print("defCGP",self.defCGP)
                #print("defCGP in if :",self.content_cgp[content])
                #print("Size in if :",len(self.content_count))MP
                self.content_cgp[content] = self.defCGP
                #print("self.content_cgp[content] = self.defCGP",self.content_cgp[content]) 
                for con in self.content_count:
                    self.defCGP = self.content_cgp[con] + (self.gama**(time-self.content_accesedTime[content]))
                    #print("defCGP loop :",self.content_cgp[content])
                    #print("loop :",con)
                    self.content_cgp[con] = self.defCGP   
            else:                                  
                self.maxFrequency = self.content_count[ max(self.content_count, key=self.content_count.get)]
                #print("maxFrequency else :",self.maxFrequency)
                self.maxDvu = self.content_hopeCount[ max(self.content_hopeCount, key=self.content_hopeCount.get)]
                #print("maxDvu else :",self.maxDvu)
                if self.maxFrequency == 0 or self.maxDvu ==0:
                       self.maxDvu=1
                       self.maxFrequency=1
                self.defCGP = (1 / self.maxFrequency ) * (d/self.maxDvu)*(x/d)
                self.content_cgp[content] = self.defCGP
                
                #print("defCGP in else :",self.defCGP)
                #print("Size in else :",len(self.content_cgp))
                #print("defCGP",self.defCGP)
                #print("self.content_cgp[content] = self.defCGP",self.content_cgp[content])                                              
            #a=v%2
            #print(" a : ",a)
            #print(" v : ",v)

            if  v != receiver and v in self.cache_size:       
                #print("")         
                #print("v != receiver and v in self.cache_size")
                #print("if cached at Node",v)
                self.controller.put_content(v)                
                #print("")
            else:
                minumCGP = self.content_cgp[ min(self.content_cgp, key=self.content_cgp.get)]
                #print("")
                #print("minumCGP :",minumCGP)
                #print("Current CGP :",self.content_cgp[content])                
                if(self.content_cgp[content]>minumCGP):
                  #print("cached at Node with replacement",v)
                  #print("else Receiver---",receiver)
                  #print("--else Cached at Node ",v)                   
                  self.controller.put_content(v)
                #else:
                  #print("Not Cached at : ",v)                   
            

        #l=self.view.content_locations(content)
        #print("")
        #print("All content locations :",l)            
        #print("Caching Nodes :",self.view.cache_nodes())

        self.controller.end_session()




@register_strategy('TEST_CACHE_0rg')
class TestCache(Strategy):
    
    @inheritdoc(Strategy)
    def __init__(self, view, controller, use_ego_betw=False, **kwargs):        
        super(TestCache, self).__init__(view, controller)

        self.cache_size = view.cache_nodes(size=True)
        self.content_count={}
        self.content_hopeCount={}
        self.content_accesedTime = {}
        self.content_cgp = {}
        self.content_accesedTime.setdefault(0,0)
        self.content_hopeCount.setdefault(0,0)
        self.defCGP = 0
        self.gama = 0.1
        self.defPinit = 0.1
        self.maxFrequency = 1;
        self.maxDvu=1;
        self.defCGP = 0;
        self.Interest={'Content':-1,'Max':-1}
        self.DUMP={}
        self.topology = view.topology()
        self.Node_Count={}
        
        
        if use_ego_betw:
            self.betw = dict((v, nx.betweenness_centrality(nx.ego_graph(self.topology, v))[v])
                             for v in self.topology.nodes_iter())            	                    
        else:
            self.betw = nx.betweenness_centrality(self.topology)
        
          
        self.deg_centrality = nx.degree_centrality(self.topology)
        self.close_centrality = nx.closeness_centrality(self.topology)
        self.bet_centrality = nx.betweenness_centrality(self.topology, normalized = True, endpoints = False) 
        self.load_cent=nx.load_centrality(self.topology, v=None, cutoff=None, normalized=True, weight=None)   

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
  

        self.content_accesedTime[content] = time
        
        

        if content in self.content_count.keys():
            self.content_count[content]=self.content_count[content]+1
            #print("increment in array : ",self.content_count)
            #print("\n\n\n\n")
        else:
            self.content_count[content]=1
            #print("1 in array : ",self.content_count[content])
        
        

            

        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)

        
        #print("Content",content)  
        #print("Receiver",receiver)
        #print("Source/Server",source)
        #print("Receiver to Source/Server: Path",path)
        
        self.Interest['Content']=content
        self.Interest['Max']=len(path)
        #print("\n\n\n\n")
        
        self.controller.start_session(time, receiver, content, log)
        
   
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        self.controller.start_session(time, receiver, content, log)
        
        #print("Topology: ",self.topology) 
        #print("Path: ",path)
        #print("Own Packet: ",self.Interest)
        
        
       
       
       
        """
        for path1 in nx.all_shortest_paths(self.topology, receiver, source):
          print(path1)  
          print("Legth: :",len(path1))       
        """  
          
          

        """ 
	C = self.view.content_locations(content) ##locations of content
        print("content Source :",C)
        """
        
        for hop in range(1, len(path)):            
            u = path[hop - 1]
            v = path[hop]

            #print("U :",u)
            #print("V :",v)

            
            """
            if self.view.has_cache(u):
               test=self.view.cache_dump(u)
               print("Length: ",len(test))                
               #print("Link Delay: ", self.view.link_delay(u,v))
               print("Cache Dump: ",self.view.cache_dump(u)) #cache dump
            """
            
            """
            print("Neighbors of U: ",nx.neighbors(self.topology,u))
            print("Neighbors of v: ",nx.neighbors(self.topology,v)) 
            print("Common Neighbors: ",list(nx.common_neighbors(self.topology, u, v)))
            print("Has edge: ",nx.Graph.has_edge(self.topology,u, v))
            print("Edges: ",nx.edges(self.topology,u))
            """
            """
            if self.view.has_cache(v):
                print("Betweenness Centrality: ", self.betw[v])
                print("Degree Centrality :",self.deg_centrality[v])
                print("Closness Centrality :",self.close_centrality[v])
                print("Bet_Centrality: ", self.bet_centrality[v])
                print("Load Centrality: ",self.load_cent[v]) 
            """    
            #print("\n\n")
            
            
            
            self.controller.forward_request_hop(u, v)            
            if self.view.has_cache(v):
               if self.controller.get_content(v):
                  serving_node = v  
                  
                  """                
                  ########################################################
                  #Calculate Hit                                    
        	  if not serving_node in self.Node_Count.keys():
        	     self.Node_Count[serving_node]=1
        	  else:   
            	     self.Node_Count[serving_node]=self.Node_Count[serving_node]+1                    	                          
                  ##########################################################            
                  """
                  
                  break
        else:            
            self.controller.get_content(v)
            serving_node = v
            """
            #################################################################
            self.Node_Count[serving_node]=1                   
            ##########################################+34 	########################              
        print self.Node_Count  
            """
        
             
        path = list(reversed(self.view.shortest_path(receiver, serving_node))) 
        #print("Reverse Path: ",path)
       
        
        c = len([v for v in path if self.view.has_cache(v)])
        #print ("Caching Nodes: ",c)

        d = len(path)
        #print ("Total path Length: ",d)

        x=0.0
        for hop in range(1, len(path)):
            u = path[hop - 1]
            v = path[hop]
            if v in self.cache_size:
                x += 1
                #print ("V: ",v) 
                self.content_hopeCount[content]= x
                    
            self.controller.forward_content_hop(u, v)
            #print("Hop Couunt: ",x)
     
            if v != receiver and self.view.has_cache(v):
               self.controller.put_content(v)
        
        #print("\n\n\n\n")
        
        """
        print ("Cached Node: ",v)
        #load calculate for each node     
        if v in self.Node_Count.keys():
            self.Node_Count[v]=self.Node_Count[v]+1            
        else:
            self.Node_Count[v]=1            
        print self.Node_Count
        """
        print("\n")
        self.controller.end_session()




@register_strategy('COCP')#paper values extraction from this orignal code 
class TestCache(Strategy):
    
    @inheritdoc(Strategy)
    def __init__(self, view, controller, use_ego_betw=False, **kwargs):        
        super(TestCache, self).__init__(view, controller)

        
        self.cache_size = view.cache_nodes(size=True)
        self.content_count={} 
        self.content_diversity={} 
        self.topology = view.topology()
        self.Flag=False  
        
        self.close_centrality = nx.closeness_centrality(self.topology)
        self.bet_centrality = nx.betweenness_centrality(self.topology, normalized = True, endpoints = False)      
      
    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
   
        
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)        
        #print("Reciever to Source Path: ",path)       
        self.controller.start_session(time, receiver, content, log)         
        
        serving_node=0
        Aserving_node=0
        for hop in range(1, len(path)):
            u = path[hop - 1]
            v = path[hop]            
            self.controller.forward_request_hop(u, v)            
            if self.view.has_cache(v) and self.Flag==False:
               if self.controller.get_content(v):                    
                    serving_node = v                     
                    self.Flag=True
                    break
               else:
                    Nei = nx.neighbors(self.topology,v)                                        
                    for NNode in Nei[0:]:
                       if self.view.has_cache(NNode):  
                          if self.controller.get_content(NNode):
                             Aserving_node = NNode 
                             self.controller.put_content(v)                                                         
                             self.Flag=True
                             serving_node = v 
                             break 
            elif self.Flag==True:
                break                       
        else:       
            self.controller.get_content(v)
            serving_node = v  
                              
        
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))        
        HopCount = len([v for v in path if self.view.has_cache(v)])#Hop count serving node 
        TotalDis = len([v for v in path]) #Total Distance between Reciever to Source
       
        
	for hop in range(1, len(path)):
	    u = path[hop - 1]
	    v = path[hop]
	    self.controller.forward_content_hop(u, v)  
	    if v != receiver and v in self.cache_size and self.Flag==False:  
	         self.controller.put_content(v)
	         break
	         
        self.Flag=False
        #print("\n\n\n")        
        
        self.controller.end_session()
        
                        
@register_strategy('TEST_CACHE')
class TestCache(Strategy):
    
    @inheritdoc(Strategy)
    def __init__(self, view, controller, use_ego_betw=False, **kwargs):        
        super(TestCache, self).__init__(view, controller)

        
        self.cache_size = view.cache_nodes(size=True)
        self.content_count={} 
        self.content_diversity={} 
        self.topology = view.topology()
        self.Flag=False  
        
        self.close_centrality = nx.closeness_centrality(self.topology)
        self.bet_centrality = nx.betweenness_centrality(self.topology, normalized = True, endpoints = False)      
      
    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
   
        if content in self.content_count.keys():
		self.content_count[content]= self.content_count[content]+1           
        else:
		self.content_count[content]= 1
		
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)        
        
        #print("Reciever to Source Path: ",path)       
        
        self.controller.start_session(time, receiver, content, log)         
        
        serving_node=0
        Aserving_node=-1
        for hop in range(1, len(path)):
            u = path[hop - 1]
            v = path[hop]            
            self.controller.forward_request_hop(u, v)            
            if self.view.has_cache(v) and self.Flag==False:
               if self.controller.get_content(v):                    
                    serving_node = v                     
                    self.Flag=True
                    break
               else:
                    Nei = nx.neighbors(self.topology,v)                                        
                    for NNode in Nei[0:]:
                       if self.view.has_cache(NNode):  
                          if self.controller.get_content(NNode):
                             Aserving_node = NNode 
                             self.controller.put_content(v)  #After Comment this line                                                       
                             self.Flag=True
                             serving_node = v 
                             break 
            elif self.Flag==True:
                break                       
        else:       
            self.controller.get_content(v)
            serving_node = v  
                              
        
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))        
        
        
        HopCount = len([v for v in path if self.view.has_cache(v)])#Hop count serving node 
        TotalDis = len([v for v in path]) #Total Distance between Reciever to Source
        
        
        MinLprob={}
        TPC=1.0
        x=0.0
        HelpNode=0
        space=0.0
        
        ####################Count Accumulative Path cache sizes ############################
        
        for n in path:               
               if n in self.cache_size:
                  TPC +=(len(self.view.cache_dump(n))) 
                  #print("TPC :",TPC)
                  #print("n :",n)
                  #print("\n")
        ####################End Count Accumulative Path cache sizes #######################
        #print("TPC out side :",TPC)
        
	for hop in range(1, len(path)):
	    	u = path[hop - 1]
	    	v = path[hop]
	    
	      
                self.maxFrequency = sum(self.content_count.values()) #Calculate Total requests on same router
		
	    	self.controller.forward_content_hop(u, v)  
	    	
	    	if v != receiver and v in self.cache_size and self.Flag==False:  
		    	x+=1 
			ContMaxFre = (self.content_count[content] / self.maxFrequency)
			#print("ContMaxFre :",ContMaxFre)
			
			HopTD = x/TotalDis 
			#print("HopTD :",HopTD)
			
			Distance = (HopCount)/(TotalDis)
			#print("Distance :",Distance)
			
			#print("Cache Space :",len(self.view.cache_dump(v)))
			
			if (len(self.view.cache_dump(v))) == 0:
				space=1
			else:			        
			 	space = (len(self.view.cache_dump(v)))/TPC
			 	
			#print("Space :",space)
			#print("\n\n")
			
			LProb = (space)*(ContMaxFre)*(HopTD)*(Distance)   
			MinLprob[v]=LProb			
			HelpNode=v
			#print("MinLprob :",MinLprob)
			#print("LProb :",LProb)
			#print("v :",v)
			#print("\n\n------------------------\n")
	
	#if self.Flag==False: and  Aserving_node==-1:		
			if len(MinLprob)==0:
			       self.controller.put_content(HelpNode)
			       #print("MinLprob :", MinLprob)
			       #print("HelpNode :", HelpNode)                		
			       #break
		       
			else:
			       MaxPN = min(MinLprob, key=MinLprob.get)
			       #print ("Cached At :",MaxPN)
			       self.controller.put_content(MaxPN)
			       #break	
	    		
	    	
	    	
	         
	         
        self.Flag=False
        #print("\n\n\n###########################")        
        
        self.controller.end_session()
        
                        
                                
