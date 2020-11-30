from itertools import product
import numpy as np

class JobIterator():
    
    
    def __init__(self,params):
        self.params = params
        self.product = list(dict(zip(params,x))for x in product(*params.values()))
        self.iter = (dict(zip(params,x))for x in product(*params.values()))
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()
    def get_index(self,i):
        return self.product[i]
    def get_njobs(self):
        return len(self.product)
    def set_attributes_by_index(self,i,obj):
        # For an arbitrary object, set the attributes to match the ith job parameters
        d = self.get_index(i)
        for k,v in d.items():
            setattr(obj,k,v)