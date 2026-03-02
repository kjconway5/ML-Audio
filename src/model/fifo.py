import torch
import numpy as np

# Simulate FIFO with parameterized depth, read/write pointers, and full/empty flags
# Can handle signle samples or streaming
class FIFO:
    def __init__(self, depth, chunk_size=None, data_width=16):
        """
        Args:
            depth: # of samples that can be held
            chunk_size: optional arg for streaming
            data_width: bit width of samples
        """
        self.depth = depth
        self.chunk_size = chunk_size
        self.data_width = data_width
        
        self.buffer = torch.zeros(depth, dtype=torch.int32)
        
        # pointers
        self.write_ptr = 0
        self.read_ptr = 0
        self.num_elements = 0
    
    def write(self, data):
        """
        Args:
            data: single sample or array of samples (int16)
            
        Returns:
            success: True if all data written, False if overflow
        """
        # convert to array if single sample
        if isinstance(data, (int, np.integer)):
            data = [data]
        elif isinstance(data, torch.Tensor):
            data = data.numpy()
        
        data = np.array(data, dtype=np.int32)
        
        # overflow check
        if self.num_elements + len(data) > self.depth:
            return False
        
        # loop thru writing data if wont overflow, update write pointer and element count
        for sample in data:
            self.buffer[self.write_ptr] = sample
            self.write_ptr = (self.write_ptr + 1) % self.depth
            self.num_elements += 1
        
        return True
    
    def read(self, num_samples=1):
        """
        Args:
            num_samples: # of samples to read
            
        Returns:
            data: Array of samples as np.int16 (None if underflow)
        """
        # underflow
        if self.num_elements < num_samples:
            return None
        
        # store read samples
        data = torch.zeros(num_samples, dtype=torch.int32)
        
        # loop and read, decrement num_elements and update pointer
        for i in range(num_samples):
            data[i] = self.buffer[self.read_ptr]
            self.read_ptr = (self.read_ptr + 1) % self.depth
            self.num_elements -= 1
        
        return data.numpy().astype(np.int16)
    
    def read_chunk(self):
        """
        Read a chunk from FIFO if chunk_size is set in creating FIFO
        Returns:
            chunk: Audio chunk as np.int16 (or None if not enough data)
        """
        if self.chunk_size is None:
            raise ValueError("chunk_size must be set to use read_chunk()")
        
        # not enough elements to read a chunk
        if self.num_elements < self.chunk_size:
            return None
        
        return self.read(self.chunk_size)
    
    def peek(self, num_samples=1):
        """
        Args:
            num_samples: # of samples to peek
            
        Returns:
            data: array of samples (None if not enough data)
        """
        if self.num_elements < num_samples:
            return None
        
        data = torch.zeros(num_samples, dtype=torch.int32)
        read_ptr_temp = self.read_ptr
        
        for i in range(num_samples):
            data[i] = self.buffer[read_ptr_temp]
            read_ptr_temp = (read_ptr_temp + 1) % self.depth
        
        return data.numpy().astype(np.int16)
    
    def is_full(self):
        return self.num_elements >= self.depth
    
    def is_empty(self):
        return self.num_elements == 0
    
    def get_num_elements(self):
        return self.num_elements
    
    def get_chunks_available(self):
        # if not created with chunk_size set theres no chunks available
        if self.chunk_size is None:
            return 0
        return self.num_elements // self.chunk_size
    
    def reset(self):
        self.write_ptr = 0
        self.read_ptr = 0
        self.num_elements = 0
        self.buffer.zero_()
    
    def get_config(self):
        return {
            'depth': self.depth,
            'chunk_size': self.chunk_size,
            'data_width': self.data_width,
            'num_elements': self.num_elements,
            'is_full': self.is_full(),
            'is_empty': self.is_empty()
        }