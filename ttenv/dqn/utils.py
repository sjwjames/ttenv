import os
import torch

# ================================================================
# Saving variables
# ================================================================

def load_state(path):
    """Load model state from file.
    
    Parameters
    ----------
    path: str
        Path to the saved model state
    """
    return torch.load(path)

def save_state(path):
    """Save model state to file.
    
    Parameters
    ----------
    path: str
        Path where to save the model state
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(path)

# ================================================================
# Input processing
# ================================================================

class Processor:
    """Base class for input processors."""
    def __init__(self, name="(unnamed)"):
        """Initialize a processor with a name."""
        self.name = name
    
    def process(self, data):
        """Process input data.
        
        Parameters
        ----------
        data: Any
            Input data to process
            
        Returns
        -------
        processed_data: Any
            Processed data
        """
        raise NotImplementedError()

    
class BatchProcessor(Processor):
    def __init__(self, shape, device='cpu', dtype=torch.float32, name=None):
        """Creates a processor for a batch of tensors of a given shape and dtype
        
        Parameters
        ----------
        shape: tuple of int
            Shape of a single element of the batch
        device: str or torch.device
            Device to place tensor on
        dtype: torch.dtype
            Number representation used for tensor contents
        name: str
            Name of the processor
        """
        super().__init__(name=name if name is not None else "BatchProcessor")
        self.shape = shape
        self.dtype = dtype
        self.device = device
    
    def process(self, data):
        """Convert data to tensor with appropriate shape and type.
        
        Parameters
        ----------
        data: numpy.ndarray
            Input data
            
        Returns
        -------
        processed_data: torch.Tensor
            Processed data as tensor
        """
        return torch.tensor(data, dtype=self.dtype, device=self.device)


class Uint8Processor(BatchProcessor):
    def __init__(self, shape, device='cpu', name=None):
        """Takes input in uint8 format which is cast to float32 and divided by 255
        before passing it to the model.
        
        Parameters
        ----------
        shape: tuple of int
            Shape of a single element
        device: str or torch.device
            Device to place tensor on
        name: str
            Name of the processor
        """
        super().__init__(shape, device, torch.float32, name=name if name is not None else "Uint8Processor")
    
    def process(self, data):
        """Convert uint8 data to normalized float32 tensor.
        
        Parameters
        ----------
        data: numpy.ndarray
            Input data in uint8 format
            
        Returns
        -------
        processed_data: torch.Tensor
            Processed data as normalized float32 tensor
        """
        return torch.tensor(data, device=self.device, dtype=torch.uint8).float() / 255.0
