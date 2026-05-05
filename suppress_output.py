import os
import sys
from contextlib import contextmanager

@contextmanager
def suppress_stderr():
    """
    Context manager to suppress BOTH stdout and stderr at the C level.
    This effectively silences all C++ logs (GLOG/MediaPipe) and Python prints.
    """
    # Flush Python buffers
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Save original file descriptors
    original_stdout_fd = os.dup(sys.stdout.fileno())
    original_stderr_fd = os.dup(sys.stderr.fileno())
    
    try:
        # Open devnull
        devnull = os.open(os.devnull, os.O_WRONLY)
        
        # Replace stdout and stderr with devnull
        os.dup2(devnull, sys.stdout.fileno())
        os.dup2(devnull, sys.stderr.fileno())
        
        os.close(devnull)
        yield
    finally:
        # Flush again
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Restore original file descriptors
        os.dup2(original_stdout_fd, sys.stdout.fileno())
        os.dup2(original_stderr_fd, sys.stderr.fileno())
        
        os.close(original_stdout_fd)
        os.close(original_stderr_fd)
