import os
import glob
import torch
from typing import Optional, Dict, Any

class CheckpointManager:
    """Model checkpoint manager"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, epoch: int, model_state: Dict[str, Any], 
                       optimizer_state: Dict[str, Any] = None, 
                       scheduler_state: Dict[str, Any] = None,
                       filename: str = None) -> str:
        """Save checkpoint
        
        Args:
            epoch: Current training epoch
            model_state: Model state dict
            optimizer_state: Optimizer state dict
            scheduler_state: Scheduler state dict
            filename: Custom filename
            
        Returns:
            Path to the saved checkpoint file
        """
        if filename is None:
            filename = f"model_epoch_{epoch:03d}.pth"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        if scheduler_state is not None:
            checkpoint['scheduler_state_dict'] = scheduler_state
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint
        
        Args:
            checkpoint_path: Checkpoint file path
            
        Returns:
            Checkpoint data dict
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Checkpoint loaded: {checkpoint_path}")
        
        return checkpoint
    
    def has_checkpoint(self) -> bool:
        """Check whether checkpoint files exist"""
        pattern = os.path.join(self.checkpoint_dir, "*.pth")
        checkpoints = glob.glob(pattern)
        return len(checkpoints) > 0
    
    def latest_checkpoint(self) -> Optional[str]:
        """Get the latest checkpoint file path"""
        pattern = os.path.join(self.checkpoint_dir, "*.pth")
        checkpoints = glob.glob(pattern)
        
        if not checkpoints:
            return None
        
        # Sort by modified time and return the latest
        latest = max(checkpoints, key=os.path.getmtime)
        return latest
    
    def list_checkpoints(self) -> list:
        """List all checkpoint files"""
        pattern = os.path.join(self.checkpoint_dir, "*.pth")
        checkpoints = glob.glob(pattern)
        
        # Sort by file name
        checkpoints.sort()
        return checkpoints
    
    def remove_old_checkpoints(self, keep_last: int = 5):
        """Remove old checkpoint files and keep only the latest ones
        
        Args:
            keep_last: Number of latest checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_last:
            return
        
        # Sort by modified time
        checkpoints.sort(key=os.path.getmtime)
        
        # Delete old checkpoints
        to_remove = checkpoints[:-keep_last]
        for checkpoint in to_remove:
            try:
                os.remove(checkpoint)
                print(f"Removed old checkpoint: {checkpoint}")
            except OSError as e:
                print(f"Error removing checkpoint {checkpoint}: {e}")
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """Get checkpoint info (without loading the full model)
        
        Args:
            checkpoint_path: Checkpoint file path
            
        Returns:
            Basic checkpoint information
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'file_size': os.path.getsize(checkpoint_path),
            'modified_time': os.path.getmtime(checkpoint_path),
            'has_optimizer': 'optimizer_state_dict' in checkpoint,
            'has_scheduler': 'scheduler_state_dict' in checkpoint,
        }
        
        return info