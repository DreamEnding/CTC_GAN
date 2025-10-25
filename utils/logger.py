import os
import json
from datetime import datetime
from typing import Dict, Any

class Logger:
    """Training logger"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
        self.metrics_file = os.path.join(log_dir, f"metrics_{timestamp}.json")
        
        # Initialize metrics storage
        self.metrics = {}
        
        # Write initial log
        self._write_log(f"Logger initialized at {datetime.now()}")
    
    def _write_log(self, message: str):
        """Write text log"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Record scalar value"""
        if tag not in self.metrics:
            self.metrics[tag] = []
        
        self.metrics[tag].append({
            'step': step,
            'value': float(value),
            'timestamp': datetime.now().isoformat()
        })
        
        # Write text log
        self._write_log(f"{tag}: {value:.6f} (step {step})")
        
        # Save metrics to JSON file
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
    
    def log_info(self, message: str):
        """Record info log"""
        self._write_log(f"INFO: {message}")
    
    def log_warning(self, message: str):
        """Record warning log"""
        self._write_log(f"WARNING: {message}")
    
    def log_error(self, message: str):
        """Record error log"""
        self._write_log(f"ERROR: {message}")
    
    def log_config(self, config: Dict[str, Any]):
        """Record configuration"""
        config_file = os.path.join(self.log_dir, "config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        self._write_log("Configuration saved")
    
    def get_metrics(self, tag: str = None):
        """Get recorded metrics"""
        if tag:
            return self.metrics.get(tag, [])
        return self.metrics