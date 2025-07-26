import time
import psutil
import GPUtil
import threading
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

class PerformanceMonitor:
    """Monitors system performance during training."""
    
    def __init__(self, log_interval=5):
        self.log_interval = log_interval
        self.monitoring = False
        self.metrics = {
            'cpu_usage': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'gpu_usage': deque(maxlen=1000),
            'gpu_memory': deque(maxlen=1000),
            'timestamps': deque(maxlen=1000)
        }
        self.start_time = None
        
    def start_monitoring(self):
        """Start performance monitoring in a separate thread."""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
            
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # CPU and Memory
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # GPU
                gpu_usage = 0
                gpu_memory = 0
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_usage = gpus[0].load * 100
                        gpu_memory = gpus[0].memoryUtil * 100
                except:
                    pass
                
                # Store metrics
                timestamp = time.time() - self.start_time
                self.metrics['cpu_usage'].append(cpu_percent)
                self.metrics['memory_usage'].append(memory_percent)
                self.metrics['gpu_usage'].append(gpu_usage)
                self.metrics['gpu_memory'].append(gpu_memory)
                self.metrics['timestamps'].append(timestamp)
                
                time.sleep(self.log_interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.log_interval)
                
    def get_current_metrics(self):
        """Get current performance metrics."""
        if not self.metrics['timestamps']:
            return None
            
        return {
            'cpu_usage': self.metrics['cpu_usage'][-1],
            'memory_usage': self.metrics['memory_usage'][-1],
            'gpu_usage': self.metrics['gpu_usage'][-1],
            'gpu_memory': self.metrics['gpu_memory'][-1],
            'elapsed_time': self.metrics['timestamps'][-1]
        }
        
    def plot_performance(self, save_path="performance_metrics.png"):
        """Plot performance metrics."""
        if not self.metrics['timestamps']:
            print("No metrics to plot")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        timestamps = list(self.metrics['timestamps'])
        
        # CPU Usage
        ax1.plot(timestamps, list(self.metrics['cpu_usage']), 'b-', linewidth=2)
        ax1.set_title('CPU Usage (%)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('CPU %')
        ax1.grid(True, alpha=0.3)
        
        # Memory Usage
        ax2.plot(timestamps, list(self.metrics['memory_usage']), 'r-', linewidth=2)
        ax2.set_title('Memory Usage (%)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Memory %')
        ax2.grid(True, alpha=0.3)
        
        # GPU Usage
        ax3.plot(timestamps, list(self.metrics['gpu_usage']), 'g-', linewidth=2)
        ax3.set_title('GPU Usage (%)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('GPU %')
        ax3.grid(True, alpha=0.3)
        
        # GPU Memory
        ax4.plot(timestamps, list(self.metrics['gpu_memory']), 'm-', linewidth=2)
        ax4.set_title('GPU Memory Usage (%)')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('GPU Memory %')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance metrics saved to {save_path}")
        
    def print_summary(self):
        """Print performance summary."""
        if not self.metrics['timestamps']:
            print("No metrics available")
            return
            
        print("\n=== Performance Summary ===")
        print(f"Monitoring duration: {self.metrics['timestamps'][-1]:.1f} seconds")
        print(f"Average CPU usage: {np.mean(self.metrics['cpu_usage']):.1f}%")
        print(f"Average memory usage: {np.mean(self.metrics['memory_usage']):.1f}%")
        print(f"Average GPU usage: {np.mean(self.metrics['gpu_usage']):.1f}%")
        print(f"Average GPU memory: {np.mean(self.metrics['gpu_memory']):.1f}%")
        print(f"Peak CPU usage: {np.max(self.metrics['cpu_usage']):.1f}%")
        print(f"Peak memory usage: {np.max(self.metrics['memory_usage']):.1f}%")
        print(f"Peak GPU usage: {np.max(self.metrics['gpu_usage']):.1f}%")
        print(f"Peak GPU memory: {np.max(self.metrics['gpu_memory']):.1f}%") 