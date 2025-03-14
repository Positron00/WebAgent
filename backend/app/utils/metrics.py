"""
Metrics Utilities
================

Utilities for tracking metrics, performance, and resource usage.
This module provides decorators and functions for monitoring 
execution time, memory usage, and other performance metrics.
"""

import time
import logging
import functools
import os
import psutil
import tracemalloc
from typing import Callable, Any, Dict, Optional
import inspect

# Initialize logging
logger = logging.getLogger(__name__)

def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure and log the execution time of a function.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function with timing
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Get the name of the class if this is a method
            if args and hasattr(args[0], '__class__'):
                class_name = args[0].__class__.__name__
                logger.debug(f"{class_name}.{func.__name__} executed in {execution_time:.4f} seconds")
            else:
                logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
    
    return wrapper

def log_memory_usage(func: Callable) -> Callable:
    """
    Decorator to track and log memory usage of a function.
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function with memory tracking
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get current process
        process = psutil.Process(os.getpid())
        
        # Get memory usage before function execution
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Start tracemalloc for detailed tracking
        tracemalloc.start()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Get detailed memory allocation info
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Get memory usage after function execution
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_diff = memory_after - memory_before
            
            # Get the name of the class if this is a method
            if args and hasattr(args[0], '__class__'):
                class_name = args[0].__class__.__name__
                logger.debug(f"{class_name}.{func.__name__} - Memory: {memory_diff:.2f} MB change, "
                            f"Current: {current / 1024 / 1024:.2f} MB, Peak: {peak / 1024 / 1024:.2f} MB")
            else:
                logger.debug(f"{func.__name__} - Memory: {memory_diff:.2f} MB change, "
                            f"Current: {current / 1024 / 1024:.2f} MB, Peak: {peak / 1024 / 1024:.2f} MB")
    
    return wrapper

class MetricsTracker:
    """
    Class for tracking and aggregating metrics during execution.
    """
    
    def __init__(self, name: str = "default"):
        """
        Initialize a metrics tracker.
        
        Args:
            name: Name for this metrics tracker
        """
        self.name = name
        self.timers = {}
        self.counters = {}
        self.gauges = {}
        self.histograms = {}
        
    def start_timer(self, name: str) -> None:
        """
        Start a named timer.
        
        Args:
            name: Timer name
        """
        self.timers[name] = {"start": time.time(), "end": None, "duration": None}
    
    def stop_timer(self, name: str) -> float:
        """
        Stop a named timer and return duration.
        
        Args:
            name: Timer name
            
        Returns:
            Duration in seconds
        """
        if name not in self.timers or self.timers[name]["start"] is None:
            logger.warning(f"Timer {name} not started")
            return 0
        
        self.timers[name]["end"] = time.time()
        self.timers[name]["duration"] = self.timers[name]["end"] - self.timers[name]["start"]
        
        return self.timers[name]["duration"]
    
    def increment_counter(self, name: str, value: int = 1) -> int:
        """
        Increment a named counter.
        
        Args:
            name: Counter name
            value: Value to increment by
            
        Returns:
            New counter value
        """
        if name not in self.counters:
            self.counters[name] = 0
        
        self.counters[name] += value
        return self.counters[name]
    
    def set_gauge(self, name: str, value: float) -> None:
        """
        Set a gauge value.
        
        Args:
            name: Gauge name
            value: Gauge value
        """
        self.gauges[name] = value
    
    def record_value(self, name: str, value: float) -> None:
        """
        Record a value in a histogram.
        
        Args:
            name: Histogram name
            value: Value to record
        """
        if name not in self.histograms:
            self.histograms[name] = []
        
        self.histograms[name].append(value)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all collected metrics.
        
        Returns:
            Dictionary of metrics
        """
        # Calculate histogram statistics
        histogram_stats = {}
        for name, values in self.histograms.items():
            if not values:
                continue
                
            histogram_stats[name] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "values": values  # Include raw values
            }
        
        return {
            "name": self.name,
            "timers": self.timers,
            "counters": self.counters,
            "gauges": self.gauges,
            "histograms": histogram_stats
        }
    
    def reset(self) -> None:
        """
        Reset all metrics.
        """
        self.timers = {}
        self.counters = {}
        self.gauges = {}
        self.histograms = {}
        
    def log_summary(self) -> None:
        """
        Log a summary of collected metrics.
        """
        metrics = self.get_metrics()
        
        logger.info(f"Metrics summary for {self.name}:")
        
        # Log timers
        if metrics["timers"]:
            logger.info("Timers:")
            for name, timer in metrics["timers"].items():
                if timer["duration"] is not None:
                    logger.info(f"  {name}: {timer['duration']:.4f} seconds")
        
        # Log counters
        if metrics["counters"]:
            logger.info("Counters:")
            for name, value in metrics["counters"].items():
                logger.info(f"  {name}: {value}")
        
        # Log gauges
        if metrics["gauges"]:
            logger.info("Gauges:")
            for name, value in metrics["gauges"].items():
                logger.info(f"  {name}: {value}")
        
        # Log histogram stats
        if metrics["histograms"]:
            logger.info("Histograms:")
            for name, stats in metrics["histograms"].items():
                logger.info(f"  {name}: count={stats['count']}, min={stats['min']:.2f}, "
                           f"max={stats['max']:.2f}, mean={stats['mean']:.2f}")

def get_system_metrics() -> Dict[str, Any]:
    """
    Get current system metrics (CPU, memory, disk).
    
    Returns:
        Dictionary of system metrics
    """
    # Get current process
    process = psutil.Process(os.getpid())
    
    # Get CPU metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    process_cpu_percent = process.cpu_percent(interval=0.1)
    
    # Get memory metrics
    memory = psutil.virtual_memory()
    process_memory = process.memory_info()
    
    # Get disk metrics
    disk = psutil.disk_usage('/')
    
    return {
        "system": {
            "cpu": {
                "percent": cpu_percent,
                "cores": psutil.cpu_count()
            },
            "memory": {
                "total": memory.total / (1024 * 1024),  # MB
                "available": memory.available / (1024 * 1024),  # MB
                "used": memory.used / (1024 * 1024),  # MB
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total / (1024 * 1024 * 1024),  # GB
                "used": disk.used / (1024 * 1024 * 1024),  # GB
                "free": disk.free / (1024 * 1024 * 1024),  # GB
                "percent": disk.percent
            }
        },
        "process": {
            "cpu_percent": process_cpu_percent,
            "memory": {
                "rss": process_memory.rss / (1024 * 1024),  # MB (Resident Set Size)
                "vms": process_memory.vms / (1024 * 1024),  # MB (Virtual Memory Size)
            },
            "threads": len(process.threads()),
            "open_files": len(process.open_files()),
            "connections": len(process.connections())
        }
    } 