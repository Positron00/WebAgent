"""
Grafana Integration Module
====================

Provides tools for integrating ML observability metrics with Grafana
for visualization, alerting, and monitoring.
"""

import os
import json
import requests
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path

from backend.app.core.logger import logger
from backend.app.core.config import Settings

class GrafanaDashboardGenerator:
    """Generates Grafana dashboards for ML observability metrics."""
    
    def __init__(
        self, 
        grafana_url: Optional[str] = None,
        token: Optional[str] = None,
        dashboard_dir: Optional[str] = None
    ):
        """
        Initialize the Grafana dashboard generator.
        
        Args:
            grafana_url: URL of the Grafana server (default: from Settings.GRAFANA_URL)
            token: Grafana service account token (default: from env var GRAFANA_SERVICE_ACCOUNT_TOKEN)
            dashboard_dir: Directory to save dashboard JSON files (default: ./dashboards)
        """
        settings = Settings()
        self.grafana_url = grafana_url or settings.GRAFANA_URL or os.environ.get("GRAFANA_URL")
        
        # Support both service account tokens and legacy API keys for backward compatibility
        self.token = token or os.environ.get("GRAFANA_SERVICE_ACCOUNT_TOKEN") or settings.GRAFANA_SERVICE_ACCOUNT_TOKEN or os.environ.get("GRAFANA_API_KEY")
        
        self.dashboard_dir = dashboard_dir or os.path.join(os.getcwd(), "dashboards")
        
        # Create dashboard directory if it doesn't exist
        if self.dashboard_dir:
            os.makedirs(self.dashboard_dir, exist_ok=True)
    
    def generate_model_performance_dashboard(
        self, 
        title: str = "ML Model Performance",
        folder: str = "ML Observability",
        model_ids: Optional[List[str]] = None,
        datasource: str = "Prometheus"
    ) -> Dict[str, Any]:
        """
        Generate a Grafana dashboard for model performance metrics.
        
        Args:
            title: Dashboard title
            folder: Grafana folder
            model_ids: List of model IDs to include (default: all)
            datasource: Grafana datasource name
            
        Returns:
            Dashboard JSON configuration
        """
        # Default variables
        variables = [
            {
                "name": "model",
                "label": "Model",
                "query": "label_values(model_id)",
                "type": "query",
                "datasource": datasource,
                "refresh": 2  # On dashboard load
            },
            {
                "name": "timeframe",
                "label": "Time Range",
                "query": "1h,6h,12h,24h,7d,30d",
                "type": "custom",
                "datasource": datasource,
                "refresh": 0  # Never
            }
        ]
        
        # Base dashboard structure
        dashboard = {
            "id": None,  # New dashboard
            "uid": None,  # Auto-generated
            "title": title,
            "tags": ["ml", "observability", "model-performance"],
            "timezone": "browser",
            "schemaVersion": 36,
            "version": 1,
            "refresh": "5s",
            "time": {
                "from": "now-6h",
                "to": "now"
            },
            "templating": {
                "list": variables
            },
            "panels": self._generate_model_performance_panels(datasource),
            "editable": True,
            "fiscalYearStartMonth": 0,
            "graphTooltip": 1,  # Shared crosshair
            "links": [],
            "liveNow": False,
            "annotations": {
                "list": [
                    {
                        "builtIn": 1,
                        "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                        "enable": True,
                        "hide": True,
                        "iconColor": "rgba(0, 211, 255, 1)",
                        "name": "Annotations & Alerts",
                        "target": {"limit": 100, "matchAny": False, "tags": [], "type": "dashboard"},
                        "type": "dashboard"
                    },
                    {
                        "datasource": datasource,
                        "enable": True,
                        "expr": "changes(model_retrain_recommended{model_id=~\"$model\"}[1m]) > 0",
                        "iconColor": "red",
                        "name": "Model Retraining Recommended",
                        "tagKeys": ["model_id"],
                        "titleFormat": "Retraining recommended for ${model_id}"
                    }
                ]
            }
        }
        
        return dashboard
    
    def generate_drift_detection_dashboard(
        self, 
        title: str = "ML Drift Detection",
        folder: str = "ML Observability",
        detector_ids: Optional[List[str]] = None,
        datasource: str = "Prometheus"
    ) -> Dict[str, Any]:
        """
        Generate a Grafana dashboard for drift detection metrics.
        
        Args:
            title: Dashboard title
            folder: Grafana folder
            detector_ids: List of detector IDs to include (default: all)
            datasource: Grafana datasource name
            
        Returns:
            Dashboard JSON configuration
        """
        # Default variables
        variables = [
            {
                "name": "detector",
                "label": "Drift Detector",
                "query": "label_values(detector_id)",
                "type": "query",
                "datasource": datasource,
                "refresh": 2  # On dashboard load
            },
            {
                "name": "timeframe",
                "label": "Time Range",
                "query": "1h,6h,12h,24h,7d,30d",
                "type": "custom",
                "datasource": datasource,
                "refresh": 0  # Never
            }
        ]
        
        # Base dashboard structure
        dashboard = {
            "id": None,  # New dashboard
            "uid": None,  # Auto-generated
            "title": title,
            "tags": ["ml", "observability", "drift-detection"],
            "timezone": "browser",
            "schemaVersion": 36,
            "version": 1,
            "refresh": "5s",
            "time": {
                "from": "now-6h",
                "to": "now"
            },
            "templating": {
                "list": variables
            },
            "panels": self._generate_drift_detection_panels(datasource),
            "editable": True,
            "fiscalYearStartMonth": 0,
            "graphTooltip": 1,  # Shared crosshair
            "links": [],
            "liveNow": False,
            "annotations": {
                "list": [
                    {
                        "builtIn": 1,
                        "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                        "enable": True,
                        "hide": True,
                        "iconColor": "rgba(0, 211, 255, 1)",
                        "name": "Annotations & Alerts",
                        "target": {"limit": 100, "matchAny": False, "tags": [], "type": "dashboard"},
                        "type": "dashboard"
                    },
                    {
                        "datasource": datasource,
                        "enable": True,
                        "expr": "changes(drift_detected{detector_id=~\"$detector\"}[1m]) > 0",
                        "iconColor": "red",
                        "name": "Drift Detected",
                        "tagKeys": ["detector_id"],
                        "titleFormat": "Drift detected for ${detector_id}"
                    }
                ]
            }
        }
        
        return dashboard
    
    def save_dashboard(self, dashboard: Dict[str, Any], filename: str) -> str:
        """
        Save a dashboard to a JSON file.
        
        Args:
            dashboard: Dashboard configuration
            filename: Name of the file to save
            
        Returns:
            Path to the saved file
        """
        file_path = os.path.join(self.dashboard_dir, filename)
        
        # Add meta information
        full_dashboard = {
            "dashboard": dashboard,
            "overwrite": True,
            "message": f"Updated at {datetime.now().isoformat()}"
        }
        
        with open(file_path, 'w') as f:
            json.dump(full_dashboard, f, indent=2)
        
        logger.info(f"Saved Grafana dashboard to {file_path}")
        return file_path
    
    def upload_dashboard(self, dashboard: Dict[str, Any], folder: Optional[str] = None) -> Optional[str]:
        """
        Upload a dashboard to Grafana.
        
        Args:
            dashboard: Dashboard configuration
            folder: Grafana folder to save to
            
        Returns:
            Dashboard URL if successful, None otherwise
        """
        if not self.grafana_url or not self.token:
            logger.warning("Grafana URL or token not provided. Cannot upload dashboard.")
            return None
        
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        # Add folder if specified
        payload = {
            "dashboard": dashboard,
            "overwrite": True
        }
        
        if folder:
            # Get or create folder
            folder_id = self._get_or_create_folder(folder)
            if folder_id:
                payload["folderId"] = folder_id
        
        # Send request
        try:
            response = requests.post(
                f"{self.grafana_url.rstrip('/')}/api/dashboards/db",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                dashboard_url = f"{self.grafana_url}/d/{result['uid']}"
                logger.info(f"Dashboard uploaded successfully: {dashboard_url}")
                return dashboard_url
            else:
                logger.warning(f"Failed to upload dashboard: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.warning(f"Error uploading dashboard: {str(e)}")
            return None
    
    def _get_or_create_folder(self, folder_name: str) -> Optional[int]:
        """
        Get or create a Grafana folder.
        
        Args:
            folder_name: Name of the folder
            
        Returns:
            Folder ID if successful, None otherwise
        """
        if not self.grafana_url or not self.token:
            return None
        
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        # Try to get folder
        try:
            response = requests.get(
                f"{self.grafana_url.rstrip('/')}/api/folders",
                headers=headers
            )
            
            if response.status_code == 200:
                folders = response.json()
                for folder in folders:
                    if folder["title"] == folder_name:
                        return folder["id"]
                
                # Create folder if it doesn't exist
                create_response = requests.post(
                    f"{self.grafana_url.rstrip('/')}/api/folders",
                    headers=headers,
                    json={"title": folder_name}
                )
                
                if create_response.status_code == 200:
                    return create_response.json()["id"]
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting or creating folder: {str(e)}")
            return None
    
    def _generate_model_performance_panels(self, datasource: str) -> List[Dict[str, Any]]:
        """
        Generate panels for model performance dashboard.
        
        Args:
            datasource: Grafana datasource name
            
        Returns:
            List of panel configurations
        """
        panels = []
        
        # Panel 1: Model Accuracy
        panels.append({
            "id": 1,
            "title": "Model Accuracy",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            "datasource": datasource,
            "targets": [
                {
                    "expr": "model_accuracy{model_id=~\"$model\"}",
                    "legendFormat": "{{model_id}}",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "palette-classic"},
                    "custom": {
                        "axisLabel": "Accuracy",
                        "axisPlacement": "auto",
                        "barAlignment": 0,
                        "drawStyle": "line",
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "hideFrom": {"legend": False, "tooltip": False, "viz": False},
                        "lineInterpolation": "smooth",
                        "lineWidth": 2,
                        "pointSize": 5,
                        "scaleDistribution": {"type": "linear"},
                        "showPoints": "never",
                        "spanNulls": True,
                        "stacking": {"group": "A", "mode": "none"},
                        "thresholdsStyle": {"mode": "line"}
                    },
                    "mappings": [],
                    "min": 0,
                    "max": 1,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "red", "value": 0},
                            {"color": "orange", "value": 0.7},
                            {"color": "green", "value": 0.9}
                        ]
                    },
                    "unit": "percentunit"
                },
                "overrides": []
            },
            "options": {
                "legend": {"displayMode": "table", "placement": "right", "showLegend": True},
                "tooltip": {"mode": "multi", "sort": "desc"}
            }
        })
        
        # Panel 2: Model Latency
        panels.append({
            "id": 2,
            "title": "Model Latency",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
            "datasource": datasource,
            "targets": [
                {
                    "expr": "model_latency_seconds{model_id=~\"$model\"}",
                    "legendFormat": "{{model_id}}",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "palette-classic"},
                    "custom": {
                        "axisLabel": "Latency",
                        "axisPlacement": "auto",
                        "barAlignment": 0,
                        "drawStyle": "line",
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "hideFrom": {"legend": False, "tooltip": False, "viz": False},
                        "lineInterpolation": "smooth",
                        "lineWidth": 2,
                        "pointSize": 5,
                        "scaleDistribution": {"type": "linear"},
                        "showPoints": "never",
                        "spanNulls": True,
                        "stacking": {"group": "A", "mode": "none"},
                        "thresholdsStyle": {"mode": "line"}
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": 0},
                            {"color": "orange", "value": 0.5},
                            {"color": "red", "value": 1}
                        ]
                    },
                    "unit": "s"
                },
                "overrides": []
            },
            "options": {
                "legend": {"displayMode": "table", "placement": "right", "showLegend": True},
                "tooltip": {"mode": "multi", "sort": "desc"}
            }
        })
        
        # Panel 3: Prediction Count
        panels.append({
            "id": 3,
            "title": "Prediction Count",
            "type": "stat",
            "gridPos": {"h": 8, "w": 8, "x": 0, "y": 8},
            "datasource": datasource,
            "targets": [
                {
                    "expr": "sum(model_predictions_total{model_id=~\"$model\"}) by (model_id)",
                    "legendFormat": "{{model_id}}",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "blue", "value": 0}
                        ]
                    },
                    "unit": "none"
                },
                "overrides": []
            },
            "options": {
                "colorMode": "value",
                "graphMode": "area",
                "justifyMode": "auto",
                "orientation": "auto",
                "reduceOptions": {
                    "calcs": ["lastNotNull"],
                    "fields": "",
                    "values": False
                },
                "text": {},
                "textMode": "auto"
            }
        })
        
        # Panel 4: Error Rate
        panels.append({
            "id": 4,
            "title": "Error Rate",
            "type": "gauge",
            "gridPos": {"h": 8, "w": 8, "x": 8, "y": 8},
            "datasource": datasource,
            "targets": [
                {
                    "expr": "sum(model_errors_total{model_id=~\"$model\"}) by (model_id) / sum(model_predictions_total{model_id=~\"$model\"}) by (model_id)",
                    "legendFormat": "{{model_id}}",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "mappings": [],
                    "max": 1,
                    "min": 0,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": 0},
                            {"color": "orange", "value": 0.05},
                            {"color": "red", "value": 0.1}
                        ]
                    },
                    "unit": "percentunit"
                },
                "overrides": []
            },
            "options": {
                "orientation": "auto",
                "reduceOptions": {
                    "calcs": ["lastNotNull"],
                    "fields": "",
                    "values": False
                },
                "showThresholdLabels": False,
                "showThresholdMarkers": True
            }
        })
        
        # Panel 5: Retraining Recommendation
        panels.append({
            "id": 5,
            "title": "Retraining Recommendation",
            "type": "stat",
            "gridPos": {"h": 8, "w": 8, "x": 16, "y": 8},
            "datasource": datasource,
            "targets": [
                {
                    "expr": "model_retrain_recommended{model_id=~\"$model\"}",
                    "legendFormat": "{{model_id}}",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "mappings": [
                        {
                            "options": {
                                "0": {"text": "No Retraining Needed", "color": "green"},
                                "1": {"text": "Retraining Recommended", "color": "red"}
                            },
                            "type": "value"
                        }
                    ],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": 0},
                            {"color": "red", "value": 1}
                        ]
                    }
                },
                "overrides": []
            },
            "options": {
                "colorMode": "background",
                "graphMode": "none",
                "justifyMode": "auto",
                "orientation": "auto",
                "reduceOptions": {
                    "calcs": ["lastNotNull"],
                    "fields": "",
                    "values": False
                },
                "textMode": "auto"
            }
        })
        
        return panels
    
    def _generate_drift_detection_panels(self, datasource: str) -> List[Dict[str, Any]]:
        """
        Generate panels for drift detection dashboard.
        
        Args:
            datasource: Grafana datasource name
            
        Returns:
            List of panel configurations
        """
        panels = []
        
        # Panel 1: Drift Score
        panels.append({
            "id": 1,
            "title": "Drift Score",
            "type": "timeseries",
            "gridPos": {"h": 9, "w": 24, "x": 0, "y": 0},
            "datasource": datasource,
            "targets": [
                {
                    "expr": "drift_score{detector_id=~\"$detector\"}",
                    "legendFormat": "{{detector_id}}",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "palette-classic"},
                    "custom": {
                        "axisLabel": "Drift Score",
                        "axisPlacement": "auto",
                        "barAlignment": 0,
                        "drawStyle": "line",
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "hideFrom": {"legend": False, "tooltip": False, "viz": False},
                        "lineInterpolation": "smooth",
                        "lineWidth": 2,
                        "pointSize": 5,
                        "scaleDistribution": {"type": "linear"},
                        "showPoints": "never",
                        "spanNulls": True,
                        "stacking": {"group": "A", "mode": "none"},
                        "thresholdsStyle": {"mode": "line"}
                    },
                    "mappings": [],
                    "min": 0,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": 0},
                            {"color": "orange", "value": 0.5},
                            {"color": "red", "value": 0.8}
                        ]
                    }
                },
                "overrides": []
            },
            "options": {
                "legend": {"displayMode": "table", "placement": "right", "showLegend": True},
                "tooltip": {"mode": "multi", "sort": "desc"}
            }
        })
        
        # Panel 2: Drift Detected
        panels.append({
            "id": 2,
            "title": "Drift Detected",
            "type": "stat",
            "gridPos": {"h": 8, "w": 8, "x": 0, "y": 9},
            "datasource": datasource,
            "targets": [
                {
                    "expr": "drift_detected{detector_id=~\"$detector\"}",
                    "legendFormat": "{{detector_id}}",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "mappings": [
                        {
                            "options": {
                                "0": {"text": "No Drift", "color": "green"},
                                "1": {"text": "Drift Detected", "color": "red"}
                            },
                            "type": "value"
                        }
                    ],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": 0},
                            {"color": "red", "value": 1}
                        ]
                    }
                },
                "overrides": []
            },
            "options": {
                "colorMode": "background",
                "graphMode": "none",
                "justifyMode": "auto",
                "orientation": "auto",
                "reduceOptions": {
                    "calcs": ["lastNotNull"],
                    "fields": "",
                    "values": False
                },
                "textMode": "auto"
            }
        })
        
        # Panel 3: Drift Detector Metrics
        panels.append({
            "id": 3,
            "title": "Drift Samples Processed",
            "type": "gauge",
            "gridPos": {"h": 8, "w": 8, "x": 8, "y": 9},
            "datasource": datasource,
            "targets": [
                {
                    "expr": "drift_samples_processed{detector_id=~\"$detector\"}",
                    "legendFormat": "{{detector_id}}",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "blue", "value": 0}
                        ]
                    },
                    "unit": "none"
                },
                "overrides": []
            },
            "options": {
                "orientation": "auto",
                "reduceOptions": {
                    "calcs": ["lastNotNull"],
                    "fields": "",
                    "values": False
                },
                "showThresholdLabels": False,
                "showThresholdMarkers": True
            }
        })
        
        # Panel 4: Drift Detection Rate
        panels.append({
            "id": 4,
            "title": "Drift Detection Rate",
            "type": "gauge",
            "gridPos": {"h": 8, "w": 8, "x": 16, "y": 9},
            "datasource": datasource,
            "targets": [
                {
                    "expr": "drift_detected_count{detector_id=~\"$detector\"} / drift_samples_processed{detector_id=~\"$detector\"}",
                    "legendFormat": "{{detector_id}}",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "mappings": [],
                    "max": 1,
                    "min": 0,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": 0},
                            {"color": "orange", "value": 0.1},
                            {"color": "red", "value": 0.3}
                        ]
                    },
                    "unit": "percentunit"
                },
                "overrides": []
            },
            "options": {
                "orientation": "auto",
                "reduceOptions": {
                    "calcs": ["lastNotNull"],
                    "fields": "",
                    "values": False
                },
                "showThresholdLabels": False,
                "showThresholdMarkers": True
            }
        })
        
        return panels

def generate_model_drift_dashboard(
    output_file: str = "model_drift_dashboard.json",
    title: str = "ML Model Drift Dashboard",
    folder: str = "ML Observability"
) -> str:
    """
    Generate a Grafana dashboard for model drift metrics.
    
    Args:
        output_file: Name of the output file
        title: Dashboard title
        folder: Grafana folder
        
    Returns:
        Path to the saved dashboard file
    """
    generator = GrafanaDashboardGenerator()
    dashboard = generator.generate_drift_detection_dashboard(title=title, folder=folder)
    return generator.save_dashboard(dashboard, output_file)

def generate_model_performance_dashboard(
    output_file: str = "model_performance_dashboard.json",
    title: str = "ML Model Performance Dashboard",
    folder: str = "ML Observability"
) -> str:
    """
    Generate a Grafana dashboard for model performance metrics.
    
    Args:
        output_file: Name of the output file
        title: Dashboard title
        folder: Grafana folder
        
    Returns:
        Path to the saved dashboard file
    """
    generator = GrafanaDashboardGenerator()
    dashboard = generator.generate_model_performance_dashboard(title=title, folder=folder)
    return generator.save_dashboard(dashboard, output_file)

def upload_dashboard_to_grafana(
    dashboard_file: str,
    grafana_url: str,
    token: str,
    folder: str = "ML Observability"
) -> Optional[str]:
    """
    Upload a dashboard to Grafana.
    
    Args:
        dashboard_file: Path to the dashboard JSON file
        grafana_url: Grafana server URL
        token: Grafana service account token
        folder: Grafana folder
        
    Returns:
        Dashboard URL if successful, None otherwise
    """
    try:
        # Load dashboard from file
        with open(dashboard_file, 'r') as f:
            dashboard_json = json.load(f)
        
        # Upload to Grafana
        generator = GrafanaDashboardGenerator(grafana_url=grafana_url, token=token)
        dashboard_data = dashboard_json.get("dashboard", dashboard_json)
        return generator.upload_dashboard(dashboard_data, folder)
    except Exception as e:
        logger.warning(f"Error uploading dashboard to Grafana: {str(e)}")
        return None 