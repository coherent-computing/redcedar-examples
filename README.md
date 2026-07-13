# Red Cedar Examples

Example notebooks demonstrating the Red Cedar quantum machine learning pipeline.

## Prerequisites

Install the Red Cedar client:

```bash
cd ../redcedar
pip install -e .
```

Set your server connection:

```bash
export REDCEDAR_URL="http://localhost:8000"       # local server
export REDCEDAR_API_KEY="your-api-key"            # API key
```

Or configure directly in the notebook:

```python
import redcedar
client = redcedar.connect(url="http://localhost:8002", api_key="your-key")
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| **01_full_pipeline.ipynb** | Complete workflow: dataset selection, PCA, Mutual Information, Resource Estimation, training, inference, and Explainable AI |

## About

These examples use the Red Cedar Python client to interact with a Red Cedar server. All quantum computation happens on the server — these notebooks require no quantum libraries.
