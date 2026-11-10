# MLflow Pyfunc Wrapper for Inference: Benefits & Usage

This document explains why using an `mlflow.pyfunc` wrapper (`PythonModel`) is beneficial during model inference, especially when working with custom ML models (e.g., pure NumPy implementations).

---

## Overview

While tracking metrics and saving raw `.pkl` files covers basic experiment logging, wrapping custom models into MLflow's standard `pyfunc` interface unlocks production-grade MLOps capabilities during inference.

---

## Key Benefits at Inference Time

### 1. Unified & Agnostic Prediction API
* **Without Wrapper:** Your inference script must import original source classes, instantiate objects manually, and call model-specific methods (e.g., `model.forward(X)` or `model.predict(X)`).
* **With Wrapper:** Provides a standardized interface regardless of the underlying framework (NumPy, PyTorch, Keras, or Scikit-Learn):
  ```python
  import mlflow

  # Standardized loading across all model types
  model = mlflow.pyfunc.load_model(f"runs:/{run_id}/numpy_model")

  # Uniform prediction interface
  predictions = model.predict(X_new)
  ```

### 2. Absolute Traceability & Dynamic Run Selection
* **Without Wrapper:** Relying on hardcoded local file paths (e.g., `data/processed_data/numpy_mlp_model.pkl`) creates brittle pipelines prone to broken paths during migration or multi-run comparisons.
* **With Wrapper:** Models are loaded directly using unique MLflow identifiers (`runs:/<run_id>/numpy_model` or registered model names). Applications like Streamlit can dynamically pull and execute any historical model run on demand.

### 3. Environment & Dependency Consistency
* When logging via `mlflow.pyfunc.log_model`, MLflow automatically packages execution details and dependencies.
* Guarantees that reloading the model on a different machine, server, or Docker container executes under identical runtime conditions, eliminating library version mismatches.

### 4. Zero-Code Production Deployment (Model Serving)
* **Without Wrapper:** Exposing the model via a REST API requires writing custom Flask/FastAPI boilerplate code to deserialize pickles and handle requests.
* **With Wrapper:** MLflow can spin up a production-ready REST API endpoint natively via CLI without additional inference code:
  ```bash
  mlflow models serve -m "runs:/<run_id>/numpy_model" --port 5001
  ```

---

## Summary Matrix

| Feature | Direct `.pkl` Load | MLflow `pyfunc` Wrapper |
| :--- | :---: | :---: |
| **Local Script Inference** | ✅ Simple & Fast | ✅ Supported |
| **Framework Agnostic API** | ❌ No | ✅ Yes (`model.predict()`) |
| **Dynamic Run Retrieval** | ❌ Hardcoded paths | ✅ Built-in via Run ID |
| **Automated Model Serving** | ❌ Manual API code | ✅ Out-of-the-box CLI |
| **Production Ready** | ❌ Low | ✅ High |

---

## Basic Implementation Template

```python
import pickle
import mlflow
import mlflow.pyfunc
from mlflow.pyfunc import PythonModelContext

class CustomModelWrapper(mlflow.pyfunc.PythonModel):
    """Generic MLflow wrapper for custom models."""

    def load_context(self, context: PythonModelContext) -> None:
        """Load the underlying model object from logged artifacts."""
        with open(context.artifacts["model_file"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context: PythonModelContext, model_input: object) -> object:
        """Standardized prediction call."""
        if hasattr(model_input, "to_numpy"):
            model_input = model_input.to_numpy()
        return self.model.predict(model_input)
```