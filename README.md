# brainmodel_utils
Basic utilities for comparing models to neural & behavioral data, along with packaging these data in Python (from Matlab).
Used in our [NeuroAI Turing Test](https://arxiv.org/abs/2502.16238).

# Installation
To install run:
```
git clone https://github.com/neuroagents-lab/brainmodel_utils
cd brainmodel_utils/
pip install -e .
```

# Usage:
After installing the package, you can use it as follows:

```
from brainmodel_utils.metrics.consistency import get_linregress_consistency
import numpy as np

## Example data (substitute with your actual model/data!)
source = np.random.randn(100, 50)  # 100 stimuli, 50 model features

# Alternatively, if `source` is from another animal (rather than a model), you can include a trials dimension:
target = np.random.randn(20, 100, 50)  # 20 trials, 100 stimuli, 50 units

## Linear regression parameters (we usually recommend Ridge with a user defined alpha that has been cross-validated
# via median consistency on a val set of your choosing!)
alpha = 1.0
map_kwargs = {
                "map_type": "sklinear",
                "map_kwargs": {
                    "regression_type": "Ridge",
                    "regression_kwargs": {"alpha": alpha},
                },
            }
# see below for other map types

## Compute consistency
consistency_results = get_linregress_consistency(
    source=source,
    target=target,
    map_kwargs=map_kwargs,
    num_bootstrap_iters=1000,  # Number of bootstrap split-half iterations
    num_parallel_jobs=100,     # Parallelization across split-halves for speed
    start_seed=42,             # Reproducibility seed
    metric="pearsonr"          # Use "rsa_pearsonr/spearmanr" for RSA instead,
                               # in which case use an Identity Map (see below)
)

print(consistency_results)
```

We support a range of standard map types (listed [here](https://github.com/neuroagents-lab/brainmodel_utils/blob/main/brainmodel_utils/neural_mappers/__init__.py)), from no mapping (`IdentityNeuralMap`) used for RSA (implemented [here](https://github.com/neuroagents-lab/brainmodel_utils/blob/main/brainmodel_utils/metrics/utils.py#L86-L89)), to 1-to-1 simple correlation-based mapping of units from source to target (`PercentileNeuralMap`), Partial Least Squares (`PLSNeuralMap`), and the [`scikit-learn`](https://scikit-learn.org/) mapping functions like Ridge/Lasso/ElasticNet (`SKLinearNeuralMap`).

Example usages are below for each, but feel free to PR your own!

```python
# Identity (only for use with RSA via "rsa_pearsonr/spearmanr")
map_kwargs = {"map_type": "identity"}

# 1-to-1 mapping, finds the best (100th percentile) unit in source to match to each target unit, on train set
map_kwargs = {"map_type": "percentile"}

# 1-to-1 mapping using 95th percentile group rather than absolute best source unit
alpha = 1.0  # we strongly recommend cross-validating this!
map_kwargs = {
    "map_type": "percentile",
    "map_kwargs": {
        "percentile": 95,
        "identity": False,  # must be set if percentile < 100, since you can no longer use the identity transform
        "regression_type": "Ridge",
        "regression_kwargs": {"alpha": alpha},
    },
}

# ElasticNet regression
alpha = 1.0  # we strongly recommend cross-validating this!
l1_ratio = 0.5  # we strongly recommend cross-validating this!
map_kwargs = {
    "map_type": "sklinear",
    "map_kwargs": {
        "regression_type": "ElasticNet",
        "regression_kwargs": {"alpha": alpha, "l1_ratio": l1_ratio},
    },
}

# Partial Least Squares (PLS) regression. Note we do not use `sklinear` for this,
# since `scikit-learn` sets `scale=True` by default for PLS,
# which is **not** what we want for neural data (we want `scale=False`).
map_kwargs = {
    "map_type": "pls",
    "map_kwargs": {
        "n_components": 25,  # we recommend cross-validating this, or going as high as feasible; e.g., 100 components is good too!
    },
}
```

Under the hood, [`PipelineNeuralMap`](https://github.com/neuroagents-lab/brainmodel_utils/blob/main/brainmodel_utils/neural_mappers/pipeline_neural_map.py) is called for each of these, which can chain these mappings with additional factorization if you like, by passing in `factor_kwargs`.

# Optional CUDA backend

CPU behavior is unchanged: `backend` defaults to `"cpu"`, and the normal installation does not require PyTorch, CuPy, cuML, or an NVIDIA GPU. GPU libraries are imported only when a CUDA-capable path is requested.

The CUDA 12 environment used for development and RTX 3080 validation used Python 3.12, RAPIDS 25.10, and `scikit-learn<1.8`:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index https://pypi.nvidia.com \
  --index-strategy unsafe-best-match \
  -e . \
  "scikit-learn<1.8" \
  "cuml-cu12==25.10.*" \
  "cupy-cuda12x[ctk]" \
  torch
```

Use a CuPy and RAPIDS wheel matching the installed CUDA major version; see the [RAPIDS installation selector](https://docs.rapids.ai/install/) and [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) for other environments. CuPy and cuML provide CUDA Ridge, while CUDA percentile correlations also require a CUDA-enabled PyTorch build.

The same Ridge configuration can select a backend explicitly:

```python
# Existing code stays on CPU because the backend defaults to "cpu".
cpu_map_kwargs = {
    "map_type": "sklinear",
    "map_kwargs": {
        "regression_type": "Ridge",
        "regression_kwargs": {"alpha": 1.0},
    },
}

# Require CUDA and fail clearly if it cannot honor the request.
cuda_map_kwargs = {
    "map_type": "sklinear",
    "map_kwargs": {
        "regression_type": "Ridge",
        "regression_kwargs": {"alpha": 1.0, "fit_intercept": True},
        "backend": "cuda",
    },
}

# Prefer CUDA, but retain CPU behavior when CUDA or an option is unsupported.
auto_map_kwargs = {
    "map_type": "sklinear",
    "map_kwargs": {
        "regression_type": "Ridge",
        "regression_kwargs": {"alpha": 1.0},
        "backend": "auto",
    },
}
```

CUDA Ridge supports only `regression_type="Ridge"` with a finite, non-negative scalar `alpha`, `fit_intercept`, and `solver` omitted or set to `"auto"` or `"svd"`. `backend="auto"` falls back to scikit-learn on unsupported regression types, options, installed cuML signatures, or unavailable CUDA; `backend="cuda"` raises instead of silently changing the computation.

RAPIDS cuML 25.10 does not provide the sklearn-compatible native multi-target behavior needed here. For the common `solver="auto"`, `alpha > 0` case, brainmodel_utils uses a batched CuPy closed-form Ridge solve, including grouped fits whose percentile-selected features differ by target. Other supported CUDA configurations use cuML estimators target by target.

CUDA percentile correlations always use float32. TF32 matrix multiplication is disabled for the correlation calculation and its previous PyTorch setting is restored afterward, which avoids reduced-mantissa TF32 changing percentile rankings. Ridge preserves float64 when either input is float64 and otherwise uses float32; use float64 Ridge inputs when large feature offsets would make float32 centering inaccurate.

For a large CUDA percentile fit whose full source-by-target correlation matrix is not needed afterward, set `store_correlations=False`; thresholds and selected source indices are still retained. The benchmark harness applies this optimization automatically on CUDA.

Use one process for a single-GPU consistency run:

```python
consistency_results = get_linregress_consistency(
    source=source,
    target=target,
    map_kwargs=cuda_map_kwargs,
    num_bootstrap_iters=1000,
    num_parallel_jobs=1,
    start_seed=42,
    metric="pearsonr",
)
```

CUDA consistency rejects `num_parallel_jobs != 1` so independent joblib processes cannot compete for the same device. See the [benchmark protocol](benchmarks/README.md) for CPU/CUDA commands, the real fly-data workload, recorded provenance, and reference-result comparisons.

# Return Type:
The function returns a dictionary of values. The most relevant one, which contains the predictivity per neuron is **`"r_xy_n_sb"`**.

The convention is that:
- **`X`** refers to the source (brain or model)
- **`Y`** refers to the target brain
- **`r`** is the Pearson or Spearman correlation
- **`n`** indicates that the metric is computed per neuron
- **`sb`** refers to the Spearman-Brown correction applied to the split-half consistencies in the denominator

With this convention in mind, we now explain each of the keys and values of the dictionary that is returned:

---

### High-Level Structure

The dictionary has two top-level keys: **`"train"`** and **`"test"`**, each mapping to another dictionary containing the following metrics:

- **`r_xy_n_sb`**  
  Noise-corrected correlation (Spearman-Brown corrected) computed per neuron.  
  This is the primary metric indicating predictivity of the source–target mapping.

- **`r_xx`**  
  Split-half correlation of the source’s predicted responses.

- **`r_xx_sb`**  
  Spearman-Brown–corrected split-half correlation of the source’s predicted responses.

- **`r_yy`**  
  Split-half correlation of the actual target brain responses.

- **`r_yy_sb`**  
  Spearman-Brown–corrected split-half correlation of the actual target brain responses.

- **`r_xy`**  
  Raw correlation between predicted and actual responses (uncorrected for noise).

- **`denom_sb`**  
  The denominator used in the noise-corrected correlation (the full "Statistical Noise Ceiling"):
  ```math
  \text{denom\_sb} = \sqrt{\left(\text{r\_xx\_sb}\right) \cdot \left(\text{r\_yy\_sb}\right)}
  ```
  If this value is undefined (e.g., negative or zero), **`r_xy_n_sb`** is set to NaN.

---

### Example of the Returned Structure

```python
{
    "train": {
        "r_xy_n_sb": <[bootstraps, splits, units]>,
        "r_xx":      <[bootstraps, splits, units]>,
        "r_xx_sb":   <[bootstraps, splits, units]>,
        "r_yy":      <[bootstraps, splits, units]>,
        "r_yy_sb":   <[bootstraps, splits, units]>,
        "r_xy":      <[bootstraps, splits, units]>,
        "denom_sb":  <[bootstraps, splits, units]>
    },
    "test": {
        "r_xy_n_sb": <[bootstraps, splits, units]>,
        "r_xx":      <[bootstraps, splits, units]>,
        "r_xx_sb":   <[bootstraps, splits, units]>,
        "r_yy":      <[bootstraps, splits, units]>,
        "r_yy_sb":   <[bootstraps, splits, units]>,
        "r_xy":      <[bootstraps, splits, units]>,
        "denom_sb":  <[bootstraps, splits, units]>
    }
}
```

# License
MIT

# Contact
If you have any questions or encounter issues, either submit a Github issue here (preferred) or [email me](https://anayebi.github.io/contact/).
