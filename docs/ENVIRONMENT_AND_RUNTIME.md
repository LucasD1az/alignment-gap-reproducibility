# Environment and runtime record

The complete reproduction writes a machine-readable report automatically:

```text
results/reproduction_runtime.json
```

Create a standalone environment report with:

```bash
python scripts/report_environment.py --output results/environment.json
```

The specific versions and hardware used during testing is:

| Measurement | Value |
|---|---|
| Operating system | Windows 11 |
| Python version | 3.14.2 |
| CPU and RAM | Intel64 Family 6 Model 141 Stepping 1, GenuineIntel 16 core and 31.78 GB RAM |
| Installation time | 58.63 s |
| Full reproduction time | 226.44 s |

And the packages versions are:

| Package | Tested version |
|---|---:|
| NumPy | 2.5.1 |
| pandas | 3.0.5 |
| SciPy | 1.18.0 |
| Matplotlib | 3.11.1 |
| PyYAML | 6.0.3 |
| Plotly | 6.9.0 |
| Kaleido | 1.3.0 |
| Pillow | 12.3.0 |
| pytest (development) | 9.1.1 |
| Ruff (development) | 0.16.1 |


The GitHub Actions workflow provides an additional reproducible test record on Ubuntu 24.04 with Python 3.10–3.12. The complete full-data reproduction is run in continuous integration on Python 3.11.
