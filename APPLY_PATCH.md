# Apply the v0.2.1 full-data patch

From the repository root in Windows PowerShell:

```powershell
Expand-Archive `
  -Path .\alignment-gap-nature-checklist-v0.2.1-no-demo-patch.zip `
  -DestinationPath . `
  -Force
```

If v0.2.0 was already applied, remove the obsolete simulated-demo files:

```powershell
Remove-Item .\config\demo.yml -Force -ErrorAction SilentlyContinue
Get-ChildItem .\data\demo -Recurse -File -ErrorAction SilentlyContinue | Remove-Item -Force
Remove-Item .\scripts\prepare_demo_data.py -Force -ErrorAction SilentlyContinue
Remove-Item .\scripts\run_demo.py -Force -ErrorAction SilentlyContinue
Remove-Item .\src\alignment_gap\demo.py -Force -ErrorAction SilentlyContinue
```

Then reinstall and verify:

```powershell
python -m pip install -e ".[dev]"
python scripts\validate_public_data.py
python scripts\reproduce_all.py
python scripts\report_environment.py --output results\environment.json
pytest
```

Review the staged files before committing:

```powershell
git status --short
git add .gitignore .github CITATION.cff LICENSE Makefile README.md PATCH_NOTES_v0.2.1.md pyproject.toml data\README.md docs scripts src tests\test_pipeline.py
git add -u config data\demo scripts src\alignment_gap tests
git diff --cached --name-status
git diff --cached --stat
git commit -m "Use full public data for reproducibility checks"
git push origin main
```

Do not add `data/temp`, the anonymization salt, private identifier maps, or generated `results/` files.
