$testFiles = Get-ChildItem -Path "tests" -Filter "*.py" | Where-Object { $_.Name -ne "__init__.py" }

$pythonPath = & python -c "import sys; print(sys.executable)"

foreach ($file in $testFiles) {
    & $pythonPath -m unittest $file.FullName
}