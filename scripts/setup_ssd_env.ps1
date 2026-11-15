# Rebuild the environment on SSD; keep the exact installed package versions.
$ErrorActionPreference='Stop'
$root='Q:\AVITOCUP'
$project="$root\AvitoTech_ML_Cup"
while (!(Test-Path "$root\uv.exe")) { Start-Sleep 2 }
New-Item -ItemType Directory -Force $project | Out-Null
Start-Transcript -Path "$root\environment-setup.log" -Append
$env:UV_PYTHON_INSTALL_DIR="$root\python"
$env:UV_LINK_MODE='copy'
$oldPython='E:\AVITOCUP\AvitoTech_ML_Cup\.venv\Scripts\python.exe'
$version=(& $oldPython -c 'import platform; print(platform.python_version())').Trim()
$packages=& "$root\uv.exe" pip freeze --python $oldPython
if ($LASTEXITCODE) { throw 'Cannot inspect existing environment' }
[IO.File]::WriteAllLines("$root\environment-before.txt",[string[]]$packages,[Text.UTF8Encoding]::new($false))
& "$root\uv.exe" python install $version
if ($LASTEXITCODE) { throw 'Python installation failed' }
$pythonDirectory=(Get-ChildItem "$root\python" -Directory -Filter "cpython-$version-*" | Select-Object -First 1).FullName
if (!$pythonDirectory.StartsWith($root)) { throw 'Unexpected Python location' }
$python=Join-Path $pythonDirectory 'python.exe'
if (!(Test-Path "$project\.venv\Scripts\python.exe")) {
    & "$root\uv.exe" venv --python $python "$project\.venv"
    if ($LASTEXITCODE) { throw 'Environment creation failed' }
}
& "$root\uv.exe" pip install --python "$project\.venv\Scripts\python.exe" -r "$root\environment-before.txt" --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match --link-mode copy
if ($LASTEXITCODE) { throw 'Package installation failed' }
& "$project\.venv\Scripts\python.exe" -c 'import sys,torch; print(sys.prefix,sys.base_prefix,torch.__file__,torch.__version__); assert torch.cuda.is_available(); assert ord(sys.prefix[0])==81'
if ($LASTEXITCODE) { throw 'SSD environment/CUDA validation failed' }
[IO.File]::WriteAllText("$root\environment-ready.json",'{"ready":true}',[Text.UTF8Encoding]::new($false))
Stop-Transcript
