[CmdletBinding()]
param(
    [string]$Version = "0.1.0",
    [switch]$Sign,
    # SHA-1 thumbprint of the code-signing certificate in the certificate store (signtool /sha1).
    # (Was misleadingly called -CertificateSubject.)
    [string]$CertificateThumbprint = "",
    # Build with a specific interpreter instead of the default lookup below (.venv, then "python"
    # on PATH). PyInstaller bundles whatever torch the interpreter running it has, so building from
    # a CPU-only torch ships a CPU-only app -- see the CUDA guard below.
    [string]$Python = "",
    # PyTorch wheel index torch is installed from before requirements.txt (PyPI's Windows torch
    # wheel is CPU-only). cu130 matches the last shipped build (torch 2.14.0+cu130) and needs an
    # NVIDIA driver >= 580; RTX 50-series (sm_120) needs CUDA 12.8 or newer either way. If torch is
    # already installed, pip leaves it alone -- the guard below still checks what's there. Pass ""
    # to skip this step (e.g. with -AllowCpu).
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu130",
    # Allow building with a CPU-only torch. Without it, the build stops before PyInstaller when
    # torch has no CUDA support, instead of silently shipping an app that can't use the GPU.
    [switch]$AllowCpu,
    # Skip the 7z archive. A CUDA build bundles several GB of libraries, and -mx=9 LZMA over that
    # takes a long time -- skip it when only the unpacked release\ folder is needed.
    [switch]$SkipArchive,
    # Keep PyInstaller's build-release\ and dist-release\ folders after a successful build instead of
    # deleting them. build-release\ is PyInstaller's analysis cache, so keeping it makes the next
    # rebuild faster at the cost of a few hundred MB; dist-release\ is empty either way, since its
    # output is moved into release\.
    [switch]$KeepBuildFiles,
    # GitHub release assets are limited to 2 GiB each. A larger archive is re-made as 7z split
    # volumes (PixelFlow-<version>-Windows.7z.001, .002, ...) of this size -- 7-Zip opens the .001.
    [long]$MaxAssetBytes = 1900MB
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    # The project's own .venv first, then whatever "python" resolves to on PATH (e.g. a CI runner
    # set up via actions/setup-python). No machine-specific paths: pass -Python for anything else.
    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $Python)) {
        $onPath = Get-Command python -ErrorAction SilentlyContinue
        if ($onPath) { $Python = $onPath.Source }
    }
}
$Dist = Join-Path $ProjectRoot "dist-release"
$Build = Join-Path $ProjectRoot "build-release"
# The only unpacked copy of the finished app: a fixed name, so the built .exe always has one stable,
# predictable place to find it (release\PixelFlow.exe) without knowing the current version number.
# Only the distributable archive carries the version in its name.
$ReleaseFolder = Join-Path $ProjectRoot "release"
$Archive = Join-Path $ProjectRoot "PixelFlow-$Version-Windows.7z"
# Older versions of this script also kept a version-named copy of release\ here -- a full duplicate
# of a multi-GB CUDA build. It's no longer created; a leftover one is removed after a successful build.
$LegacyPackage = Join-Path $ProjectRoot "PixelFlow-$Version-Windows"

# Microsoft Edge WebView2 Runtime is required to open the app's window at all (pywebview's
# edgechromium backend needs it). A pinned "Fixed Version" copy is bundled as a sibling
# "WebView2Runtime" folder next to PixelFlow.exe, and desktop.py points the WebView2 loader at it via
# the WEBVIEW2_BROWSER_EXECUTABLE_FOLDER environment variable -- see the sibling photo_professional_tool project's
# scripts/build_release.ps1 for the original rationale (same runtime, reused here). Unlike the
# normal Evergreen runtime, a bundled Fixed Version copy does NOT auto-update with Chromium
# security patches, so $WebView2Version should be bumped deliberately every so often.
$WebView2Version = "153.0.4234.32"
$WebView2CabName = "Microsoft.WebView2.FixedVersionRuntime.$WebView2Version.x64.cab"
$WebView2Url = "https://msedge.sf.dl.delivery.mp.microsoft.com/filestreamingservice/files/c3d95bc1-a0a7-4ca6-aaa1-fa0ac3dd1a37/$WebView2CabName"
$WebView2CacheDir = Join-Path $ProjectRoot "vendor\webview2-cache"
$WebView2Cab = Join-Path $WebView2CacheDir $WebView2CabName
$WebView2ExtractedRoot = Join-Path $WebView2CacheDir "extracted"
$WebView2RuntimeSource = Join-Path $WebView2ExtractedRoot "Microsoft.WebView2.FixedVersionRuntime.$WebView2Version.x64"
# The sibling photo_professional_tool project already downloaded and extracted this exact pinned
# version for its own build -- reuse its cache instead of re-downloading ~300MB when present.
$SiblingCacheDir = Join-Path (Split-Path -Parent $ProjectRoot) "photo_professional_tool\vendor\webview2-cache"

# Runs a Python snippet from a temp file rather than `python -c`: Windows PowerShell 5.1 mangles
# double quotes inside native-command arguments. Returns the interpreter's exit code.
function Invoke-PythonSnippet([string]$Code, [string[]]$Arguments = @()) {
    $file = Join-Path ([System.IO.Path]::GetTempPath()) ("pixelflow-build-" + [guid]::NewGuid().ToString("N") + ".py")
    [System.IO.File]::WriteAllText($file, $Code, (New-Object System.Text.UTF8Encoding $false))
    try {
        & $Python $file @Arguments | ForEach-Object { Write-Host $_ }
        return $LASTEXITCODE
    } finally {
        Remove-Item $file -ErrorAction SilentlyContinue
    }
}

# Remove-Item -Recurse in a OneDrive-synced folder intermittently fails with "Access denied" while
# OneDrive is still processing the deletions (seen on the first full release build): retry a few
# times before giving up.
function Remove-Tree([string]$Path) {
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        if (-not (Test-Path $Path)) { return }
        try {
            Remove-Item $Path -Recurse -Force -ErrorAction Stop
            return
        } catch {
            if ($attempt -eq 5) { throw }
            Write-Warning "Removing $Path failed (attempt $attempt of 5): $($_.Exception.Message) Retrying..."
            Start-Sleep -Seconds 3
        }
    }
}

if (-not (Test-Path $Python)) {
    throw "Python interpreter not found. Create .venv, put python on PATH, or pass -Python."
}
# Check the signing setup before the multi-minute build, not after it.
if ($Sign) {
    if ([string]::IsNullOrWhiteSpace($CertificateThumbprint)) {
        throw "-CertificateThumbprint is required when -Sign is used."
    }
    # signtool ships with the Windows SDK, which doesn't put it on PATH (GitHub's windows-latest
    # runners included): fall back to the newest SDK's x64 copy.
    $SignTool = Get-Command signtool -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
    if (-not $SignTool -and ${env:ProgramFiles(x86)}) {
        $SignTool = Get-ChildItem (Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin\10.*\x64\signtool.exe") -ErrorAction SilentlyContinue |
            Sort-Object { [version]$_.Directory.Parent.Name } | Select-Object -Last 1 -ExpandProperty FullName
    }
    if (-not $SignTool) { throw "-Sign needs signtool.exe (Windows SDK); it is neither on PATH nor in Windows Kits\10\bin." }
}

Push-Location $ProjectRoot
try {
    # $ErrorActionPreference doesn't cover native commands -- check their exit codes explicitly, so a
    # failed install or build stops here instead of packaging (and cleaning up around) a stale output.
    Write-Host "Building with $Python"
    # requirements-lock.txt pins the exact versions the app was tested with (scripts/make_lock.py);
    # as a constraints file it only fixes versions, it never adds packages.
    $constraints = @()
    if (Test-Path (Join-Path $ProjectRoot "requirements-lock.txt")) { $constraints = @("-c", "requirements-lock.txt") }
    if ($TorchIndexUrl) {
        # Must come before requirements.txt: once any torch satisfies "torch>=2.7", pip won't
        # replace it, and on Windows the PyPI wheel it would otherwise pick is CPU-only. Constrained
        # too: an unpinned newer torch from the index would otherwise be swapped for the locked
        # version from PyPI (CPU-only) by the requirements.txt step below.
        & $Python -m pip install torch --index-url $TorchIndexUrl @constraints
        if ($LASTEXITCODE -ne 0) { throw "pip install torch --index-url $TorchIndexUrl failed (exit code $LASTEXITCODE)." }
    }
    & $Python -m pip install -r requirements.txt @constraints
    if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed (exit code $LASTEXITCODE)." }
    & $Python -m pip install -r requirements-desktop.txt @constraints
    if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements-desktop.txt failed (exit code $LASTEXITCODE)." }

    # CUDA guard: PyInstaller bundles whatever torch this interpreter has. Exit code 3 = CPU-only
    # build, 4 = CUDA older than 12.8 (no RTX 50-series / sm_120 kernels). Checks the build, not a
    # GPU, so it also works on a GPU-less CI runner.
    $torchCheck = @'
import sys, torch
cuda = torch.version.cuda
print(f"torch {torch.__version__}, CUDA {cuda or 'none (CPU-only build)'}")
if not cuda:
    sys.exit(3)
major, minor = (int(p) for p in cuda.split(".")[:2])
sys.exit(0 if (major, minor) >= (12, 8) else 4)
'@
    $torchStatus = Invoke-PythonSnippet $torchCheck
    if ($torchStatus -eq 3 -or $torchStatus -eq 4) {
        $reason = if ($torchStatus -eq 3) { "a CPU-only torch" } else { "a torch built for CUDA older than 12.8 (no RTX 50-series support)" }
        if ($AllowCpu) {
            Write-Warning "Building with $reason (-AllowCpu): the packaged app will not use an RTX 50-series GPU."
        } else {
            throw "Refusing to build with $reason -- the app would silently run on CPU. Install a CUDA torch (e.g. pip install torch --index-url https://download.pytorch.org/whl/cu130) or pass -AllowCpu."
        }
    } elseif ($torchStatus -ne 0) {
        throw "Could not import torch with $Python (exit code $torchStatus)."
    }

    # The version the app reports (/api/health, /api/system, the System panel). Generated, not
    # committed (.gitignore); removed again after the build so a dev checkout reports "dev".
    $VersionFile = Join-Path $ProjectRoot "backend\_version.py"
    [System.IO.File]::WriteAllText($VersionFile, "__version__ = `"$Version`"`n", (New-Object System.Text.UTF8Encoding $false))
    try {
        & $Python -m PyInstaller --noconfirm --distpath $Dist --workpath $Build PixelFlow.spec
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed (exit code $LASTEXITCODE)." }
    } finally {
        Remove-Item $VersionFile -ErrorAction SilentlyContinue
    }

    # Move (not copy) PyInstaller's output into release\: on the same drive that's a rename, so the
    # multi-GB build is never duplicated on disk, not even briefly.
    Remove-Tree $ReleaseFolder
    $oneFolder = Join-Path $Dist "PixelFlow"
    $oneFile = Join-Path $Dist "PixelFlow.exe"
    if (Test-Path $oneFolder) {
        Move-Item $oneFolder $ReleaseFolder
    } elseif (Test-Path $oneFile) {
        New-Item -ItemType Directory -Path $ReleaseFolder | Out-Null
        Move-Item $oneFile (Join-Path $ReleaseFolder "PixelFlow.exe")
    } else {
        throw "PyInstaller output not found in $Dist"
    }

    New-Item -ItemType Directory -Force -Path $WebView2CacheDir | Out-Null
    if (-not (Test-Path $WebView2RuntimeSource) -and (Test-Path (Join-Path $SiblingCacheDir "extracted\Microsoft.WebView2.FixedVersionRuntime.$WebView2Version.x64"))) {
        Write-Host "Reusing WebView2 Runtime $WebView2Version already cached by the sibling project..."
        Copy-Item (Join-Path $SiblingCacheDir "extracted\Microsoft.WebView2.FixedVersionRuntime.$WebView2Version.x64") $WebView2RuntimeSource -Recurse
    }
    # A >100MB sanity check guards against a previous run being interrupted mid-download.
    if (-not (Test-Path $WebView2RuntimeSource) -and (-not (Test-Path $WebView2Cab) -or (Get-Item $WebView2Cab).Length -lt 100MB)) {
        Write-Host "Downloading WebView2 Runtime $WebView2Version (x64)..."
        Invoke-WebRequest -Uri $WebView2Url -OutFile $WebView2Cab
    }
    if (-not (Test-Path $WebView2RuntimeSource)) {
        Write-Host "Extracting WebView2 Runtime..."
        New-Item -ItemType Directory -Force -Path $WebView2ExtractedRoot | Out-Null
        & expand.exe -F:* $WebView2Cab $WebView2ExtractedRoot | Out-Null
    }
    if (-not (Test-Path (Join-Path $WebView2RuntimeSource "msedgewebview2.exe"))) {
        throw "WebView2 Runtime extraction did not produce msedgewebview2.exe at $WebView2RuntimeSource"
    }
    Copy-Item $WebView2RuntimeSource (Join-Path $ReleaseFolder "WebView2Runtime") -Recurse

    # Record what was actually bundled, so a shipped build can be identified after the fact.
    $buildInfoScript = @'
import json, platform, sys, datetime
import torch, diffusers, transformers
json.dump({
    "version": sys.argv[1],
    "git_sha": sys.argv[2],
    "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "diffusers": diffusers.__version__,
    "transformers": transformers.__version__,
}, open(sys.argv[3], "w", encoding="utf-8"), indent=2)
'@
    $gitSha = (& git rev-parse HEAD 2>$null)
    if ($LASTEXITCODE -ne 0 -or -not $gitSha) { $gitSha = "unknown" }
    $buildInfoStatus = Invoke-PythonSnippet $buildInfoScript @($Version, $gitSha, (Join-Path $ReleaseFolder "build-info.json"))
    if ($buildInfoStatus -ne 0) { throw "Writing build-info.json failed (exit code $LASTEXITCODE)." }
    Get-Content (Join-Path $ReleaseFolder "build-info.json") | Write-Host

    if ($Sign) {
        $exe = Join-Path $ReleaseFolder "PixelFlow.exe"
        & $SignTool sign /sha1 $CertificateThumbprint /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 $exe
        if ($LASTEXITCODE -ne 0) { throw "signtool failed (exit code $LASTEXITCODE) -- the release is not signed." }
    }

    $hashPath = "$Archive.sha256"
    $sevenZip = Get-Command 7z -ErrorAction SilentlyContinue
    if (-not $sevenZip -and $env:ProgramFiles) {
        # The 7-Zip installer doesn't add itself to PATH -- use its default install location too.
        $sevenZip = Get-Command (Join-Path $env:ProgramFiles "7-Zip\7z.exe") -ErrorAction SilentlyContinue
    }
    if ($SkipArchive) {
        # An archive left over from an earlier build no longer matches release\ -- remove it rather
        # than leave a stale copy that looks like this build's.
        foreach ($stale in @(Get-ChildItem -Path "$Archive*" -ErrorAction SilentlyContinue)) {
            Remove-Item $stale.FullName -Force; Write-Host "Removed stale $($stale.FullName)"
        }
        Write-Host "Skipping 7z archive (-SkipArchive)."
    } elseif ($sevenZip) {
        foreach ($stale in @(Get-ChildItem -Path "$Archive*" -ErrorAction SilentlyContinue)) { Remove-Item $stale.FullName -Force }
        & $sevenZip.Source a -t7z -mx=9 -m0=lzma2 -ms=on $Archive (Join-Path $ReleaseFolder "*")
        if ($LASTEXITCODE -ne 0) { throw "7z failed (exit code $LASTEXITCODE)." }
        if ((Get-Item $Archive).Length -gt $MaxAssetBytes) {
            # Too big for one GitHub release asset (2 GiB limit): split volumes instead.
            Remove-Item $Archive -Force
            $volumeSize = "{0}m" -f [math]::Floor($MaxAssetBytes / 1MB)
            & $sevenZip.Source a -t7z -mx=9 -m0=lzma2 -ms=on "-v$volumeSize" $Archive (Join-Path $ReleaseFolder "*")
            if ($LASTEXITCODE -ne 0) { throw "7z (split volumes) failed (exit code $LASTEXITCODE)." }
            Write-Host "Archive is larger than $MaxAssetBytes bytes: split into $volumeSize volumes."
        }
        # One "<hash>  <file>" line per archive file (a single .7z, or every .7z.NNN volume).
        Get-ChildItem -Path "$Archive*" | Where-Object { $_.Name -notlike "*.sha256" } | Sort-Object Name | ForEach-Object {
            "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash, $_.Name
        } | Set-Content $hashPath
        Write-Host "Release archive: $Archive (see $hashPath for every file)"
        Write-Host "SHA-256 file: $hashPath"
    } else {
        Write-Warning "7z was not found. The unpacked build is ready at $ReleaseFolder."
    }
    Write-Host "Release folder: $ReleaseFolder"

    # Only reached when everything above succeeded -- a failed build keeps its intermediates for
    # debugging.
    $leftovers = @($LegacyPackage)
    if (-not $KeepBuildFiles) { $leftovers += @($Build, $Dist) }
    foreach ($leftover in $leftovers) {
        if (Test-Path $leftover) { Remove-Tree $leftover; Write-Host "Cleaned up $leftover" }
    }
} finally {
    Pop-Location
}
