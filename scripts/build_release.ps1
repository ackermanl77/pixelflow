[CmdletBinding()]
param(
    [string]$Version = "0.1.0",
    [switch]$Sign,
    [string]$CertificateSubject = "",
    # Build with a specific interpreter instead of the default lookup below -- e.g. the project's
    # own .venv, which has the CUDA build of torch. PyInstaller bundles whatever torch the
    # interpreter running it has, so building from a CPU-only torch ships a CPU-only app.
    [string]$Python = "",
    # Skip the 7z archive. A CUDA build bundles several GB of libraries, and -mx=9 LZMA over that
    # takes a long time -- skip it when only the unpacked release\ folder is needed.
    [switch]$SkipArchive,
    # Keep PyInstaller's build-release\ and dist-release\ folders after a successful build instead of
    # deleting them. build-release\ is PyInstaller's analysis cache, so keeping it makes the next
    # rebuild faster at the cost of a few hundred MB; dist-release\ is empty either way, since its
    # output is moved into release\.
    [switch]$KeepBuildFiles
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = "C:\Users\MELEJ\AppData\Local\Python\pythoncore-3.14-64\python.exe" }
if (-not (Test-Path $Python)) { $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path $Python)) {
    # Falls back to whatever "python" resolves to on PATH (e.g. a CI runner set up via
    # actions/setup-python) when neither dev-machine-specific path above exists.
    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath) { $Python = $onPath.Source }
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
# "WebView2Runtime" folder next to PixelFlow.exe, and desktop.py points pywebview at it via
# webview.settings['WEBVIEW2_RUNTIME_PATH'] -- see the sibling photo_professional_tool project's
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

if (-not (Test-Path $Python)) {
    throw "Python interpreter not found. Install Python globally or create .venv."
}

Push-Location $ProjectRoot
try {
    # $ErrorActionPreference doesn't cover native commands -- check their exit codes explicitly, so a
    # failed install or build stops here instead of packaging (and cleaning up around) a stale output.
    & $Python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed (exit code $LASTEXITCODE)." }
    & $Python -m pip install -r requirements-desktop.txt
    if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements-desktop.txt failed (exit code $LASTEXITCODE)." }
    & $Python -m PyInstaller --noconfirm --distpath $Dist --workpath $Build PixelFlow.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed (exit code $LASTEXITCODE)." }

    # Move (not copy) PyInstaller's output into release\: on the same drive that's a rename, so the
    # multi-GB build is never duplicated on disk, not even briefly.
    if (Test-Path $ReleaseFolder) { Remove-Item $ReleaseFolder -Recurse -Force }
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

    if ($Sign) {
        if ([string]::IsNullOrWhiteSpace($CertificateSubject)) {
            throw "-CertificateSubject is required when -Sign is used."
        }
        $exe = Join-Path $ReleaseFolder "PixelFlow.exe"
        & signtool sign /sha1 $CertificateSubject /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /a $exe
    }

    $hashPath = "$Archive.sha256"
    $sevenZip = Get-Command 7z -ErrorAction SilentlyContinue
    if ($SkipArchive) {
        # An archive left over from an earlier build no longer matches release\ -- remove it rather
        # than leave a stale copy that looks like this build's.
        foreach ($stale in @($Archive, $hashPath)) {
            if (Test-Path $stale) { Remove-Item $stale -Force; Write-Host "Removed stale $stale" }
        }
        Write-Host "Skipping 7z archive (-SkipArchive)."
    } elseif ($sevenZip) {
        if (Test-Path $Archive) { Remove-Item $Archive -Force }
        & $sevenZip.Source a -t7z -mx=9 -m0=lzma2 -ms=on $Archive (Join-Path $ReleaseFolder "*")
        if ($LASTEXITCODE -ne 0) { throw "7z failed (exit code $LASTEXITCODE)." }
        (Get-FileHash $Archive -Algorithm SHA256).Hash | Set-Content $hashPath
        Write-Host "Release archive: $Archive"
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
        if (Test-Path $leftover) { Remove-Item $leftover -Recurse -Force; Write-Host "Cleaned up $leftover" }
    }
} finally {
    Pop-Location
}
