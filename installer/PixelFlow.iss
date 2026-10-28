#define MyAppName "PixelFlow"
; Passed by scripts\build_release.ps1 -Installer as /DMyAppVersion=<its -Version>, so the installer
; can't drift from the build it packages; the fallback only applies to a manual ISCC run.
#ifndef MyAppVersion
  #define MyAppVersion "0.1.0"
#endif
; One publisher name across the projects, the same CompanyName as the exe's version resource.
#define MyAppPublisher "MELEJ"
#define MyAppExeName "PixelFlow.exe"

[Setup]
AppId={{4F598967-7B3B-4957-9886-9FFFD6550B33}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\PixelFlow
DefaultGroupName=PixelFlow
DisableProgramGroupPage=yes
OutputDir=..\installer-output
OutputBaseFilename=PixelFlow-{#MyAppVersion}-Setup
; build_release.ps1 -Sign passes /DSign and the "codesign" tool (/Scodesign=...): Inno Setup then
; signs this installer and the uninstaller it writes.
#ifdef Sign
SignTool=codesign
SignedUninstaller=yes
#endif
; The CUDA build is several GB, and a GitHub release asset can't exceed 2 GiB: the setup is split
; into PixelFlow-<version>-Setup.exe plus PixelFlow-<version>-Setup-1.bin, -2.bin... slices, each
; below that limit. Download them all into one folder and run the .exe.
DiskSpanning=yes
DiskSliceSize=1900000000
Compression=lzma2
SolidCompression=yes
LZMANumBlockThreads=4
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
; Per user by default (%LOCALAPPDATA%\Programs, no admin rights); "for all users" from the dialog
; or /ALLUSERS. User data never lives in the install folder (backend/paths.py).
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline dialog
; A running PixelFlow holds its DLLs open: ask it to close (Restart Manager) before replacing files.
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; ..\release is build_release.ps1's finished folder: PixelFlow.exe, _internal (with the CUDA torch)
; and the bundled WebView2Runtime the window needs. dist-release\ is only PyInstaller's raw output.
Source: "..\release\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\PixelFlow"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\PixelFlow"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,PixelFlow}"; Flags: nowait postinstall skipifsilent

[Code]
// The deepest bundled file (inside torch/diffusers) is ~140 characters below the install folder, and
// Windows paths stop at 260: a deeper folder fails halfway through copying ("MoveFile failed; code
// 3") and rolls back. Refuse it up front instead -- in the wizard and in a silent install alike.
const
  MaxAppDirLength = 110;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  if Length(ExpandConstant('{app}')) > MaxAppDirLength then
    Result := 'The install folder path is too long (' + IntToStr(Length(ExpandConstant('{app}'))) +
      ' characters). PixelFlow''s files sit deep inside it, and Windows paths are limited to 260 ' +
      'characters. Go back and choose a folder of at most ' + IntToStr(MaxAppDirLength) +
      ' characters, such as the default.';
end;

// Outputs, uploads, workflows, history, logs and (unless moved elsewhere) downloaded models live in
// %LOCALAPPDATA%\PixelFlow (backend/paths.py). They are kept on uninstall unless the user agrees
// here. SuppressibleMsgBox: a /SUPPRESSMSGBOXES silent uninstall answers IDNO (keep the data).
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    if DirExists(ExpandConstant('{localappdata}\PixelFlow')) then
      if SuppressibleMsgBox('Also delete your PixelFlow images, workflows, history, logs and downloaded models (' +
          ExpandConstant('{localappdata}\PixelFlow') + ')? Models can be many GB to download again.',
          mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES then
        DelTree(ExpandConstant('{localappdata}\PixelFlow'), True, True, True);
end;
