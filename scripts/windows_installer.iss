#ifndef AppVersion
  #define AppVersion "0.4.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\BlurAll"
#endif
#ifndef OutputDir
  #define OutputDir "..\dist\installer"
#endif

[Setup]
AppId={{3CB4671C-6AF6-4CBE-825B-107D1FAFB21D}
AppName=Blur All
AppVersion={#AppVersion}
AppPublisher=Blur All
DefaultDirName={localappdata}\Programs\BlurAll
DefaultGroupName=Blur All
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\BlurAll.exe
OutputDir={#OutputDir}
OutputBaseFilename=BlurAll-Setup-{#AppVersion}
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Blur All"; Filename: "{app}\BlurAll.exe"
Name: "{autodesktop}\Blur All"; Filename: "{app}\BlurAll.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\BlurAll.exe"; Description: "Launch Blur All"; Flags: nowait postinstall skipifsilent
