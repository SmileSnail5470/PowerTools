import argparse
import os
import shlex
import subprocess
import sys
import platform


APP_NAME = "PowerTools"
VERSION = "1.0.6"
COPYRIGHT = "SmileSnail5470"
MAIN_ENTRY = "main.py"
ICON_PATH_MAC = "app/ui/resources/images/logo.icns"
ICON_PATH_WIN = "app/ui/resources/images/logo.ico"


def find_project_root():
    return os.path.dirname(os.path.abspath(__file__))

def get_base_options(outdir):
    return [
        sys.executable, '-m', 'nuitka',
        '--enable-plugin=pyside6',
        '--include-qt-plugins=multimedia',
        '--assume-yes-for-downloads',
        '--output-dir={}'.format(outdir),
        '--output-filename={}'.format(APP_NAME),
        '--remove-output',
        '--nofollow-import-to=*.tests',
        '--noinclude-pytest-mode=nofollow',
        '--noinclude-unittest-mode=nofollow',
        '--include-package=cryptography',
        '--include-package=app.algorithms',
        '--include-data-dir=app/algorithms=app/algorithms',
        '--include-data-dir=app/library=app/license'
    ]

def build_macos_command(args, project_root, outdir):
    cmd = get_base_options(outdir)
    cmd.append('--mode=app-dist')
    
    cmd.extend([
        '--macos-app-mode=gui',
        f'--macos-app-name={APP_NAME}',
        f'--macos-app-version={VERSION}',
        f'--macos-signed-app-name=com.{COPYRIGHT}.{APP_NAME}',
    ])
    icon_path = os.path.join(project_root, ICON_PATH_MAC)
    if os.path.exists(icon_path):
        cmd.append(f'--macos-app-icon={icon_path}')
    else:
        print(f"Warning: Icon not found at {icon_path}")

    return cmd

def build_windows_command(args, project_root, outdir):
    cmd = get_base_options(outdir)
    cmd.append('--mode=standalone')
    
    cmd.extend([
        '--include-package=cupy',
        '--include-package-data=cupy',
        '--include-package=cupy_backends',
        '--windows-console-mode=disable',
        f'--company-name={COPYRIGHT}',
        f'--product-name={APP_NAME}',
        f'--file-version={VERSION}',
        f'--product-version={VERSION}',
    ])
    icon_path = os.path.join(project_root, ICON_PATH_WIN)
    if os.path.exists(icon_path):
        cmd.append(f'--windows-icon-from-ico={icon_path}')
    else:
        print(f"Warning: Icon not found at {icon_path}")

    return cmd

def main():
    parser = argparse.ArgumentParser(description='Build PowerTools with Nuitka')
    parser.add_argument('--output-dir', default='dist', help='Output directory')
    args = parser.parse_args()

    project_root = find_project_root()
    entrypoint = os.path.join(project_root, MAIN_ENTRY)
    outdir = os.path.abspath(args.output_dir)

    if not os.path.exists(entrypoint):
        raise SystemExit(f'Could not find entrypoint {MAIN_ENTRY} in {project_root}')

    if sys.platform == 'darwin':
        cmd = build_macos_command(args, project_root, outdir)
    elif sys.platform == 'win32':
        cmd = build_windows_command(args, project_root, outdir)
    else:
        raise SystemExit('Unsupported platform: {}'.format(sys.platform))

    cmd.append(entrypoint)

    print('-' * 60)
    print('Starting Nuitka Build...')
    print(f'Platform: {platform.system()} {platform.release()}')
    print('Command:')
    print(' '.join(shlex.quote(c) for c in cmd))
    print('-' * 60)

    try:
        os.makedirs(outdir, exist_ok=True)
        subprocess.check_call(cmd, cwd=project_root)
        print(f'\n[SUCCESS] Build finished. Output located in: {outdir}')
    except subprocess.CalledProcessError as e:
        print(f'\n[ERROR] Nuitka build failed with exit code {e.returncode}')
        sys.exit(e.returncode)

if __name__ == '__main__':
    main()