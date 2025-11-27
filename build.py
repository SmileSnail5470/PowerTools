import argparse
import os
import shlex
import subprocess
import sys


VERSION = "0.0.1"


def find_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def build_command(args, project_root):
    entrypoint = os.path.join(project_root, 'main.py')
    if not os.path.exists(entrypoint):
        raise SystemExit('Could not find entrypoint main.py in project root')

    base_cmd = [sys.executable, '-m', 'nuitka']

    # Basic options
    if sys.platform == 'win32':
        base_cmd += [
            '--standalone',
            '--enable-plugins=pyside6'
        ]
    elif sys.platform == 'darwin':
        base_cmd += [
            '--mode=app-dist',
            '--enable-plugins=pyside6',
            '--include-qt-plugins=sensible',
            '--show-memory',
            '--show-progress',
            "--macos-create-app-bundle",
            "--assume-yes-for-download",
            "--macos-disable-console",
            f"--macos-app-version={VERSION}",
            "--macos-app-name=PowerTools",
            "--macos-app-icon=app/ui/resources/images/logo.icns",
            "--copyright=SmileSnail5470",
            '--output-dir={outdir}'.format(outdir=os.path.abspath(args.output_dir)),
        ]

    # Final target
    base_cmd.append(entrypoint)

    return base_cmd


def main():
    parser = argparse.ArgumentParser(description='Build PowerTools with Nuitka')
    parser.add_argument('--output-dir', default='dist', help='Output directory')

    args = parser.parse_args()

    project_root = find_project_root()

    cmd = build_command(args, project_root)

    print('Nuitka build command:')
    print(' '.join(shlex.quote(c) for c in cmd))

    outdir = os.path.abspath(args.output_dir)
    os.makedirs(outdir, exist_ok=True)

    try:
        subprocess.check_call(cmd, cwd=project_root)
        print('\nBuild finished. Output in:', outdir)
    except subprocess.CalledProcessError as e:
        print('\nNuitka build failed with exit code', e.returncode)
        raise


if __name__ == '__main__':
    main()
