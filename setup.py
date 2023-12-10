import distutils.core, subprocess

dist = distutils.core.run_setup("./detectron2/setup.py")
for x in dist.install_requires:
    subprocess.call(['python3', '-m', 'pip', 'install', x])