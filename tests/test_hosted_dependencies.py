"""Guard frozen dependency parity while isolating CPU wheel discovery from PyPI."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


def pins(path):
    return [line for line in path.read_text().splitlines()
            if line and not line.startswith(('#','--'))]


class HostedDependencyTransport(unittest.TestCase):
    def test_hosted_dependencies_keep_all_frozen_versions(self):
        source=ROOT/'requirements/app-cpu.lock'
        hosted=ROOT/'requirements/app-community-cloud.lock'
        self.assertEqual(pins(hosted),pins(source))
        options=[line for line in hosted.read_text().splitlines() if line.startswith('--')]
        self.assertEqual(options,[
            '--index-url https://pypi.org/simple',
            '--find-links https://download.pytorch.org/whl/cpu/torch/',
            '--find-links https://download.pytorch.org/whl/cpu/torchvision/'])
        self.assertIn('torch==2.7.1+cpu',pins(hosted))
        self.assertIn('torchvision==0.22.1+cpu',pins(hosted))

    def test_root_and_wrapper_use_hosted_transport_and_forbid_source_builds(self):
        for path in (ROOT/'requirements.txt',ROOT/'deploy/community_cloud/requirements.txt'):
            lines=[x for x in path.read_text().splitlines() if x and not x.startswith('#')]
            self.assertIn('--only-binary=:all:',lines)
            include=next(x.removeprefix('-r ') for x in lines if x.startswith('-r '))
            self.assertEqual((path.parent/include).resolve(),ROOT/'requirements/app-community-cloud.lock')


if __name__=='__main__':unittest.main()
