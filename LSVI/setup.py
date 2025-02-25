import sys

import setuptools

# READ README.md for long description on PyPi.
try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write(f"Failed to read README.md:\n  {e}\n")
    sys.stderr.flush()
    long_description = ""

setuptools.setup(
    name="lsvi",
    author="Yvann Le Fay",
    description="Variational inference package.",
    long_description=long_description,
    version="0.1",
    packages=setuptools.find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "pytest",
        "scipy",
        "pymc",
        "particles>=0.4",
        "numpy",
        "tqdm",
        "scikit-learn",
        "levy-stable-jax",
        "blackjax"
    ],
    long_description_content_type="text/markdown",
    keywords="statistics variational inference divergence optimization exponential distributions gaussians",
    license="MIT",
    license_files=("LICENSE",),
)
