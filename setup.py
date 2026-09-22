from setuptools import find_packages, setup

setup(
    name="trix-disasm-suite",
    version="0.0.0",
    description="TRiX revised paper reference implementation (unreleased)",
    packages=find_packages(include=["baselines*", "benchmark*", "envs*", "experiments*", "training*"]),
    python_requires=">=3.11",
    install_requires=["numpy==1.26.4"],
)
