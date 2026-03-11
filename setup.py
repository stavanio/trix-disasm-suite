from setuptools import setup, find_packages

setup(
    name="trix-disasm-bench",
    version="1.0.0",
    description="TRiX: Neuro-Symbolic Safety for Foundation Model Agents in Robotic Disassembly",
    author="Stavan Dholakia, Shivani Shukla, Abhishek Singh, Aditya Gazta",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "matplotlib>=3.5",
        "scipy>=1.7",
    ],
)
