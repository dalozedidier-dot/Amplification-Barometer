from setuptools import find_packages, setup

setup(
    name="amplification-barometer",
    version="0.2.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
    ],
    author="Votre Nom",
    description="Amplification Barometer: auditable composites, score stability audit, stress tests, and demo ODE",
    python_requires=">=3.10",
)
