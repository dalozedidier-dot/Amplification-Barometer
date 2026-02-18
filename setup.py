from setuptools import find_packages, setup

setup(
    name="amplification-barometer",
    version="0.4.11",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "plotly>=5.20.0",
    ],
    author="Votre Nom",
    description="Auditable amplification barometer: composites, stability audits, stress tests, anti-gaming suite, and L(t) performance checks",
    python_requires=">=3.10",
)
