from setuptools import setup, find_packages

setup(
    name="spectral-heads",
    version="0.1.0",
    packages=find_packages(where="."),
    install_requires=["torch>=2.0", "transformers>=4.30", "numpy>=1.20"],
    python_requires=">=3.9",
    author="Luciano Benjamin Nieto",
    description="Zero-training attention head pruning via spectral key-gram analysis",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    url="https://github.com/Rylow999/spectral-prune",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
