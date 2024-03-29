import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MIML",  # Replace with your own username
    version="0.0.1",
    author="Mridula Maddukuri",
    author_email="mridulamaddukuri@utexas.edu",
    description="MIML",
    packages=setuptools.find_packages(exclude=("test",)),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
