import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AutoML_tutorial-amoeba04",  # Replace with your own username
    version="0.1.0",
    author="amoeba04",
    author_email="amoeba04@gmail.com",
    description="AutoML tutorial code (implementing)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amoeba04/AutoML_tutorial",
    packages=setuptools.find_packages(),
    classifiers=["Operating System :: OS Independent",],
)
