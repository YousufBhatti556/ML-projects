from setuptools import find_packages, setup
from typing import List
def get_requirements(filepath:str)->List[str]:
    # this function will return all the libraries as list in requirements.txt
    requirements = []
    with open(filepath) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements



setup(
    name="mlproject",
    version="0.0.1",
    author="Yousuf",
    author_email="yousufbhatti99@mail.com",
    packages=find_packages(),  # Looks for mlproject/
    install_requires=[
        "pandas",
        "numpy",
        "seaborn"
    ],
)