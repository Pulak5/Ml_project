from setuptools import find_packages,setup
from typing import List
hyphen_e_dot="-e ."
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file:
        requirements=file.readlines()
        requirements=[r.replace("\n","") for r in requirements]

        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)

    return requirements

setup(
    name="Ml_project",
    version="0.0.1",
    author="Pulak kumar sarkar",
    author_email="pulakkumarsarkar15@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)