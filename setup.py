from setuptools import find_packages, setup
from typing import List

e_dot = "-e ."

def get_requriments(filepath: str) -> List[str]:
    requirements = []

    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [ i.replace("\n", "") for i in requirements]

        if e_dot in requirements:
            requirements.remove(e_dot)

setup(name='Diabetes-Prediction',
      version='0.1',
      description='Machine Learning Diabetes Prediction project',
      author='Anishwar Behera',
      author_email='anishwarbehera@gmail.com',
      packages=find_packages(),
      install_requires = get_requriments("requirements.txt")
    )
