from setuptools import setup,find_packages
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirement(file_path:str)->List[str]:
    '''
    this function return a list of requirement
    '''
    requirements= []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [i.replace("\n","") for i in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements
setup(
    name='ml_project',
    version='0.0.1',
    author='Champ',
    author_email='ichamp_in@hotmail.com',
    packages=find_packages(),
    install_requires= get_requirement('requirements.txt')
)