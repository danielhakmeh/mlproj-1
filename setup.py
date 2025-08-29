from setuptools import setup, find_packages
from typing import List

def get_reguirements(file_path:str) -> list[str]:  
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]  
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements 

setup(    name= "MLPROJ-1",
    version= "0.1.0",
    author= "Dany",     
    author_email= "danielhakmeh@gmail.com",
    description= "A sample Python project", 
    packages= find_packages(),  
    install_requires= get_reguirements("requirements.txt")
)