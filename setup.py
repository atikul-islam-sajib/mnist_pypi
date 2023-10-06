from setuptools import setup, find_packages

setup(
    name = "MnistClassifier",
    version = "0.1.0",  # Update with the version number of your package
    description = "A demo Python project for me to understand how to create it!",
    author="Atikul Isla Sajib",
    author_email="atikulislamsajib137@gmail.com",
    url="https://github.com/atikul-islam-sajib/irisClass",  # Update with your project's GitHub repository URL
    packages=find_packages(),
    install_requires=[
        "numpy>=1.0",  # Add any required dependencies here
        "torch>=1.0",
        "scikit-learn>=0.0.1",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="Mnist classification machine-learning",
    project_urls={
        "Bug Tracker": "https://github.com/atikul-islam-sajib/mnist_pypi/issues",
        "Documentation": "https://github.com/atikul-islam-sajib/irisClass/blob/main/README.md",
        "Source Code": "https://github.com/atikul-islam-sajib/mnist_pypi.git",
    },
    
)














# from setuptools import setup, find_packages

# setup(
#     name    = "irisClass",
#     description = "a demo python project for me to understand how to create it !!!",
#     author = "Atikul Isla Sajib",
#     author_email = "atikulislamsajib137@gmail.com",
#     packages = find_packages(),
#     install_requires = [
#     ],
# )