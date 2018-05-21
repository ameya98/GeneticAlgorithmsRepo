import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='fmga',
    version='0.2.1',
    description='Genetic algorithms for 2-dimensional function maximization.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Ameya Daigavane',
    author_email='ameya.d.98@gmail.com',
    url='https://github.com/ameya98/GeneticAlgorithmsRepo/tree/master/fmga',
    packages=setuptools.find_packages(),
    keywords=['genetic', 'genetic_algorithms'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
