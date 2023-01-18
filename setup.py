import setuptools, platform

with open("README.md", "r", encoding='utf_8') as fh:
    long_description = fh.read()

install_requires=[
    'spacy', # used for tokenizing multiligual text
    'torchtext' # not included with Pytorch by default
]

setuptools.setup(
    name="axformer",
    version="0.1.0",
    author="Shital Shah",
    author_email="sytelus@gmail.com",
    description="Research playground for Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sytelus/axformer",
    packages=setuptools.find_packages(),
	license='MIT',
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    include_package_data=True,
    install_requires=install_requires
)
