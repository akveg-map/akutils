import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='akutils',
    version='1.1.1',
    author='Timm W. Nawrocki and Matthew J. Macander',
    author_email='twnawrocki@alaska.edu',
    description='Functions and utilities for Alaska Vegetation (AKVEG) Map and Database.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/accs-uaa/akutils',
    project_urls = {
        "Bug Tracker": "https://github.com/accs-uaa/akutils/issues"
    },
    license='MIT',
    packages=['akutils'],
    install_requires=['requests'],
)
