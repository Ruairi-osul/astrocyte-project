import setuptools


with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.readlines()


setuptools.setup(
    name="astro",
    version="0.1.0",
    author="Ruairi O'Sullivan",
    author_email="ruairi.osullivan.work@gmail.com",
    description="Analysis of data for astrocyte project",
    long_description="Analysis of data for astrocyte project",
    url="https://github.com/Ruairi-osul/astrocyte-project",
    packages=setuptools.find_packages(),
    package_data={"astro": ["py.typed"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
)
