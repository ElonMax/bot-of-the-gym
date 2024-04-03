from setuptools import setup, find_packages


setup(
    name="bots",
    version="0.0.1",
    keywords="language models, chatbot, instruct, fine-tuning",
    url="https://github.com/ElonMax/bot-of-the-gym",
    description="Instruments for tuning langauge models",
    packages=find_packages(
        where="src",
    ),
    package_dir={
        "": "src",
    },
    entry_points={
    }
)
