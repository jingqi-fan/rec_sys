from setuptools import setup, find_packages

setup(
    name="hybrid-recommender-pipeline",
    version="0.1.0",
    packages=find_packages(
        include=[
            "configs*",
            "data*",
            "embedding*",
            "eval*",
            "feature*",
            "generative*",
            "llm_train*",
            "model*",
            "proto*",
            "rerank*",
            "server*",
            "trainer*",
        ],
        exclude=["examples*", "experiments*", "tests*", "tutorials*"],
    ),
)
