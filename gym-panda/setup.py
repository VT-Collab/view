from setuptools import setup

setup(name="gym_panda",
      version="0.1",
      author="Collab",
      packages=["gym_panda", "gym_panda.envs"],
      install_requires = ["gym", "numpy", "pybullet"]
)
