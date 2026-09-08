from setuptools import setup, find_packages

setup(
    name='rosa',
    version='0.0.1',
    description='ROSA: target expression-free spatiotemporal imputation for organogenesis via optimal transport',
    author='Zhipeng Zhou, Jinyun Niu, Yang Zhang, Zhiming Dai',
    author_email='zhouzhp@mail2.sysu.edu.cn',
    maintainer='Zhiming Dai (Corresponding Author)',
    maintainer_email='daizhim@mail.sysu.edu.cn',
    packages=find_packages(include=['rosa', 'rosa.*']),
    python_requires='>=3.10',
)