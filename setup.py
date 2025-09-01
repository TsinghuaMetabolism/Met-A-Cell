from setuptools import setup, find_packages

# 读取 README 文件作为长描述
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 定义项目元数据
setup(
    name="metacell",
    version="0.1.0",
    author="Yizi Zhao",
    author_emial="zhaoyz21@mails.tsinghua.edu.cn",
    description="A scalable toolkit for analyzing single-cell metabolomics built jointly with anndata.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # 项目主页地址，GitHub
    url="https://github.com/yourusername/metacell",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/metacell/issues",
        "Documentation": "https://github.com/yourusername/metacell/blob/master/README.md",
    },
    classifiers=[
        "Programming Language :: Python :: 3.9.10",
        "License :: OSI Approved :: MIT License.txt",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    package_dir={"":"src"},
    packages=find_packages(where="src"),
    package_data={
        "metacell": ["resource/*.csv", "resource/*.json", "resource/*.xlsx"],
    },
    python_requires=">=3.9",
    # 依赖项列表
    install_requires=[
        "anndata>=0.10.8",
        "numpy>=1.26.4",
        "pandas>=2.1.0",
        "scipy>=1.11.4",
        "pyopenms>=2.7.0",
        "pybaselines>=1.1.0",
        "matplotlib>=3.7.3",
        "tqdm>=4.62.3",
        "scanpy>=1.10.2"
    ],
    # 开发时额外依赖，支持GPU依赖
    # extras_require={
    #     "dev": ["pytest", "flake8", "black", "sphinx"],
    #     "gpu": ["numba[cuda]"],
    # },
    include_package_data=True,
    # 提供命令行工具接口
    entry_points={
        "console_scrips":[
            "metacell=metacell.__main__:main",
        ],
    },
)