# 第二节 Python虚拟环境部署

Python虚拟环境是项目开发中的重要工具，它可以为不同项目创建独立的Python环境，避免依赖冲突。本节将介绍三种主流的Python虚拟环境部署方式。

## 1. Conda 环境管理

Conda是一个开源的包管理系统和环境管理系统，特别适合数据科学和机器学习项目。

### 1.1 安装Miniconda

Linux和macOS系统输入以下命令安装：

```bash
# 下载Miniconda安装包
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# 运行安装脚本
bash Miniconda3-latest-Linux-x86_64.sh
```

Windows系统优先推荐访问[清华大学开源软件镜像站](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/)，以获得更快的下载速度，或者访问[Miniconda 官方网站](https://docs.conda.io/en/latest/miniconda.html)，根据你的系统选择最新的 `Windows-x86_64.exe` 版本下载。下载完成后，双击 `.exe` 文件启动安装。

## 2. Python -m venv 部署

venv是Python 3.3+内置的虚拟环境模块，轻量且简单。

### 2.1 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv myproject_env
```

### 2.2 激活和停用环境

**Linux/macOS:**
```bash
# 激活环境
source myproject_env/bin/activate

# 停用环境
deactivate
```

**Windows:**
```bash
# 激活环境
myproject_env\Scripts\activate

# 停用环境
deactivate
```

## 3. UV 环境管理

UV是一个用Rust编写的极快的Python包安装器和解析器，是pip和pip-tools的替代品。

### 3.1 安装UV

**方式一：使用pip安装**
```bash
pip install uv
```

**方式二：Linux和macOS用户使用curl安装（推荐）**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**方式三：Windows用户安装**
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 4. 三种方式最快速部署方法

### 4.1 Conda 最快速部署（推荐数据科学项目）

```bash
# 一键创建并激活环境，同时安装常用包
conda create -n myproject python=3.9 numpy pandas matplotlib jupyter -y
conda activate myproject

# 如果需要更多科学计算包
conda create -n myproject python=3.9 anaconda -y
conda activate myproject
```

### 4.2 Python venv 最快速部署（推荐通用项目）

```bash
# 一行命令创建、激活环境并升级pip
python -m venv venv && source venv/bin/activate && pip install --upgrade pip

# Windows用户使用：
python -m venv venv && venv\Scripts\activate && pip install --upgrade pip

# 快速安装项目依赖（如果有requirements.txt）
pip install -r requirements.txt
```


### 4.3 UV 最快速部署（推荐现代Python项目）

```bash
# 一键初始化项目并创建环境
uv init myproject && cd myproject

# 快速添加依赖并自动创建虚拟环境
uv add fastapi uvicorn requests

# 直接运行代码（自动管理环境）
uv run python main.py
```
