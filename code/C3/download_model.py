import requests
from pathlib import Path

def download_visualized_bge_model():
    """
    下载 Visual BGE 模型权重文件
    如果模型文件不存在，则从 Hugging Face 下载
    """
    # 定义模型路径和下载URL
    model_dir = Path("../../models/bge")
    model_file = model_dir / "Visualized_base_en_v1.5.pth"
    download_url = "https://hf-mirror.com/BAAI/bge-visualized/resolve/main/Visualized_base_en_v1.5.pth?download=true"
    
    # 检查模型文件是否已存在
    if model_file.exists():
        print(f"模型文件已存在: {model_file}")
        print(f"文件大小: {model_file.stat().st_size / (1024*1024):.1f} MB")
        return str(model_file)
    
    # 创建目录
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"创建模型目录: {model_dir}")
    
    # 下载模型
    print(f"开始下载模型...")
    print(f"下载地址: {download_url}")
    
    try:
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(model_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # 显示下载进度
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\r下载进度: {progress:.1f}% ({downloaded_size/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB)", end='')
        
        print(f"\n模型下载完成: {model_file}")
        print(f"文件大小: {model_file.stat().st_size / (1024*1024):.1f} MB")
        return str(model_file)
        
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        # 如果下载失败，删除不完整的文件
        if model_file.exists():
            model_file.unlink()
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        if model_file.exists():
            model_file.unlink()
        return None

if __name__ == "__main__":
    model_path = download_visualized_bge_model()
    if model_path:
        print(f"✅ 模型准备就绪: {model_path}")
    else:
        print("❌ 模型下载失败")
