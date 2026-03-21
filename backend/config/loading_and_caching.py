import os
import pickle

def save_to_disk(data, folder_path: str, var_name: str):
    """
    Lưu biến Python xuống ổ cứng dưới dạng .pkl
    
    Args:
        data: dữ liệu cần lưu
        folder_path: thư mục lưu
        var_name: tên biến (dùng làm tên file)
    """
    
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, f"{var_name}.pkl")

    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    print(f"💾 Đã lưu {var_name} tại: {file_path}")

def load_from_disk(folder_path: str, var_name: str):
    """
    Load dữ liệu từ file .pkl
    """
    
    file_path = os.path.join(folder_path, f"{var_name}.pkl")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    print(f"📂 Đã load {var_name} từ: {file_path}")
    return data


def load_or_create(**kwargs):

    """
    Lưu biến Python xuống ổ cứng dưới dạng .pkl hoặc lấy dữ liệu từ file

    Args:
        data: dữ liệu cần lưu
        folder_path: thư mục lưu
        var_name: tên biến (dùng làm tên file)
    
    """
    data = kwargs.get("data")
    folder_path = kwargs.get("folder_path")
    var_name = kwargs.get("var_name")

    file_path = os.path.join(folder_path, f"{var_name}.pkl")

    if os.path.exists(file_path):
        print(f"⚡ Load cache {var_name}")
        return load_from_disk(folder_path, var_name)

    print(f"Tạo mới {var_name}")

    save_to_disk(data, folder_path, var_name)
    return data
