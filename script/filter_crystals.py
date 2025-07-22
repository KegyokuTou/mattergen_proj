import os
from ase.io import read, write

def filter_structures_by_element(input_file, output_file, element_symbol='Li'):
    """
    读取一个包含多个晶体结构的extxyz文件，并筛选出包含特定元素的结构。

    参数:
    input_file (str): 输入的extxyz文件名。
    output_file (str): 输出的extxyz文件名。
    element_symbol (str): 要筛选的元素符号，默认为'Li'。
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：找不到输入文件 '{input_file}'")
        return

    print(f"正在从 '{input_file}' 读取所有晶体结构...")
    
    # 使用 index=':' 来读取文件中的所有结构（帧）
    # 这会返回一个包含所有 ase.Atoms 对象的列表
    try:
        all_structures = read(input_file, index=':')
        print(f"成功读取 {len(all_structures)} 个结构。")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 创建一个空列表，用于存放包含目标元素的结构
    structures_with_element = []

    print(f"正在筛选包含元素 '{element_symbol}' 的结构...")
    
    # 遍历所有读取到的结构
    for i, atoms in enumerate(all_structures):
        # get_chemical_symbols() 返回该结构中所有原子的化学符号列表
        symbols_in_structure = atoms.get_chemical_symbols()
        
        # 检查目标元素是否在该列表中
        if element_symbol in symbols_in_structure:
            # 如果存在，则将该结构（atoms对象）添加到我们的列表中
            structures_with_element.append(atoms)
            # print(f"  - 结构 {i+1}: 找到 '{element_symbol}'，予以保留。") # 如果需要详细输出，可以取消此行注释
        # else:
            # print(f"  - 结构 {i+1}: 不含 '{element_symbol}'，予以剔除。") # 如果需要详细输出，可以取消此行注释


    # 检查是否找到了任何符合条件的结构
    if not structures_with_element:
        print(f"筛选完成，但未找到任何包含 '{element_symbol}' 的结构。")
    else:
        print(f"筛选完成！共找到 {len(structures_with_element)} 个包含 '{element_symbol}' 的结构。")
        
        # 将筛选后的结构列表写入到新的文件中
        print(f"正在将结果保存到 '{output_file}'...")
        try:
            write(output_file, structures_with_element)
            print("文件保存成功！")
        except Exception as e:
            print(f"写入文件时出错: {e}")

# --- 主程序执行部分 ---
if __name__ == "__main__":
    # 定义输入和输出文件名
    input_filename = '/Users/tangjiayu/Desktop/tmp/generated_crystals.extxyz'
    output_filename = '/Users/tangjiayu/Desktop/tmp/filtered_crystals_with_Li.extxyz'

    # 调用函数执行筛选
    filter_structures_by_element(input_filename, output_filename, element_symbol='Li')
