import os
import glob

def detect_encoding(file_path):
    """
    检测文件编码
    """
    import chardet
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def merge_txt_files(data_dir="data", output_file="data/train.txt"):
    """
    将data文件夹中的所有txt文件合并到train.txt中（排除train.txt和backup文件）
    """
    # 获取data文件夹中所有的txt文件
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    
    # 排除输出文件本身和其他不需要合并的文件
    exclude_patterns = ["train.txt", "train_backup.txt", "*backup*"]
    txt_files = [f for f in txt_files 
                 if not any(pattern in os.path.basename(f).lower() for pattern in exclude_patterns)]
    
    print(f"找到 {len(txt_files)} 个txt文件:")
    for file in txt_files:
        print(f"  - {file}")
    
    if not txt_files:
        print("没有找到其他txt文件需要合并")
        return
    
    # 读取所有txt文件内容
    all_content = []
    for file_path in txt_files:
        content = ""
        encoding = None
        
        # 尝试检测编码
        try:
            detected_encoding = detect_encoding(file_path)
            print(f"检测到文件编码: {file_path} -> {detected_encoding}")
            
            with open(file_path, 'r', encoding=detected_encoding) as f:
                content = f.read()
        except:
            # 如果检测失败，依次尝试常见编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            for enc in encodings:
                try:
                    with open(file_path, 'r', encoding=enc) as f:
                        content = f.read()
                        encoding = enc
                        print(f"使用编码 {enc} 成功读取: {file_path}")
                        break
                except:
                    continue
        
        if content.strip():  # 只添加非空内容
            # 确保内容是正确的编码
            try:
                # 如果内容不是UTF-8兼容的，尝试修复编码问题
                if isinstance(content, str):
                    # 尝试修复常见的编码问题
                    fixed_content = content.encode('utf-8', errors='ignore').decode('utf-8')
                    all_content.append(fixed_content)
                    print(f"已读取并修复编码: {file_path} ({len(fixed_content)} 字符)")
                else:
                    all_content.append(content)
                    print(f"已读取: {file_path} ({len(content)} 字符)")
            except:
                all_content.append(content)
                print(f"已读取(未修复编码): {file_path} ({len(content)} 字符)")
        else:
            print(f"警告: {file_path} 文件内容为空或无法读取")
    
    # 将所有内容写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, content in enumerate(all_content):
            f.write(content)
            if i < len(all_content) - 1:  # 不在最后一个内容后添加分隔符
                f.write("\n\n")  # 在不同文件内容之间添加空行分隔
    
    total_chars = sum(len(c) for c in all_content)
    print(f"\n合并完成!")
    print(f"总共合并了 {len(all_content)} 个文件")
    print(f"总字符数: {total_chars}")
    print(f"输出文件: {output_file}")

if __name__ == "__main__":
    merge_txt_files()