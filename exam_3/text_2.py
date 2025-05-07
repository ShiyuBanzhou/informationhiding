# -*- coding: utf-8 -*-

def text_to_binary(text, encoding='utf-8'):
    """
    将输入的文本字符串转换为二进制序列字符串。

    参数:
    text (str): 需要转换的文本。
    encoding (str): 文本编码格式，默认为 'utf-8'。

    返回:
    str: 表示文本的二进制序列字符串。
    """
    try:
        # 将文本编码为字节串
        byte_array = text.encode(encoding)
        # 将每个字节转换为8位二进制表示，并连接起来
        binary_sequence = ''.join(format(byte, '08b') for byte in byte_array)
        return binary_sequence
    except Exception as e:
        print(f"文本转二进制错误: {e}")
        return None

def binary_to_text(binary_sequence, encoding='utf-8'):
    """
    将二进制序列字符串转换回文本字符串。

    参数:
    binary_sequence (str): 需要转换的二进制序列字符串。
                           必须是8的倍数长度。
    encoding (str): 文本编码格式，默认为 'utf-8'。

    返回:
    str: 从二进制序列解码出的文本。
    """
    if len(binary_sequence) % 8 != 0:
        print("错误: 二进制序列的长度必须是8的倍数。")
        return None
    
    try:
        byte_array = bytearray()
        # 每8位分割一次，转换为一个字节
        for i in range(0, len(binary_sequence), 8):
            byte_chunk = binary_sequence[i:i+8]
            byte_array.append(int(byte_chunk, 2))
        
        # 将字节数组解码为文本
        text = byte_array.decode(encoding)
        return text
    except ValueError:
        print(f"错误: 二进制序列 '{binary_sequence}' 包含无效的二进制数字。")
        return None
    except Exception as e:
        print(f"二进制转文本错误: {e}")
        return None

# --- 主程序逻辑与用户交互 ---
if __name__ == "__main__":
    print("文本 <-> 二进制序列转换工具")

    while True:
        print("\n请选择操作:")
        print("1. 文本转换为二进制序列")
        print("2. 二进制序列转换为文本")
        print("3. 退出")
        
        choice = input("请输入选项 (1, 2, 或 3): ")

        if choice == '1':
            input_text = input("请输入要转换为二进制的文本: ")
            binary_result = text_to_binary(input_text)
            if binary_result:
                print(f"\n文本: '{input_text}'")
                print(f"对应的二进制序列 (UTF-8):\n{binary_result}")
        
        elif choice == '2':
            input_binary = input("请输入要转换为文本的二进制序列: ")
            # 移除非二进制字符，以防用户输入空格等
            cleaned_binary = "".join(filter(lambda x: x in '01', input_binary))
            if len(cleaned_binary) != len(input_binary):
                print(f"(注意: 输入中包含非二进制字符已被移除，处理序列为: {cleaned_binary})")

            text_result = binary_to_text(cleaned_binary)
            if text_result:
                print(f"\n二进制序列: '{cleaned_binary}'")
                print(f"解码后的文本 (UTF-8):\n'{text_result}'")
        
        elif choice == '3':
            print("感谢使用，程序退出。")
            break
        
        else:
            print("无效选项，请输入 1, 2, 或 3。")

    # 示例：转换 "22072120hgy"
    print("\n--- 示例 ---")
    example_text = "22072120hgy"
    print(f"原始文本: {example_text}")
    
    binary_representation = text_to_binary(example_text)
    if binary_representation:
        print(f"二进制表示: {binary_representation}")
        
        decoded_text = binary_to_text(binary_representation)
        if decoded_text:
            print(f"从二进制解码回文本: {decoded_text}")

    # 示例：转换包含中文字符的文本
    example_chinese_text = "你好，世界！"
    print(f"\n原始中文文本: {example_chinese_text}")

    binary_chinese_representation = text_to_binary(example_chinese_text)
    if binary_chinese_representation:
        print(f"中文文本的二进制表示 (UTF-8):\n{binary_chinese_representation}")

        decoded_chinese_text = binary_to_text(binary_chinese_representation)
        if decoded_chinese_text:
            print(f"从二进制解码回中文文本:\n{decoded_chinese_text}")

