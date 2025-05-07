from PIL import Image
import numpy as np

# 1. 读入隐秘载体灰度图
img = Image.open("./exam_2/1.png").convert("L")
arr = np.array(img, dtype=np.uint8)

# 2. 读入你在 LSB 位面上画好的二值图，保证它的 dtype 也是 uint8，值为 0 或 1
drawn = Image.open("./exam_2/number.png").convert("1")  # PIL “1” 模式是二值
drawn = np.array(drawn, dtype=np.uint8)  # drawn 中的像素要么 0，要么 255
drawn = drawn // 255                     # 归一化到 0/1

# 3. 正确地清零 LSB 并写入新的位
arr = (arr & 0xFE) | drawn

# 4. 保存
Image.fromarray(arr).save("./exam_2/tampered.png")
