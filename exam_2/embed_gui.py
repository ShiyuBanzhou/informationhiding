import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image, ImageTk
import numpy as np
import hashlib # 为哈希功能添加

class EmbedWatermarkApp:
    def __init__(self, master):
        self.master = master
        master.title("嵌入水印程序") # 窗口标题
        self.frame = tk.Frame(master)
        self.frame.pack(padx=10, pady=10)

        # --- 原始奇偶校验水印部分 ---
        tk.Label(self.frame, text="脆弱水印 (奇偶校验)", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=3, pady=(0,5))
        self.embed_parity_btn = tk.Button(
            self.frame, text="选择图像并嵌入 (奇偶校验)", width=35, command=self.embed_parity_watermark
        )
        self.embed_parity_btn.grid(row=1, column=0, columnspan=3, pady=5)

        # --- 新的基于哈希的分块水印部分 ---
        tk.Label(self.frame, text="脆弱水印 (分块Hash, 7-MSB)", font=("Arial", 12, "bold")).grid(row=2, column=0, columnspan=3, pady=(15,5))
        
        self.block_size_label = tk.Label(self.frame, text="块大小 (例如 8 或 16):")
        self.block_size_label.grid(row=3, column=0, sticky="e", padx=5)
        self.block_size_var = tk.StringVar(value="16") # 默认块大小
        self.block_size_entry = tk.Entry(self.frame, textvariable=self.block_size_var, width=5)
        self.block_size_entry.grid(row=3, column=1, sticky="w", padx=5)

        self.embed_hash_btn = tk.Button(
            self.frame, text="选择图像并嵌入 (分块Hash)", width=35, command=self.embed_hash_block_watermark
        )
        self.embed_hash_btn.grid(row=4, column=0, columnspan=3, pady=5)


        # 显示标签和画布 (用于输出共享)
        labels = ["原始图像 (Original Image)", "隐秘载体 (Stego Image)", "辅助信息图 (Auxiliary Map)"]
        for idx, txt in enumerate(labels):
            tk.Label(self.frame, text=txt).grid(row=5, column=idx, pady=5) # 图像标签

        self.canvas_orig = tk.Label(self.frame, borderwidth=1, relief="solid", width=32, height=16) # 初始大小
        self.canvas_orig.grid(row=6, column=0, padx=5, pady=5, sticky="nsew")
        self.canvas_stego = tk.Label(self.frame, borderwidth=1, relief="solid", width=32, height=16)
        self.canvas_stego.grid(row=6, column=1, padx=5, pady=5, sticky="nsew")
        self.canvas_aux = tk.Label(self.frame, borderwidth=1, relief="solid", width=32, height=16) # 之前是 canvas_parity
        self.canvas_aux.grid(row=6, column=2, padx=5, pady=5, sticky="nsew")

        # 配置网格列权重以便画布调整大小
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_columnconfigure(2, weight=1)
        # 配置网格行权重以便画布调整大小
        self.frame.grid_rowconfigure(6, weight=1)


        # 进度条和状态栏
        self.progress = ttk.Progressbar(
            self.frame, orient="horizontal", length=400, mode="determinate" # 水平进度条
        )
        self.progress.grid(row=7, column=0, columnspan=3, pady=10, sticky="ew")
        self.status_label = tk.Label(self.frame, text="请选择操作...", fg="blue") # 状态显示标签
        self.status_label.grid(row=8, column=0, columnspan=3, pady=5)

        # 初始化图像属性
        self.img_orig_pil = None # 原始PIL图像对象
        self.img_stego_pil = None # 隐秘载体PIL图像对象
        self.img_aux_pil = None # 用于奇偶校验图或其他可视化信息的通用名称

    def _get_block_size(self):
        try:
            block_size = int(self.block_size_var.get())
            if block_size <= 0 or block_size > 64: # 实用限制
                messagebox.showerror("错误", "块大小必须是正整数 (例如 8, 16, 32).")
                return None
            return block_size
        except ValueError:
            messagebox.showerror("错误", "块大小必须是有效的整数.")
            return None

    def _get_hash_bits_for_block(self, data_bytes, num_bits_to_embed):
        """计算 SHA-256 哈希值并返回指定数量的前导比特。"""
        h = hashlib.sha256(data_bytes).digest()  # 哈希的原始字节
        hash_all_bits_str = "".join(format(byte, '08b') for byte in h) # 完整的比特串
        
        if num_bits_to_embed > len(hash_all_bits_str):
            # 如果块大小能很好地容纳哈希长度，则理想情况下不应发生此情况
            # 或者这意味着我们需要一个产生更少比特的哈希函数
            # 目前，为保持一致性，我们将使用可用比特并在必要时用0填充
            # (尽管对于SHA256，如果num_bits_to_embed <= 256，则不会有问题)
            return hash_all_bits_str.ljust(num_bits_to_embed, '0')
        return hash_all_bits_str[:num_bits_to_embed]

    def embed_hash_block_watermark(self):
        block_size = self._get_block_size()
        if block_size is None:
            return

        path = filedialog.askopenfilename(
            title="选择灰度图像 (Select Grayscale Image)",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"), ("All files", "*.*")]
        )
        if not path:
            self.status_label.config(text="操作取消", fg="orange") # 用户取消操作
            return

        try:
            img = Image.open(path).convert("L") # 打开图像并转换为灰度图
            self.img_orig_pil = img.copy() # 保留原始图像的副本以供显示
            arr_orig = np.array(img, dtype=np.uint8) # 将图像转换为NumPy数组
            stego_arr = arr_orig.copy() # 将要修改的数组
            h_img, w_img = arr_orig.shape # 获取图像高度和宽度
            
            num_hash_bits_per_block = block_size * block_size # 每个块要嵌入的哈希比特数

            self.progress['value'] = 0 # 重置进度条
            self.progress['maximum'] = (h_img // block_size + 1) * (w_img // block_size + 1) # 大致的块数量
            self.status_label.config(text="开始嵌入分块Hash水印...", fg="black") # 更新状态
            self.master.update_idletasks() # 更新UI
            
            processed_blocks = 0
            for r_start in range(0, h_img, block_size):
                for c_start in range(0, w_img, block_size):
                    r_end = min(r_start + block_size, h_img)
                    c_end = min(c_start + block_size, w_img)
                    
                    current_block_orig_pixels = arr_orig[r_start:r_end, c_start:c_end] # 当前原始像素块
                    
                    if current_block_orig_pixels.size == 0: # 跳过空块
                        continue

                    # 1. 计算原始块7个最高有效位(MSB)的哈希值
                    block_7msb_data = (current_block_orig_pixels >> 1).astype(np.uint8) # 获取高7位
                    hash_input_bytes = block_7msb_data.tobytes() # 转换为字节串以便进行哈希计算
                    
                    # 根据当前（可能是部分的）块大小确定实际要嵌入的比特数
                    actual_lsbs_in_block = current_block_orig_pixels.size # 当前块中可用的LSB数量
                    bits_to_embed_for_this_block = min(num_hash_bits_per_block, actual_lsbs_in_block)

                    block_hash_bits_str = self._get_hash_bits_for_block(hash_input_bytes, bits_to_embed_for_this_block)

                    # 2. 将这些哈希比特嵌入到 stego_arr 块的最低有效位(LSB)中
                    block_for_stego = stego_arr[r_start:r_end, c_start:c_end] # 获取stego_arr中对应的块
                    flat_stego_pixels = block_for_stego.ravel() # 扁平化以便于LSB操作

                    for i in range(len(block_hash_bits_str)): # 迭代遍历哈希字符串中的比特数
                        if i < flat_stego_pixels.size: # 确保不会超出像素范围
                            pixel_val = flat_stego_pixels[i]
                            bit_to_embed = int(block_hash_bits_str[i])
                            flat_stego_pixels[i] = (pixel_val & 0xFE) | bit_to_embed # 清除LSB并设置新的比特
                    
                    # 更新主要的 stego_arr (ravel 可能创建了一个视图或副本，reshape后赋值回去)
                    stego_arr[r_start:r_end, c_start:c_end] = flat_stego_pixels.reshape(block_for_stego.shape)
                    
                    processed_blocks += 1
                    self.progress['value'] = processed_blocks # 更新进度条
                    self.master.update_idletasks() # 更新UI

            self.img_stego_pil = Image.fromarray(stego_arr) # 从NumPy数组创建隐秘载体图像
            # 对于基于哈希的方法，辅助图不像奇偶校验图那样标准，可以显示LSB或不显示
            # 让我们创建隐秘图像LSB平面的可视化作为辅助图
            lsb_map_arr = (stego_arr & 1) * 255 # LSB为1的显示为白色(255)，0为黑色(0)
            self.img_aux_pil = Image.fromarray(lsb_map_arr.astype(np.uint8))


            save_path = filedialog.asksaveasfilename(
                title="保存隐秘载体 (Save Stego Image)",
                defaultextension=".png", # 默认扩展名
                filetypes=[("PNG", "*.png"), ("BMP", "*.bmp"), ("TIFF", "*.tiff")] # 可选文件类型
            )
            if not save_path:
                self.status_label.config(text="嵌入已取消，文件未保存.", fg="orange")
                # 如果取消保存，则清除显示
                self._clear_canvas(self.canvas_stego)
                self._clear_canvas(self.canvas_aux)
                return
            
            self.img_stego_pil.save(save_path) # 保存隐秘载体图像
            self.progress['value'] = self.progress['maximum'] # 完成进度
            success_msg = f"分块Hash水印嵌入完成。\n隐秘载体已保存至: {save_path}"
            self.status_label.config(text=success_msg, fg="green") # 显示成功信息
            messagebox.showinfo("嵌入成功", success_msg) # 弹出成功信息框

            # 显示三幅图
            self._show_image_on_canvas(self.img_orig_pil, self.canvas_orig)
            self._show_image_on_canvas(self.img_stego_pil, self.canvas_stego)
            self._show_image_on_canvas(self.img_aux_pil, self.canvas_aux, interpolation=Image.NEAREST) # 辅助图使用最近邻插值

        except FileNotFoundError:
            messagebox.showerror("错误", "未找到文件或路径不正确.")
            self.status_label.config(text="错误：文件未找到", fg="red")
        except Exception as e:
            messagebox.showerror("错误", f"发生错误: {str(e)}")
            self.status_label.config(text=f"错误: {str(e)}", fg="red")
            self.progress['value'] = 0


    def embed_parity_watermark(self): # 从 embed_watermark 重命名而来，这是原有的奇偶校验方法
        path = filedialog.askopenfilename(
            title="选择灰度图像 (Select Grayscale Image)",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"), ("All files", "*.*")]
        )
        if not path:
            self.status_label.config(text="操作取消", fg="orange") # 用户取消操作
            return

        try:
            img = Image.open(path).convert("L") # 打开图像并转换为灰度图
            self.img_orig_pil = img.copy() # 保留原始图像的副本以供显示
            arr = np.array(img, dtype=np.uint8) # 将图像转换为NumPy数组
            h, w = arr.shape # 获取图像高度和宽度
            total_pixels = h * w # 计算总像素数

            self.progress['value'] = 0 # 重置进度条
            self.progress['maximum'] = total_pixels # 设置进度条最大值
            self.status_label.config(text="开始嵌入奇偶校验水印...", fg="black") # 更新状态
            self.master.update_idletasks() # 更新UI

            # --- 向量化奇偶校验计算和嵌入 ---
            # self.status_label.config(text="处理中：计算校验位并嵌入...", fg="black") # 这行可以省略，因为很快
            # self.progress['value'] = int(total_pixels * 0.1) 
            # self.master.update_idletasks()

            # 获取每个像素的高7位 (p_7msb = pixel_value // 2)
            arr_7msb = arr >> 1
            
            # 使用位异或运算计算7个MSB的奇偶校验位
            # calculated_parity 将为0（偶数个1）或1（奇数个1）
            calculated_parity = np.zeros_like(arr_7msb, dtype=np.uint8) # 初始化校验位数组
            for i in range(7):  # 检查7个MSB的每一位 (0 到 6)
                calculated_parity = calculated_parity ^ ((arr_7msb >> i) & 1) # 异或操作计算奇偶性
            
            # 将原始像素的LSB清零，然后将计算得到的奇偶校验位设置到LSB
            stego_arr = (arr & 0xFE) | calculated_parity # 嵌入校验位到LSB
            
            # 创建用于可视化的奇偶校验位图 (0代表偶校验, 255代表奇校验)
            parity_map_arr = calculated_parity * 255 # 校验位为1的显示为白色(255)，0为黑色(0)
            
            self.progress['value'] = int(total_pixels * 0.8) # 指示更多进度
            self.status_label.config(text="处理中：生成图像...", fg="black")
            self.master.update_idletasks()

            self.img_stego_pil = Image.fromarray(stego_arr.astype(np.uint8)) # 从NumPy数组创建隐秘载体图像
            self.img_aux_pil = Image.fromarray(parity_map_arr.astype(np.uint8)) # 存储为辅助图 (原为 self.img_parity_pil)
            
            num_ones_in_parity = np.sum(calculated_parity) # 统计校验位为1的像素数量

            # --- 保存隐秘载体 ---
            save_path = filedialog.asksaveasfilename(
                title="保存隐秘载体 (Save Stego Image)",
                defaultextension=".png", # 默认扩展名
                filetypes=[("PNG", "*.png"), ("BMP", "*.bmp"), ("TIFF", "*.tiff")] # 可选文件类型
            )
            if not save_path:
                self.status_label.config(text="嵌入已取消，文件未保存.", fg="orange")
                self.progress['value'] = 0
                # 如果取消保存，则清除显示
                self._clear_canvas(self.canvas_stego)
                self._clear_canvas(self.canvas_aux) # 原为 self.canvas_parity
                return
            
            self.img_stego_pil.save(save_path) # 保存隐秘载体图像

            self.progress['value'] = total_pixels # 完成进度
            success_msg = (f"完成：嵌入 {num_ones_in_parity} 个'1' (占总像素 {num_ones_in_parity/total_pixels*100:.2f}%)\n"
                           f"隐秘载体已保存至: {save_path}")
            self.status_label.config(text=success_msg, fg="green") # 显示成功信息
            messagebox.showinfo("嵌入成功", success_msg) # 弹出成功信息框

            # 显示三幅图
            self._show_image_on_canvas(self.img_orig_pil, self.canvas_orig)
            self._show_image_on_canvas(self.img_stego_pil, self.canvas_stego)
            self._show_image_on_canvas(self.img_aux_pil, self.canvas_aux, interpolation=Image.NEAREST) # 校验位图使用最近邻插值

        except FileNotFoundError:
            messagebox.showerror("错误", "未找到文件或路径不正确.")
            self.status_label.config(text="错误：文件未找到", fg="red")
        except Exception as e:
            messagebox.showerror("错误", f"发生错误: {str(e)}")
            self.status_label.config(text=f"错误: {str(e)}", fg="red")
            self.progress['value'] = 0


    def _show_image_on_canvas(self, pil_img, canvas_widget, max_size_w=256, max_size_h=256, interpolation=Image.LANCZOS):
        # 在Tkinter Label上显示PIL图像
        if pil_img is None:
            self._clear_canvas(canvas_widget) # 如果图像为空，清除画布
            return
        try:
            w, h = pil_img.size # 获取图像原始宽高
            if w == 0 or h == 0: return # 避免空图像

            # 如果可用，获取画布实际大小以便更好地适应
            canvas_widget.update_idletasks() # 确保尺寸是最新的
            target_w = min(max_size_w, canvas_widget.winfo_width() if canvas_widget.winfo_width() >1 else max_size_w)
            target_h = min(max_size_h, canvas_widget.winfo_height() if canvas_widget.winfo_height() >1 else max_size_h)


            scale = min(target_w / w, target_h / h) # 计算缩放比例
            nw, nh = int(w * scale), int(h * scale) # 计算新的宽度和高度
            
            if nw < 1: nw = 1 # 确保新宽度至少为1
            if nh < 1: nh = 1 # 确保新高度至少为1

            img_resized = pil_img.resize((nw, nh), interpolation) # 调整图像大小
            tk_img = ImageTk.PhotoImage(img_resized) # 转换为Tkinter兼容的图像格式
            
            canvas_widget.configure(image=tk_img, width=nw, height=nh) # 在Label上配置图像
            canvas_widget.image = tk_img # 保留对tk_img的引用，防止被垃圾回收
        except Exception as e:
            # print(f"显示图像错误: {e}") # 调试信息
            self._clear_canvas(canvas_widget) # 出错时清除画布

    def _clear_canvas(self, canvas_widget):
        # 清除Label上的图像
        canvas_widget.configure(image=None, width=256, height=256) # 重置为默认大小或原始占位符
        canvas_widget.image = None


if __name__ == "__main__":
    root = tk.Tk() # 创建主窗口
    app = EmbedWatermarkApp(root) # 创建应用实例
    root.mainloop() # 进入Tkinter事件循环