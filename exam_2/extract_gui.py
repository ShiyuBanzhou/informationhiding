import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image, ImageTk
import numpy as np
import hashlib # 为哈希功能添加

class ExtractWatermarkApp:
    def __init__(self, master):
        self.master = master
        master.title("提取并检测水印程序") # 窗口标题
        self.frame = tk.Frame(master)
        self.frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True) # 主框架

        # --- 原始奇偶校验检测部分 ---
        tk.Label(self.frame, text="脆弱水印 (奇偶校验)", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=3, pady=(0,5))
        self.extract_parity_btn = tk.Button(
            self.frame, text="选择隐秘载体并检测 (奇偶校验)", width=40, command=self.extract_and_detect_parity
        )
        self.extract_parity_btn.grid(row=1, column=0, columnspan=3, pady=5)

        # --- 新的基于哈希的分块检测部分 ---
        tk.Label(self.frame, text="脆弱水印 (分块Hash, 7-MSB)", font=("Arial", 12, "bold")).grid(row=2, column=0, columnspan=3, pady=(15,5))
        
        self.block_size_label = tk.Label(self.frame, text="块大小 (例如 8 或 16):")
        self.block_size_label.grid(row=3, column=0, sticky="e", padx=5)
        self.block_size_var = tk.StringVar(value="16") # 默认块大小
        self.block_size_entry = tk.Entry(self.frame, textvariable=self.block_size_var, width=5)
        self.block_size_entry.grid(row=3, column=1, sticky="w", padx=5)

        self.extract_hash_btn = tk.Button(
            self.frame, text="选择隐秘载体并检测 (分块Hash)", width=40, command=self.extract_and_detect_hash
        )
        self.extract_hash_btn.grid(row=4, column=0, columnspan=3, pady=5)


        # 显示标签与画布：隐秘载体、篡改覆盖、篡改地图
        labels = ["隐秘载体 (Stego Image)", "篡改覆盖 (Tamper Overlay)", "篡改区域图 (Tamper Map)"]
        for idx, txt in enumerate(labels):
            tk.Label(self.frame, text=txt).grid(row=5, column=idx, pady=5) # 图像标签

        self.canvas_stego = tk.Label(self.frame, borderwidth=1, relief="solid", width=32, height=16) # 隐秘载体图像画布
        self.canvas_stego.grid(row=6, column=0, padx=5, pady=5, sticky="nsew")
        self.canvas_overlay = tk.Label(self.frame, borderwidth=1, relief="solid", width=32, height=16) # 篡改覆盖图像画布
        self.canvas_overlay.grid(row=6, column=1, padx=5, pady=5, sticky="nsew")
        self.canvas_map = tk.Label(self.frame, borderwidth=1, relief="solid", width=32, height=16) # 篡改区域图画布
        self.canvas_map.grid(row=6, column=2, padx=5, pady=5, sticky="nsew")
        
        # 配置网格列权重以便画布调整大小
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_columnconfigure(2, weight=1)
        # 配置网格行权重以便画布调整大小
        self.frame.grid_rowconfigure(6, weight=1)

        # 透明度滑块
        tk.Label(self.frame, text="覆盖透明度 (Overlay Alpha):").grid(row=7, column=1, pady=(10,0))
        self.alpha_scale = ttk.Scale(
            self.frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, # 水平滑块，范围0.0到1.0
            length=200, command=self.update_alpha_display_proxy # 使用代理
        )
        self.alpha_scale.set(0.5) # 默认透明度为0.5
        self.alpha_scale.grid(row=8, column=1, pady=(0,10))
        
        # 状态栏
        self.status_label = tk.Label(self.frame, text="请选择操作...", fg="blue") # 状态显示标签
        self.status_label.grid(row=9, column=0, columnspan=3, pady=10)

        # 存储中间图像
        self.img_stego_pil = None       # 载入的隐秘载体图像 (PIL, 灰度)
        self.img_stego_rgb_pil = None # 载入的隐秘载体图像 (PIL, RGB格式，用于叠加层基底)
        self.img_overlay_pil = None   # 生成的带篡改标记的覆盖图像 (PIL, RGB)，这是完全标记红色的版本
        self.img_tamper_map_pil = None # 生成的篡改区域图 (PIL, 灰度)
        self.tamper_mask_pixel_level = None # 像素级篡改布尔掩码 (True表示篡改)

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

    def _get_hash_bits_for_block(self, data_bytes, num_bits_to_embed): # 与嵌入部分相同
        """计算 SHA-256 哈希值并返回指定数量的前导比特。"""
        h = hashlib.sha256(data_bytes).digest()
        hash_all_bits_str = "".join(format(byte, '08b') for byte in h)
        if num_bits_to_embed > len(hash_all_bits_str):
            return hash_all_bits_str.ljust(num_bits_to_embed, '0')
        return hash_all_bits_str[:num_bits_to_embed]

    def extract_and_detect_hash(self):
        block_size = self._get_block_size()
        if block_size is None: return

        path = filedialog.askopenfilename(
            title="选择隐秘载体 (Select Stego Image)",
            filetypes=[("PNG", "*.png"), ("BMP", "*.bmp"), ("TIFF", "*.tiff"), ("All files", "*.*")]
        )
        if not path:
            self.status_label.config(text="操作取消", fg="orange") # 用户取消操作
            return

        try:
            self.img_stego_pil = Image.open(path).convert("L") # 载入图像并转为灰度图
            stego_arr = np.array(self.img_stego_pil, dtype=np.uint8) # 将图像转换为NumPy数组
            h_img, w_img = stego_arr.shape # 获取图像高度和宽度
            
            self.status_label.config(text="开始检测分块Hash水印...", fg="black") # 更新状态
            self.master.update_idletasks() # 更新UI

            num_hash_bits_per_block = block_size * block_size # 每个块应有的哈希比特数
            # 块级篡改图 (布尔型)
            tamper_map_blocks = np.zeros(((h_img + block_size - 1) // block_size, 
                                          (w_img + block_size - 1) // block_size), dtype=bool)

            for r_idx, r_start in enumerate(range(0, h_img, block_size)):
                for c_idx, c_start in enumerate(range(0, w_img, block_size)):
                    r_end = min(r_start + block_size, h_img)
                    c_end = min(c_start + block_size, w_img)
                    current_stego_block = stego_arr[r_start:r_end, c_start:c_end] # 当前隐秘块

                    if current_stego_block.size == 0: continue # 跳过空块
                    
                    # 根据当前（可能是部分的）块大小确定实际要检查的比特数
                    actual_lsbs_in_block = current_stego_block.size
                    bits_to_check_for_this_block = min(num_hash_bits_per_block, actual_lsbs_in_block)

                    # 1. 从当前隐秘块的7个MSB重新计算哈希值
                    block_7msb_data = (current_stego_block >> 1).astype(np.uint8) # 获取高7位
                    hash_input_bytes = block_7msb_data.tobytes() # 转换为字节串
                    recalculated_hash_bits_str = self._get_hash_bits_for_block(hash_input_bytes, bits_to_check_for_this_block)

                    # 2. 从LSB中提取嵌入的哈希值
                    flat_stego_block_pixels = current_stego_block.ravel() # 扁平化块像素
                    extracted_bits_list = []
                    for i in range(bits_to_check_for_this_block): # 仅检查相关数量的比特
                         if i < flat_stego_block_pixels.size:
                            extracted_bits_list.append(str(flat_stego_block_pixels[i] & 1)) # 提取LSB
                    extracted_hash_bits_str = "".join(extracted_bits_list)
                    
                    # 确保提取的哈希与重新计算的哈希长度相同以便比较（如果较短则填充，尽管长度应匹配）
                    if len(extracted_hash_bits_str) < len(recalculated_hash_bits_str):
                         extracted_hash_bits_str = extracted_hash_bits_str.ljust(len(recalculated_hash_bits_str), '0')


                    # 3. 比较
                    if recalculated_hash_bits_str != extracted_hash_bits_str:
                        tamper_map_blocks[r_idx, c_idx] = True # 标记块为已篡改

            # 从块级映射创建像素级篡改掩码以进行可视化
            self.tamper_mask_pixel_level = np.kron(tamper_map_blocks, np.ones((block_size, block_size), dtype=bool))
            self.tamper_mask_pixel_level = self.tamper_mask_pixel_level[:h_img, :w_img] # 裁剪到原始图像大小

            # 创建可视化篡改图（篡改块为黑色，其他为白色）
            tamper_map_visual_arr = np.where(self.tamper_mask_pixel_level, 0, 255).astype(np.uint8)
            self.img_tamper_map_pil = Image.fromarray(tamper_map_visual_arr)

            # --- 生成篡改覆盖图像 ---
            # self.status_label.config(text="处理中：生成篡改可视化图像...", fg="black") # 可选的状态更新
            # self.master.update_idletasks()

            self.img_stego_rgb_pil = self.img_stego_pil.convert("RGB") # 将原始隐秘载体转换为RGB格式以便进行彩色叠加
            overlay_base_arr = np.array(self.img_stego_rgb_pil, dtype=np.uint8)
            
            overlay_display_arr = overlay_base_arr.copy() # 创建副本用于修改
            overlay_display_arr[self.tamper_mask_pixel_level] = [255, 0, 0] # 将篡改区域标记为红色
            
            # self.img_overlay_pil 将是完全标记红色的版本，用于混合
            self.img_overlay_pil = Image.fromarray(overlay_display_arr)
            
            tampered_blocks_count = np.sum(tamper_map_blocks) # 计算被篡改的块数量
            total_blocks = tamper_map_blocks.size
            tamper_percentage = (tampered_blocks_count / total_blocks) * 100 if total_blocks > 0 else 0

            if tampered_blocks_count > 0:
                result_msg = f"检测完成：发现 {tampered_blocks_count} 个块被篡改 ({tamper_percentage:.2f}%)."
                self.status_label.config(text=result_msg, fg="red") # 显示篡改信息
                messagebox.showwarning("检测结果", result_msg) # 弹出警告框
            else:
                result_msg = "检测完成：未检测到篡改."
                self.status_label.config(text=result_msg, fg="green") # 显示未篡改信息
                messagebox.showinfo("检测结果", result_msg) # 弹出信息框

            # 显示图像
            self._show_image_on_canvas(self.img_stego_pil, self.canvas_stego) # 显示原始载入的隐秘载体
            self.update_alpha_display(self.alpha_scale.get()) # 显示混合后的覆盖图像
            self._show_image_on_canvas(self.img_tamper_map_pil, self.canvas_map, interpolation=Image.NEAREST) # 篡改图使用最近邻插值

        except FileNotFoundError:
            messagebox.showerror("错误", "未找到文件或路径不正确.")
            self.status_label.config(text="错误：文件未找到", fg="red")
        except Exception as e:
            messagebox.showerror("错误", f"发生错误: {str(e)}")
            self.status_label.config(text=f"错误: {str(e)}", fg="red")
            # 出错时清除所有画布
            self._clear_all_canvases()


    def extract_and_detect_parity(self): # 从 extract_and_detect 重命名而来，这是原有的奇偶校验检测方法
        path = filedialog.askopenfilename(
            title="选择隐秘载体 (Select Stego Image)",
            filetypes=[("PNG", "*.png"), ("BMP", "*.bmp"), ("TIFF", "*.tiff"), ("All files", "*.*")]
        )
        if not path:
            self.status_label.config(text="操作取消", fg="orange") # 用户取消操作
            return

        try:
            # 载入图像并转为灰度图
            self.img_stego_pil = Image.open(path).convert("L")
            arr = np.array(self.img_stego_pil, dtype=np.uint8) # 将图像转换为NumPy数组
            h, w = arr.shape # 获取图像高度和宽度
            total_pixels = h * w # 计算总像素数

            self.status_label.config(text="开始检测奇偶校验水印...", fg="black") # 更新状态
            self.master.update_idletasks() # 更新UI

            # --- 向量化奇偶校验检查 ---
            # self.status_label.config(text="处理中：校验LSB与MSB奇偶性...", fg="black") # 可选状态
            # self.master.update_idletasks()

            # 提取每个像素中存储的LSB (假定为嵌入的奇偶校验位)
            stored_lsb = arr & 1
            
            # 获取每个像素的高7位 (arr_7msb = pixel_value // 2)
            arr_7msb = arr >> 1
            
            # 根据高7位重新计算期望的奇偶校验位
            expected_parity = np.zeros_like(arr_7msb, dtype=np.uint8) # 初始化期望校验位数组
            for i in range(7): # 遍历7个比特位
                expected_parity = expected_parity ^ ((arr_7msb >> i) & 1) # 异或操作计算奇偶性
            
            # 比较存储的LSB和重新计算的奇偶校验位
            # tamper_mask_pixel_level为布尔数组，True表示该像素被篡改
            self.tamper_mask_pixel_level = (stored_lsb != expected_parity) # 像素级
            
            # 创建篡改区域图：篡改处为黑色 (0), 未篡改处为白色 (255)
            tamper_map_arr = np.where(self.tamper_mask_pixel_level, 0, 255).astype(np.uint8)
            self.img_tamper_map_pil = Image.fromarray(tamper_map_arr) # 从NumPy数组创建篡改区域图

            # --- 生成篡改覆盖图像 ---
            # self.status_label.config(text="处理中：生成篡改可视化图像...", fg="black") # 可选状态
            # self.master.update_idletasks()

            self.img_stego_rgb_pil = self.img_stego_pil.convert("RGB") # 将原始隐秘载体转换为RGB格式以便进行彩色叠加
            overlay_base_arr = np.array(self.img_stego_rgb_pil).astype(np.uint8) 
            
            overlay_display_arr = overlay_base_arr.copy() # 创建副本用于修改
            overlay_display_arr[self.tamper_mask_pixel_level] = [255, 0, 0] # 红色
            
            self.img_overlay_pil = Image.fromarray(overlay_display_arr) # 完全标记的覆盖层
            
            tampered_count = np.sum(self.tamper_mask_pixel_level) # 计算被篡改的像素数量
            tamper_percentage = (tampered_count / total_pixels) * 100 if total_pixels > 0 else 0 # 计算篡改百分比

            if tampered_count > 0:
                result_msg = (f"检测完成：发现 {tampered_count} 个像素被篡改 "
                              f"({tamper_percentage:.2f}%).")
                self.status_label.config(text=result_msg, fg="red") # 显示篡改信息
                messagebox.showwarning("检测结果", result_msg) # 弹出警告框
            else:
                result_msg = "检测完成：未检测到篡改."
                self.status_label.config(text=result_msg, fg="green") # 显示未篡改信息
                messagebox.showinfo("检测结果", result_msg) # 弹出信息框

            # 显示图像
            self._show_image_on_canvas(self.img_stego_pil, self.canvas_stego) # 显示原始载入的隐秘载体
            self.update_alpha_display(self.alpha_scale.get()) # 显示混合后的覆盖图像
            self._show_image_on_canvas(self.img_tamper_map_pil, self.canvas_map, interpolation=Image.NEAREST) # 篡改图使用最近邻插值

        except FileNotFoundError:
            messagebox.showerror("错误", "未找到文件或路径不正确.")
            self.status_label.config(text="错误：文件未找到", fg="red")
        except Exception as e:
            messagebox.showerror("错误", f"发生错误: {str(e)}")
            self.status_label.config(text=f"错误: {str(e)}", fg="red")
            # 出错时清除所有画布
            self._clear_all_canvases()

    def update_alpha_display_proxy(self, alpha_value_str):
        # 此函数由滑块调用，然后调用主更新函数
        self.update_alpha_display(alpha_value_str)


    def update_alpha_display(self, alpha_value_str):
        # 根据滑块值更新覆盖图像的透明度并显示
        alpha = float(alpha_value_str) # 将滑块值转换为浮点数
        if self.img_stego_rgb_pil is None or self.img_overlay_pil is None or self.tamper_mask_pixel_level is None:
            # 如果基础图像未准备好，则不执行操作或显示占位符
            if self.img_stego_pil: # 如果至少原始隐秘载体已加载，则在覆盖画布上显示它
                 self._show_image_on_canvas(self.img_stego_pil.convert("RGB"), self.canvas_overlay)
            else:
                self._clear_canvas(self.canvas_overlay) # 清除覆盖画布
            return

        # 确保基础隐秘载体图像 (RGB) 和完全红色的覆盖层是NumPy数组
        base_arr = np.array(self.img_stego_rgb_pil, dtype=np.float32)
        # self.img_overlay_pil 是在篡改区域带有红色标记的图像 (即 marked_overlay_arr)
        marked_overlay_arr = np.array(self.img_overlay_pil, dtype=np.float32) # 这已经是带有红色标记的RGB图像

        # 混合: (1-alpha)*基础图像 + alpha*标记覆盖层
        # 仅将alpha应用于标记区域以获得更直观的“覆盖”效果
        blended_arr_float = base_arr.copy() # 从基础图像开始
        
        # 获取篡改区域中的原始像素
        original_tampered_pixels = base_arr[self.tamper_mask_pixel_level] 
        # 标记颜色（红色）
        red_color = np.array([255,0,0], dtype=np.float32) 

        # 仅混合篡改区域：(1-alpha)*原始像素颜色 + alpha*红色
        blended_tampered_pixels = (1 - alpha) * original_tampered_pixels + alpha * red_color
        blended_arr_float[self.tamper_mask_pixel_level] = blended_tampered_pixels # 将混合后的颜色应用回这些区域
        
        blended_arr_uint8 = np.clip(blended_arr_float, 0, 255).astype(np.uint8) # 确保像素值在有效范围内 [0, 255]
        
        blended_img_pil = Image.fromarray(blended_arr_uint8) # 从NumPy数组创建混合后的图像
        self._show_image_on_canvas(blended_img_pil, self.canvas_overlay) # 显示混合后的图像


    def _show_image_on_canvas(self, pil_img, canvas_widget, max_size_w=256, max_size_h=256, interpolation=Image.LANCZOS):
        # 在Tkinter Label上显示PIL图像
        if pil_img is None:
            self._clear_canvas(canvas_widget) # 如果图像为空，清除画布
            return
        try:
            w, h = pil_img.size # 获取图像原始宽高
            if w == 0 or h == 0: return # 避免空图像导致除零错误
            
            # 如果可用，获取画布实际大小以便更好地适应
            canvas_widget.update_idletasks() # 确保尺寸是最新的
            target_w = min(max_size_w, canvas_widget.winfo_width() if canvas_widget.winfo_width() > 1 else max_size_w)
            target_h = min(max_size_h, canvas_widget.winfo_height() if canvas_widget.winfo_height() > 1 else max_size_h)
            
            scale = min(target_w / w, target_h / h) if w > 0 and h > 0 else 1.0 # 计算缩放比例
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

    def _clear_all_canvases(self):
        # 清除所有画布上的图像并重置相关变量
        self._clear_canvas(self.canvas_stego)
        self._clear_canvas(self.canvas_overlay)
        self._clear_canvas(self.canvas_map)
        self.img_stego_pil = None
        self.img_stego_rgb_pil = None
        self.img_overlay_pil = None
        self.img_tamper_map_pil = None
        self.tamper_mask_pixel_level = None


if __name__ == "__main__":
    root = tk.Tk() # 创建主窗口
    app = ExtractWatermarkApp(root) # 创建应用实例
    root.mainloop() # 进入Tkinter事件循环