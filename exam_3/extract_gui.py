import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext # 导入 scrolledtext
from PIL import Image, ImageTk, ImageDraw
import random # 用于解置乱
from collections import Counter # 用于多数表决

# --- 常量与预计算 ---
# 预计算8x8 DCT变换矩阵
def dct_matrix(n=8):
    """计算并返回n*n的DCT变换矩阵"""
    f = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0:
                f[i, j] = math.sqrt(1 / n) * math.cos((j + 0.5) * math.pi * i / n)
            else:
                f[i, j] = math.sqrt(2 / n) * math.cos((j + 0.5) * math.pi * i / n)
    return f

F_DCT = dct_matrix()  # DCT变换矩阵
FT_DCT = F_DCT.T      # DCT逆变换矩阵 (F_DCT的转置)

# 用于嵌入的DCT系数位置 (行, 列)
COEFF_LOC_BIT = [(5, 2), (4, 3)]

# --- 核心提取逻辑 ---
def extract_single_bit_from_block(block_data):
    """从一个8x8 DCT块中提取单个比特"""
    dct_block = F_DCT @ block_data @ FT_DCT
    coeff1_val = dct_block[COEFF_LOC_BIT[0][0], COEFF_LOC_BIT[0][1]]
    coeff2_val = dct_block[COEFF_LOC_BIT[1][0], COEFF_LOC_BIT[1][1]]
    return '1' if coeff1_val >= coeff2_val else '0'

def decode_repetition_code(raw_bits, repetition_factor):
    """使用多数表决解码使用简单重复码编码的比特。"""
    if repetition_factor <= 1:
        return raw_bits # 无需解码

    decoded_bits = []
    num_raw_bits = len(raw_bits)
    
    for i in range(0, num_raw_bits, repetition_factor):
        chunk = raw_bits[i : i + repetition_factor]
        if not chunk: # 如果长度不正确，不应发生
            continue
        # 如果块短于重复因子（在序列末尾）
        # 可以决定忽略它或仍然执行多数表决。
        # 此示例：即使对于不完整的块，如果它们不为空，也进行多数表决。
        
        count = Counter(chunk)
        # 确定最常见的比特。在平局的情况下（例如，因子为2，一个'0'和一个'1'）
        # 可以做出标准决定（例如，'0'）。
        # 这里：如果'1'更常见，则为'1'，否则为'0'。
        if count['1'] > count['0']:
            decoded_bits.append('1')
        elif count['0'] > count['1']:
            decoded_bits.append('0')
        else: # 平局或块中只有一个比特（如果RF=1，但已在上面捕获）
              # 或奇数RF和不完整的块，其中没有明确的多数
            if chunk: # 如果块不为空，则取第一个比特作为备用
                decoded_bits.append(chunk[0]) 
            # 或者，可以在此处发出错误信号或忽略该块。

    return "".join(decoded_bits)


def extract_watermark(stego_image, num_original_bits_to_extract,
                      enable_repetition_code, repetition_factor,
                      enable_scrambling, scrambling_key,
                      progress_callback=None):
    """核心水印提取函数，支持解置乱和重复码。"""
    h, w = stego_image.shape
    
    # 1. 确定需要从图像中提取的*原始*比特数
    num_raw_bits_to_extract = num_original_bits_to_extract
    if enable_repetition_code and repetition_factor > 1:
        num_raw_bits_to_extract *= repetition_factor

    max_extractable_raw_bits = (h // 8) * (w // 8)
    if num_raw_bits_to_extract > max_extractable_raw_bits:
        print(f"警告: 请求的原始比特提取长度 ({num_raw_bits_to_extract}) > 图像容量 ({max_extractable_raw_bits})。已限制。")
        num_raw_bits_to_extract = max_extractable_raw_bits


    # 2. 生成块索引并解置乱它们（如果启用）
    num_total_blocks = (h // 8) * (w // 8)
    block_indices = list(range(num_total_blocks)) # 物理块索引的原始顺序

    if enable_scrambling:
        if scrambling_key is None or scrambling_key == "":
            # 在提取情况下，如果密钥与嵌入密钥不匹配，
            # 这理想情况下应导致错误或显示警告，因为没有正确的密钥无法解置乱。
            # 此示例：如果没有密钥，则不进行解置乱。
            print("警告: 未指定解置乱密钥，将按标准顺序读取块。")
        else:
            try:
                random.seed(str(scrambling_key))
                random.shuffle(block_indices) # 生成与嵌入时相同的置乱顺序
            except Exception as e:
                print(f"错误: 无效的解置乱密钥: {e}。将按标准顺序读取块。")
                block_indices = list(range(num_total_blocks))


    # 3. 提取原始比特
    raw_extracted_bits = []
    for i in range(num_raw_bits_to_extract):
        if i >= len(block_indices): # 如果num_raw_bits_to_extract正确，则不应发生
            break 
        
        actual_physical_block_idx = block_indices[i]
        
        r_idx = (actual_physical_block_idx // (w // 8)) * 8
        c_idx = (actual_physical_block_idx % (w // 8)) * 8
        
        block_data = stego_image[r_idx : r_idx+8, c_idx : c_idx+8].astype(float)
        extracted_bit = extract_single_bit_from_block(block_data)
        raw_extracted_bits.append(extracted_bit)
        
        if progress_callback:
            progress_callback(i + 1, num_raw_bits_to_extract)
            
    raw_extracted_bits_str = "".join(raw_extracted_bits)

    # 4. 解码原始比特（如果使用了重复码）
    if enable_repetition_code and repetition_factor > 1:
        final_extracted_bits = decode_repetition_code(raw_extracted_bits_str, repetition_factor)
    else:
        final_extracted_bits = raw_extracted_bits_str
        
    return final_extracted_bits, raw_extracted_bits_str # 也返回原始比特用于调试/分析

# --- GUI类 ---
class ExtractGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('DCT水印提取与分析') # 窗口标题
        self.stego_image = None      
        self.extracted_bits_final = '' # 最终解码后的比特
        self.extracted_bits_raw = ''   # 提取的原始比特（解码前）
        self.original_watermark_bits = None 
        self.max_extractable_bits = 0 # 指图像中的原始比特

        # --- UI控件 ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # 左侧面板：文件操作，参数
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        file_ops_frame = ttk.LabelFrame(left_panel, text="文件操作", padding="10")
        file_ops_frame.pack(fill=tk.X, pady=(0,10))
        ttk.Button(file_ops_frame, text='加载含水印图像', command=self.load_stego_image).pack(fill=tk.X, pady=2) # 按钮文本
        ttk.Button(file_ops_frame, text='加载原始水印 (.txt)', command=self.load_original_watermark).pack(fill=tk.X, pady=2) # 按钮文本

        params_frame = ttk.LabelFrame(left_panel, text="提取参数", padding="10") # 标签文本
        params_frame.pack(fill=tk.X, pady=(0,10))

        ttk.Label(params_frame, text="期望原始长度:").grid(row=0, column=0, sticky=tk.W, pady=2) # 标签文本
        self.num_bits_var = tk.IntVar(value=128) # *原始*水印的长度
        self.num_bits_entry = ttk.Entry(params_frame, textvariable=self.num_bits_var, width=10)
        self.num_bits_entry.grid(row=0, column=1, columnspan=2, sticky=tk.EW, pady=2)

        self.cb_rep_code_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(params_frame, text="解码重复码", variable=self.cb_rep_code_var).grid(row=1, column=0, sticky=tk.W, pady=(5,0)) # 复选框文本
        self.rep_factor_var = tk.IntVar(value=3)
        ttk.Entry(params_frame, textvariable=self.rep_factor_var, width=5).grid(row=1, column=1, sticky=tk.W, padx=5, pady=(5,0))
        ttk.Label(params_frame, text="重复因子").grid(row=1, column=2, sticky=tk.W, pady=(5,0)) # 标签文本
        
        self.cb_scramble_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(params_frame, text="启用块解置乱", variable=self.cb_scramble_var).grid(row=2, column=0, sticky=tk.W) # 复选框文本
        self.scramble_key_var = tk.StringVar(value="key123") 
        ttk.Entry(params_frame, textvariable=self.scramble_key_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, columnspan=2)
        # ttk.Label(params_frame, text="解置乱密钥").grid(row=2, column=2, sticky=tk.W) # 标签文本
        ttk.Button(left_panel, text='执行提取与分析', command=self.perform_extraction_and_analysis, style="Accent.TButton").pack(fill=tk.X, pady=10, ipady=5) # 按钮文本
        style = ttk.Style()
        style.configure("Accent.TButton", font=('Helvetica', 10, 'bold'))


        # 右侧面板：图像，结果，分析
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        img_label_frame = ttk.LabelFrame(right_panel, text="含水印图像") # 标签文本
        img_label_frame.pack(side=tk.TOP, expand=False, fill=tk.X, pady=(0,5)) # 图像高度减小
        self.canvas_stego = tk.Canvas(img_label_frame, width=256, height=200, bg="lightgrey") # 画布高度减小
        self.canvas_stego.pack(expand=True, fill=tk.BOTH)

        self.progress_bar = ttk.Progressbar(right_panel, length=300, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        results_notebook = ttk.Notebook(right_panel)
        results_notebook.pack(expand=True, fill=tk.BOTH, pady=(5,0))

        tab_extracted = ttk.Frame(results_notebook)
        results_notebook.add(tab_extracted, text='提取比特 (解码后)') # 标签页文本
        self.text_result_final = scrolledtext.ScrolledText(tab_extracted, width=40, height=10, wrap=tk.CHAR)
        self.text_result_final.pack(expand=True, fill=tk.BOTH, padx=2, pady=2)

        tab_raw_extracted = ttk.Frame(results_notebook) # 用于原始比特的新标签页
        results_notebook.add(tab_raw_extracted, text='提取比特 (原始)') # 标签页文本
        self.text_result_raw = scrolledtext.ScrolledText(tab_raw_extracted, width=40, height=10, wrap=tk.CHAR)
        self.text_result_raw.pack(expand=True, fill=tk.BOTH, padx=2, pady=2)

        tab_analysis = ttk.Frame(results_notebook)
        results_notebook.add(tab_analysis, text='错误分析') # 标签页文本
        self.text_analysis_result = scrolledtext.ScrolledText(tab_analysis, width=40, height=10, wrap=tk.WORD)
        self.text_analysis_result.pack(expand=True, fill=tk.BOTH, padx=2, pady=2)
        self.text_analysis_result.insert(tk.END, "请先加载原始水印并执行提取。\n") # 初始文本
        self.text_analysis_result.config(state=tk.DISABLED)
        
        self.status_label_var = tk.StringVar(value="请加载含水印图像和可选的原始水印。") # 状态栏文本
        ttk.Label(right_panel, textvariable=self.status_label_var).pack(fill=tk.X, pady=5, side=tk.BOTTOM)


    def _update_image_canvas(self, canvas, np_image, title="图像"): # title参数改为中文
        if np_image is None:
            canvas.delete("all")
            canvas.update_idletasks()
            canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
            if canvas_w <=1: canvas_w=200 # 调整后的默认大小
            if canvas_h <=1: canvas_h=200
            canvas.create_text(canvas_w//2, canvas_h//2, text=f"无{title}", anchor="center") # 显示中文
            return

        h_img, w_img = np_image.shape[:2]
        img_pil = Image.fromarray(np_image)
        
        canvas.update_idletasks()
        canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
        if canvas_w <= 1: canvas_w = 200 
        if canvas_h <= 1: canvas_h = 200

        scale = min(canvas_w / w_img, canvas_h / h_img, 1.0) 
        new_w, new_h = int(w_img * scale), int(h_img * scale)
        
        if new_w > 0 and new_h > 0:
            img_pil_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self._photo_ref_extract = ImageTk.PhotoImage(img_pil_resized) 
            canvas.delete("all") 
            canvas.create_image(canvas_w//2, canvas_h//2, anchor="center", image=self._photo_ref_extract)
        else:
            canvas.delete("all")
            canvas.create_text(canvas_w//2, canvas_h//2, text=f"{title}尺寸问题", anchor="center") # 显示中文

    def load_original_watermark(self):
        filepath = filedialog.askopenfilename(title="选择原始水印 (.txt)", filetypes=[('文本文件', '*.txt')]) # 对话框标题
        if not filepath: return
        try:
            with open(filepath, 'r') as f:
                bits_content = f.read().strip()
            if not all(c in '01' for c in bits_content):
                messagebox.showerror("错误", "原始水印文件无效 (仅允许 '0' 和 '1')。") # 消息框文本
                return
            self.original_watermark_bits = bits_content
            self.num_bits_var.set(len(self.original_watermark_bits)) # 设置期望长度
            self.status_label_var.set(f"原始水印已加载 ({len(self.original_watermark_bits)} 比特)。") # 状态栏文本
            messagebox.showinfo('成功', f'原始水印加载成功: {len(self.original_watermark_bits)} 比特。') # 消息框文本
        except Exception as e:
            messagebox.showerror('加载错误', f'加载原始水印文件失败: {e}') # 消息框文本

    def load_stego_image(self):
        filepath = filedialog.askopenfilename(title="选择含水印图像", filetypes=[('图像文件', '*.png;*.jpg;*.bmp')]) # 对话框标题
        if not filepath: return
        try:
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None: raise ValueError("无法读取图像文件。")
            h_orig, w_orig = img.shape
            h_new, w_new = (h_orig // 8) * 8, (w_orig // 8) * 8
            if h_new == 0 or w_new == 0:
                messagebox.showerror("错误", f"图像尺寸 ({w_orig}x{h_orig}) 过小。") # 消息框文本
                return
            if h_orig != h_new or w_orig != w_new:
                img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
                messagebox.showinfo("提示", f"图像已调整为 {w_new}x{h_new}。") # 消息框文本
            self.stego_image = img
            self.max_extractable_bits = (h_new // 8) * (w_new // 8) # 最大原始比特数
            self.title(f'DCT提取 (图像容量: {self.max_extractable_bits} 原始比特)') # 窗口标题
            self.status_label_var.set(f"含水印图像已加载 ({w_new}x{h_new})。最大原始比特: {self.max_extractable_bits}。") # 状态栏文本
            self.canvas_stego.bind("<Configure>", lambda e: self._update_image_canvas(self.canvas_stego, self.stego_image, "含水印图像")) # title参数
            self._update_image_canvas(self.canvas_stego, self.stego_image, "含水印图像") # title参数
            self.text_result_final.delete('1.0', tk.END)
            self.text_result_raw.delete('1.0', tk.END)
            self.text_analysis_result.config(state=tk.NORMAL)
            self.text_analysis_result.delete('1.0', tk.END)
            self.text_analysis_result.insert(tk.END, "提取后可进行分析。\n") # 初始文本
            self.text_analysis_result.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror('加载错误', f'加载含水印图像失败: {e}') # 消息框文本

    def _progress_update_callback(self, current_step, total_steps):
        self.progress_bar['maximum'] = total_steps
        self.progress_bar['value'] = current_step
        self.update_idletasks()

    def perform_extraction_and_analysis(self):
        if self.stego_image is None:
            messagebox.showwarning('警告', '请先加载含水印图像。') # 消息框文本
            return
        try:
            num_original_to_extract = self.num_bits_var.get()
            if num_original_to_extract <= 0:
                messagebox.showwarning('警告', '期望原始长度必须为正。') # 消息框文本
                return
        except tk.TclError:
            messagebox.showerror("错误", "期望原始长度必须是有效数字。") # 消息框文本
            return

        enable_rep = self.cb_rep_code_var.get()
        rep_factor = self.rep_factor_var.get() if enable_rep else 1
        if enable_rep and rep_factor <= 0: # 重复因子应>0, 如果是1等于没重复
            messagebox.showerror("错误", "启用时重复因子必须大于0。") # 消息框文本
            return
        
        enable_scramble = self.cb_scramble_var.get()
        scramble_key = self.scramble_key_var.get() if enable_scramble else None
        if enable_scramble and not scramble_key:
             messagebox.showwarning("警告", "块解置乱已启用，但未提供密钥。将按标准顺序读取块。") # 消息框文本


        self.status_label_var.set("正在提取水印...") # 状态栏文本
        self.progress_bar['value'] = 0
        self.text_result_final.delete('1.0', tk.END)
        self.text_result_raw.delete('1.0', tk.END)
        self.text_analysis_result.config(state=tk.NORMAL)
        self.text_analysis_result.delete('1.0', tk.END)

        try:
            self.extracted_bits_final, self.extracted_bits_raw = extract_watermark(
                self.stego_image, num_original_to_extract,
                enable_rep, rep_factor,
                enable_scramble, scramble_key,
                self._progress_update_callback
            )
            
            # 显示解码后的比特
            display_bits_final = ""
            for i in range(0, len(self.extracted_bits_final), 64):
                display_bits_final += self.extracted_bits_final[i:i+64] + "\n"
            self.text_result_final.insert('1.0', display_bits_final.strip())

            # 显示原始比特
            display_bits_raw = ""
            for i in range(0, len(self.extracted_bits_raw), 64):
                display_bits_raw += self.extracted_bits_raw[i:i+64] + "\n"
            self.text_result_raw.insert('1.0', display_bits_raw.strip())
            
            extraction_info = f'提取完成! {len(self.extracted_bits_final)} 解码比特 ({len(self.extracted_bits_raw)} 原始比特)。' # 消息框文本
            messagebox.showinfo('提取完成', extraction_info) # 消息框文本
            self.status_label_var.set(extraction_info) # 状态栏文本

            if self.original_watermark_bits:
                self.analyze_errors()
            else:
                self.text_analysis_result.insert(tk.END, "未加载原始水印，无法进行错误分析。\n") # 分析区文本
        
        except Exception as e:
            messagebox.showerror('提取或分析错误', f'错误: {e}') # 消息框文本
            self.status_label_var.set("提取/分析失败。") # 状态栏文本
            import traceback
            traceback.print_exc()
        finally:
            self.text_analysis_result.config(state=tk.DISABLED)

    def analyze_errors(self):
        if not self.original_watermark_bits or self.extracted_bits_final is None : # 与最终比特比较
            self.text_analysis_result.insert(tk.END, "错误分析条件未满足。\n") # 分析区文本
            return

        original = self.original_watermark_bits
        extracted_final = self.extracted_bits_final
        
        compare_len = min(len(original), len(extracted_final))
        if compare_len == 0:
            self.text_analysis_result.insert(tk.END, "无可比较的比特。\n") # 分析区文本
            return

        error_count = 0
        error_details = []

        for i in range(compare_len):
            if original[i] != extracted_final[i]:
                error_count += 1
                # 索引i指*解码后*的比特。
                # 当使用重复和置乱时，
                # 将其映射到特定的*物理块*更为复杂。
                # 对于错误分析，解码比特的索引是相关的。
                error_details.append(f"  - 比特索引 (解码后) {i}: 原始='{original[i]}', 提取='{extracted_final[i]}'")
        
        ber = (error_count / compare_len) * 100 if compare_len > 0 else 0

        analysis_report = f"--- 错误分析 (对比解码后比特) ---\n" # 分析报告标题
        analysis_report += f"比较比特数: {compare_len}\n"
        analysis_report += f"原始水印长度: {len(original)}\n"
        analysis_report += f"提取水印长度 (解码后): {len(extracted_final)}\n"
        analysis_report += f"错误比特数: {error_count}\n"
        analysis_report += f"误码率 (BER): {ber:.2f}%\n\n"

        if error_count > 0:
            analysis_report += "错误详情 (索引基于解码后序列):\n" # 错误详情标题
            analysis_report += "\n".join(error_details)
        else:
            analysis_report += "恭喜！在比较的比特范围内未发现错误。\n" # 无错误消息
        
        self.text_analysis_result.insert(tk.END, analysis_report)


    def save_extracted_bits(self): # 现在应保存解码后的比特
        if not self.extracted_bits_final:
            messagebox.showwarning('警告', '没有提取（解码后）的水印信息可供保存。') # 消息框文本
            return
        filepath = filedialog.asksaveasfilename(title="保存提取（解码后）的比特", defaultextension='.txt') # 对话框标题
        if filepath:
            try:
                with open(filepath, 'w') as f:
                    f.write(self.extracted_bits_final)
                messagebox.showinfo('成功', f'提取（解码后）的水印比特已保存至: {filepath}') # 消息框文本
            except Exception as e:
                messagebox.showerror('保存错误', f'保存文件失败: {e}') # 消息框文本

if __name__ == '__main__':
    app = ExtractGUI()
    app.mainloop()
