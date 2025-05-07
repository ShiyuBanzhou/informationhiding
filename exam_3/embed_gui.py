import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import random # 用于置乱

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

# 用于嵌入的DCT系数位置 (行, 列) - 每个块嵌入1比特
COEFF_LOC_BIT = [(5, 2), (4, 3)]

DEFAULT_EMBEDDING_STRENGTH = 20.0

# --- 核心嵌入逻辑 ---
def apply_dct_and_embed_single_bit(block_data, bit_to_embed, strength):
    """对单个8x8块应用DCT并根据给定单比特嵌入信息"""
    dct_block = F_DCT @ block_data @ FT_DCT
    
    coeff1_val = dct_block[COEFF_LOC_BIT[0][0], COEFF_LOC_BIT[0][1]]
    coeff2_val = dct_block[COEFF_LOC_BIT[1][0], COEFF_LOC_BIT[1][1]]

    if bit_to_embed == '1':
        if coeff1_val <= coeff2_val: # 如果不满足 c1 > c2，则调整
            avg = (coeff1_val + coeff2_val) / 2.0
            dct_block[COEFF_LOC_BIT[0][0], COEFF_LOC_BIT[0][1]] = avg + strength / 2.0
            dct_block[COEFF_LOC_BIT[1][0], COEFF_LOC_BIT[1][1]] = avg - strength / 2.0
    elif bit_to_embed == '0':
        if coeff1_val >= coeff2_val: # 如果不满足 c1 < c2，则调整
            avg = (coeff1_val + coeff2_val) / 2.0
            dct_block[COEFF_LOC_BIT[0][0], COEFF_LOC_BIT[0][1]] = avg - strength / 2.0
            dct_block[COEFF_LOC_BIT[1][0], COEFF_LOC_BIT[1][1]] = avg + strength / 2.0
    # 如果已满足条件或bit_to_embed为None(理论上不应发生在此函数)，则不修改
                
    return FT_DCT @ dct_block @ F_DCT


def embed_watermark(image, bits, strength,
                    enable_repetition_code, repetition_factor,
                    enable_scrambling, scrambling_key,
                    progress_callback=None):
    """核心水印嵌入函数，包含重复嵌入和块置乱选项"""
    h, w = image.shape
    stego_image = image.copy().astype(float)
    
    # 1. 水印预处理：纠错编码 (简单重复码)
    processed_bits = list(bits) 
    if enable_repetition_code and repetition_factor > 1:
        temp_bits = []
        for bit in bits:
            temp_bits.extend([bit] * repetition_factor)
        processed_bits = temp_bits
    
    num_processed_bits = len(processed_bits)

    # 2. 图像容量检查 (每个物理块嵌入1比特)
    max_embeddable_payload_bits = (h // 8) * (w // 8) 
    
    if num_processed_bits > max_embeddable_payload_bits:
        messagebox.showwarning("容量警告", 
                               f"处理后待嵌入水印长度 ({num_processed_bits} bits) 超出图像最大容量 ({max_embeddable_payload_bits} bits)。\n"
                               f"超出部分将被截断。")
        processed_bits = processed_bits[:max_embeddable_payload_bits]
        num_processed_bits = len(processed_bits) # 更新处理后的比特数

    # 3. 生成块索引并置乱 (如果启用)
    num_total_blocks = (h // 8) * (w // 8)
    block_indices = list(range(num_total_blocks)) # 物理块的原始顺序索引
    
    if enable_scrambling:
        if scrambling_key is None or scrambling_key == "":
            messagebox.showwarning("置乱警告", "未提供置乱密钥，将不进行置乱。")
        else:
            try:
                # 使用密钥作为随机种子以保证可重复的置乱顺序
                random.seed(str(scrambling_key)) # 确保密钥是可用于种子的类型
                random.shuffle(block_indices)
            except Exception as e: 
                messagebox.showerror("错误", f"置乱密钥无效: {e}。将不进行置乱。")
                block_indices = list(range(num_total_blocks)) # 恢复原始顺序

    # 4. 嵌入过程
    # 遍历 processed_bits，每个比特嵌入到一个物理块中
    # 物理块的选择顺序由 block_indices 决定
    
    for bit_idx in range(num_processed_bits):
        if bit_idx >= len(block_indices): # 确保不会超出可用物理块的数量
            messagebox.showwarning("嵌入中止", f"水印比特数 ({num_processed_bits}) 超过可用物理块数 ({len(block_indices)})。部分水印未嵌入。")
            break

        actual_physical_block_idx = block_indices[bit_idx] # 获取当前比特要嵌入的物理块的（可能置乱后的）索引
        
        # 将物理块索引转换为二维坐标
        r_idx = (actual_physical_block_idx // (w // 8)) * 8
        c_idx = (actual_physical_block_idx % (w // 8)) * 8
        
        current_block_data = stego_image[r_idx : r_idx+8, c_idx : c_idx+8]
        bit_to_embed = processed_bits[bit_idx]
        
        modified_block = apply_dct_and_embed_single_bit(current_block_data, bit_to_embed, strength)
        stego_image[r_idx : r_idx+8, c_idx : c_idx+8] = modified_block
        
        if progress_callback:
            progress_callback(bit_idx + 1, num_processed_bits)

    stego_image = np.clip(stego_image, 0, 255)
    return stego_image.astype(np.uint8)

# --- 热力图与可视化 ---
def heatmap(dmat, highlight_coords=None): # highlight_coords 现在是单个 [(r,c), (r,c)]
    m = np.abs(dmat)
    ptp_m = np.ptp(m)
    if ptp_m == 0: 
        norm = np.full_like(m, 128, dtype=np.uint8)
    else:
        norm = ((m - m.min()) / ptp_m * 255).astype(np.uint8)
    
    img = Image.fromarray(norm).resize((256, 256), Image.Resampling.NEAREST)
    
    if highlight_coords: 
        draw = ImageDraw.Draw(img)
        size = 256 // 8 
        for (r, c) in highlight_coords: # (row, col)
            draw.rectangle([c * size, r * size, (c + 1) * size - 1, (r + 1) * size - 1], outline='red', width=2)
    return img

def visualize_block_transform(original_block, bit_to_embed=None, strength=DEFAULT_EMBEDDING_STRENGTH):
    """可视化单个块的变换 (单比特)"""
    dct_before = F_DCT @ original_block @ FT_DCT
    
    text_before = "DCT变换前:\n"
    text_before += f"  系数 ({COEFF_LOC_BIT[0]},{COEFF_LOC_BIT[1]}): {dct_before[COEFF_LOC_BIT[0]] :.1f} vs {dct_before[COEFF_LOC_BIT[1]] :.1f}\n" 

    dct_after = dct_before.copy()
    idct_result_block_clipped = None
    text_after = ""

    if bit_to_embed is not None:
        temp_block_for_idct = original_block.copy()
        idct_result_block = apply_dct_and_embed_single_bit(temp_block_for_idct, bit_to_embed, strength)
        idct_result_block_clipped = np.clip(idct_result_block, 0, 255).astype(np.uint8)
        
        dct_after_embedding = F_DCT @ idct_result_block @ FT_DCT

        text_after = f"DCT变换后 (嵌入比特: {bit_to_embed}):\n" 
        text_after += f"  系数 ({COEFF_LOC_BIT[0]},{COEFF_LOC_BIT[1]}): {dct_after_embedding[COEFF_LOC_BIT[0]] :.1f} vs {dct_after_embedding[COEFF_LOC_BIT[1]] :.1f}\n" 
        dct_after = dct_after_embedding

    img_before_heatmap = heatmap(dct_before, COEFF_LOC_BIT)
    photo_before_heatmap = ImageTk.PhotoImage(img_before_heatmap)
    
    photo_after_heatmap = None
    if bit_to_embed is not None:
        img_after_heatmap = heatmap(dct_after, COEFF_LOC_BIT)
        photo_after_heatmap = ImageTk.PhotoImage(img_after_heatmap)

    photo_idct_block = None
    if idct_result_block_clipped is not None:
        img_idct = Image.fromarray(idct_result_block_clipped).resize((256,256), Image.Resampling.NEAREST)
        photo_idct_block = ImageTk.PhotoImage(img_idct)

    win = tk.Toplevel()
    win.title('单块8x8变换可视化') 
    win.attributes('-topmost', True)

    tk.Label(win, text=text_before, justify=tk.LEFT).pack(pady=(5,0))
    lbl_b_hm = tk.Label(win, image=photo_before_heatmap)
    lbl_b_hm.image = photo_before_heatmap
    lbl_b_hm.pack()

    if photo_after_heatmap:
        tk.Label(win, text=text_after, justify=tk.LEFT).pack(pady=(5,0))
        lbl_a_hm = tk.Label(win, image=photo_after_heatmap)
        lbl_a_hm.image = photo_after_heatmap
        lbl_a_hm.pack()

    if photo_idct_block:
        tk.Label(win, text='► IDCT重建图像块 (8x8)').pack(pady=(5,0)) 
        lbl_i = tk.Label(win, image=photo_idct_block)
        lbl_i.image = photo_idct_block
        lbl_i.pack()
        if idct_result_block is not None:
            diff = original_block - idct_result_block
            max_abs_diff = np.max(np.abs(diff))
            mse = np.mean(diff**2)
            tk.Label(win, text=f'与原块最大绝对差: {max_abs_diff:.2f}, MSE: {mse:.2f}').pack(pady=(0,5)) 
    win.geometry('')


# --- GUI 类 ---
class EmbedGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('DCT水印嵌入') 
        self.cover_image = None
        self.watermark_bits = ''
        self.stego_image = None
        self.max_embeddable_bits_info = "(请先加载图像)" 

        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        file_ops_frame = ttk.LabelFrame(left_panel, text="文件操作", padding="10") 
        file_ops_frame.pack(fill=tk.X, pady=(0,10))
        ttk.Button(file_ops_frame, text='加载载体图像', command=self.load_cover_image).pack(fill=tk.X, pady=2) 
        ttk.Button(file_ops_frame, text='加载水印文件 (.txt)', command=self.load_watermark_bits).pack(fill=tk.X, pady=2) 
        
        wm_entry_frame = ttk.Frame(file_ops_frame)
        wm_entry_frame.pack(fill=tk.X, pady=2)
        ttk.Label(wm_entry_frame, text="或输入短水印:").pack(side=tk.LEFT) 
        self.bits_entry_var = tk.StringVar()
        ttk.Entry(wm_entry_frame, textvariable=self.bits_entry_var, width=15).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5,0))
        ttk.Button(wm_entry_frame, text="使用", command=self.use_entry_bits, width=5).pack(side=tk.LEFT, padx=(2,0)) 

        basic_params_frame = ttk.LabelFrame(left_panel, text="基本与高级参数", padding="10") 
        basic_params_frame.pack(fill=tk.X, pady=(0,10))
        
        ttk.Label(basic_params_frame, text="嵌入强度 (P):").grid(row=0, column=0, sticky=tk.W, pady=2) 
        self.strength_var = tk.DoubleVar(value=DEFAULT_EMBEDDING_STRENGTH)
        ttk.Entry(basic_params_frame, textvariable=self.strength_var, width=10).grid(row=0, column=1, sticky=tk.EW, pady=2, columnspan=2)
        
        self.max_cap_label_var = tk.StringVar(value=f"最大容量: {self.max_embeddable_bits_info}") 
        ttk.Label(basic_params_frame, textvariable=self.max_cap_label_var).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=2)

        self.cb_rep_code_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(basic_params_frame, text="启用重复码", variable=self.cb_rep_code_var).grid(row=2, column=0, sticky=tk.W, pady=(5,0)) 
        self.rep_factor_var = tk.IntVar(value=3)
        ttk.Entry(basic_params_frame, textvariable=self.rep_factor_var, width=5).grid(row=2, column=1, sticky=tk.W, padx=5, pady=(5,0))
        ttk.Label(basic_params_frame, text="重复因子").grid(row=2, column=2, sticky=tk.W, pady=(5,0)) 
        
        self.cb_scramble_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(basic_params_frame, text="启用块置乱", variable=self.cb_scramble_var).grid(row=3, column=0, sticky=tk.W) 
        self.scramble_key_var = tk.StringVar(value="key123") 
        ttk.Entry(basic_params_frame, textvariable=self.scramble_key_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, columnspan=2)
        ttk.Label(basic_params_frame, text="置乱密钥").grid(row=3, column=2, sticky=tk.W, pady=(5,0)) 


        vis_frame = ttk.LabelFrame(left_panel, text="单块可视化", padding="10") 
        vis_frame.pack(fill=tk.X, pady=(0,10))
        ttk.Label(vis_frame, text='块索引:').grid(row=0, column=0, sticky=tk.W) 
        self.block_idx_var = tk.IntVar(value=0)
        ttk.Entry(vis_frame, textvariable=self.block_idx_var, width=7).grid(row=0, column=1)
        ttk.Button(vis_frame, text='可视化', command=self.visualize_selected_block).grid(row=0, column=2, padx=(5,0)) 

        # --- 新增：保存按钮 ---
        ttk.Button(left_panel, text='保存含水印图像', command=self.save_stego_image).pack(fill=tk.X, pady=5) 


        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        img_display_pane = ttk.PanedWindow(right_panel, orient=tk.HORIZONTAL)
        img_display_pane.pack(expand=True, fill=tk.BOTH, pady=(0,10))

        cover_label_frame = ttk.LabelFrame(img_display_pane, text="载体图像", width=280, height=300) 
        img_display_pane.add(cover_label_frame, weight=1)
        self.canvas_cover = tk.Canvas(cover_label_frame, bg="lightgrey")
        self.canvas_cover.pack(expand=True, fill=tk.BOTH)

        stego_label_frame = ttk.LabelFrame(img_display_pane, text="含水印图像", width=280, height=300) 
        img_display_pane.add(stego_label_frame, weight=1)
        self.canvas_stego = tk.Canvas(stego_label_frame, bg="lightgrey")
        self.canvas_stego.pack(expand=True, fill=tk.BOTH)
        
        self.progress_bar = ttk.Progressbar(right_panel, length=300, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(0,5))
        
        self.status_label_var = tk.StringVar(value="请先加载图像和水印。") 
        ttk.Label(right_panel, textvariable=self.status_label_var).pack(fill=tk.X)

        ttk.Button(left_panel, text='执行嵌入', command=self.perform_embedding, style="Accent.TButton").pack(fill=tk.X, pady=10, ipady=5) 
        style = ttk.Style()
        style.configure("Accent.TButton", font=('Helvetica', 10, 'bold'))

    def _update_image_canvas(self, canvas, np_image, title="图像"): # title参数改为中文
        if np_image is None:
            canvas.delete("all")
            canvas.update_idletasks()
            canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
            if canvas_w <=1: canvas_w=256
            if canvas_h <=1: canvas_h=256
            canvas.create_text(canvas_w//2, canvas_h//2, text=f"无{title}", anchor="center") # 显示中文
            return

        h_img, w_img = np_image.shape[:2]
        img_pil = Image.fromarray(np_image)
        
        canvas.update_idletasks()
        canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
        if canvas_w <= 1: canvas_w = 256 
        if canvas_h <= 1: canvas_h = 256

        scale = min(canvas_w / w_img, canvas_h / h_img, 1.0) 
        new_w, new_h = int(w_img * scale), int(h_img * scale)
        
        if new_w > 0 and new_h > 0:
            img_pil_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            photo_ref_attr = f'_photo_ref_{"cover" if canvas == self.canvas_cover else "stego"}'
            setattr(self, photo_ref_attr, ImageTk.PhotoImage(img_pil_resized))
            
            canvas.delete("all") 
            canvas.create_image(canvas_w//2, canvas_h//2, anchor="center", image=getattr(self, photo_ref_attr))
        else:
            canvas.delete("all")
            canvas.create_text(canvas_w//2, canvas_h//2, text=f"{title}尺寸问题", anchor="center") # 显示中文


    def update_max_capacity_display(self):
        if self.cover_image is None:
            self.max_embeddable_bits_info = "(请先加载图像)" 
        else:
            h, w = self.cover_image.shape
            num_blocks = (h // 8) * (w // 8)
            # 每个物理块现在只嵌入1个 (可能是重复编码后的) 比特
            self.max_embeddable_bits_info = f"{num_blocks} 比特 (物理块数)" 
        self.max_cap_label_var.set(f"最大可嵌入处理后比特数: {self.max_embeddable_bits_info}") 


    def load_cover_image(self):
        filepath = filedialog.askopenfilename(title="选择载体图像", filetypes=[('图像文件', '*.png;*.jpg;*.jpeg;*.bmp'), ('所有文件', '*.*')]) 
        if not filepath: return
        try:
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None: raise ValueError("无法读取图像文件。")
            h_orig, w_orig = img.shape
            h_new, w_new = (h_orig // 8) * 8, (w_orig // 8) * 8
            if h_new == 0 or w_new == 0:
                messagebox.showerror("错误", f"图像尺寸 ({w_orig}x{h_orig}) 过小。") 
                return
            if h_orig != h_new or w_orig != w_new:
                img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
                messagebox.showinfo("提示", f"图像已调整为 {w_new}x{h_new} 以适应8x8分块。") 
            
            self.cover_image = img
            self.update_max_capacity_display() 
            self.status_label_var.set(f"载体图像已加载 ({w_new}x{h_new})。") 
            self.canvas_cover.bind("<Configure>", lambda e, c=self.canvas_cover, i=self.cover_image, t="载体图像": self._update_image_canvas(c, i, t)) # title参数改为中文
            self._update_image_canvas(self.canvas_cover, self.cover_image, "载体图像") # title参数改为中文
            self.stego_image = None
            self._update_image_canvas(self.canvas_stego, self.stego_image, "含水印图像") # title参数改为中文
        except Exception as e:
            messagebox.showerror('加载错误', f'加载载体图像失败: {e}') 
            self.cover_image = None
            self.update_max_capacity_display()


    def load_watermark_bits(self):
        filepath = filedialog.askopenfilename(title="选择水印比特文件 (.txt)", filetypes=[('文本文件', '*.txt')]) 
        if not filepath: return
        try:
            with open(filepath, 'r') as f:
                bits_content = f.read().strip()
            if not all(c in '01' for c in bits_content):
                messagebox.showerror("错误", "水印文件内容无效，只能包含 '0' 和 '1'。") 
                return
            self.watermark_bits = bits_content
            self.bits_entry_var.set(self.watermark_bits[:50] + ("..." if len(self.watermark_bits) > 50 else ""))
            self.status_label_var.set(f"水印已加载 ({len(self.watermark_bits)} 比特)。") 
        except Exception as e:
            messagebox.showerror('加载错误', f'加载水印文件失败: {e}') 

    def use_entry_bits(self):
        bits_content = self.bits_entry_var.get().strip()
        if not all(c in '01' for c in bits_content):
            messagebox.showerror("错误", "输入的水印比特无效。") 
            return
        if not bits_content:
            messagebox.showwarning("提示", "输入的水印比特为空。") 
            return
        self.watermark_bits = bits_content
        self.status_label_var.set(f"使用输入水印 ({len(self.watermark_bits)} 比特)。") 

    def _progress_update_callback(self, current_step, total_steps):
        self.progress_bar['maximum'] = total_steps
        self.progress_bar['value'] = current_step
        self.update_idletasks()

    def perform_embedding(self):
        if self.cover_image is None:
            messagebox.showwarning('警告', '请先加载载体图像。') 
            return
        if not self.watermark_bits:
            messagebox.showwarning('警告', '请先加载或输入水印比特。') 
            return

        self.status_label_var.set("正在嵌入水印...") 
        self.progress_bar['value'] = 0
        
        strength = self.strength_var.get()
        enable_rep_code = self.cb_rep_code_var.get()
        rep_factor = self.rep_factor_var.get() if enable_rep_code else 1
        if enable_rep_code and rep_factor <= 0: 
            messagebox.showerror("错误", "启用时重复因子必须大于0。") 
            self.status_label_var.set("嵌入失败。") 
            return

        enable_scramble = self.cb_scramble_var.get()
        scramble_key = self.scramble_key_var.get() if enable_scramble else None
        
        try:
            self.stego_image = embed_watermark(
                self.cover_image, self.watermark_bits, strength,
                enable_rep_code, rep_factor,
                enable_scramble, scramble_key,
                self._progress_update_callback
            )
            self.canvas_stego.bind("<Configure>", lambda e, c=self.canvas_stego, i=self.stego_image, t="含水印图像": self._update_image_canvas(c, i, t)) # title参数改为中文
            self._update_image_canvas(self.canvas_stego, self.stego_image, "含水印图像") # title参数改为中文
            
            messagebox.showinfo('完成', f'水印嵌入完成！') 
            self.status_label_var.set(f"水印嵌入完成。") 
        except Exception as e:
            messagebox.showerror('嵌入错误', f'嵌入过程中发生错误: {e}') 
            self.status_label_var.set("嵌入失败。") 
            import traceback
            traceback.print_exc()

    def save_stego_image(self): 
        if self.stego_image is None:
            messagebox.showwarning('警告', '没有可保存的含水印图像。') 
            return
        filepath = filedialog.asksaveasfilename(title="保存含水印图像", defaultextension='.png', filetypes=[('PNG图像','*.png'), ('BMP图像', '*.bmp')]) 
        if filepath:
            try:
                cv2.imwrite(filepath, self.stego_image)
                messagebox.showinfo('成功', f'含水印图像已保存至: {filepath}') 
            except Exception as e:
                messagebox.showerror('保存错误', f'保存图像失败: {e}') 


    def visualize_selected_block(self):
        if self.cover_image is None:
            messagebox.showwarning('警告', '请先加载载体图像。') 
            return
        
        try:
            block_linear_idx = self.block_idx_var.get() # 这是原始顺序的块索引
        except tk.TclError:
            messagebox.showerror("错误", "块索引必须是有效的整数。") 
            return

        h,w = self.cover_image.shape
        max_idx_val = (h//8)*(w//8) -1

        if block_linear_idx < 0 or block_linear_idx > max_idx_val :
            messagebox.showwarning('警告', f'块索引超出范围 (0 到 {max_idx_val})。') 
            return
        
        blocks_per_row = self.cover_image.shape[1] // 8
        block_row = block_linear_idx // blocks_per_row
        block_col = block_linear_idx % blocks_per_row
        
        original_block_data = self.cover_image[block_row*8:(block_row+1)*8, block_col*8:(block_col+1)*8].astype(float)
        
        bit_for_vis = None
        if self.watermark_bits:
            temp_block_indices = list(range((h//8)*(w//8)))
            if self.cb_scramble_var.get() and self.scramble_key_var.get():
                try:
                    random.seed(str(self.scramble_key_var.get()))
                    random.shuffle(temp_block_indices)
                except: 
                    temp_block_indices = list(range((h//8)*(w//8))) 
            
            try:
                embedding_order_idx = temp_block_indices.index(block_linear_idx)
            except ValueError: 
                embedding_order_idx = -1

            if embedding_order_idx != -1:
                processed_bits_for_vis_idx = embedding_order_idx
                
                original_bits_temp = list(self.watermark_bits)
                processed_bits_temp = []
                if self.cb_rep_code_var.get() and self.rep_factor_var.get() > 1:
                    for b_orig in original_bits_temp:
                        processed_bits_temp.extend([b_orig] * self.rep_factor_var.get())
                else:
                    processed_bits_temp = original_bits_temp
                
                if processed_bits_for_vis_idx < len(processed_bits_temp):
                    bit_for_vis = processed_bits_temp[processed_bits_for_vis_idx]

        visualize_block_transform(original_block_data, bit_for_vis, self.strength_var.get())
                                  
if __name__ == '__main__':
    app = EmbedGUI()
    app.mainloop()
