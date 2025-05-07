import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw

# Precompute 8x8 DCT transform matrix
def dct_matrix(n=8):
    f = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0:
                f[i, j] = math.sqrt(1 / n) * math.cos((j + 0.5) * math.pi * i / n)
            else:
                f[i, j] = math.sqrt(2 / n) * math.cos((j + 0.5) * math.pi * i / n)
    return f

F  = dct_matrix()
FT = F.T

# Embed watermark bits into image
def embed_watermark(image, bits, progress_callback=None):
    h, w = image.shape
    stego = np.zeros_like(image)
    total = (h // 8) * (w // 8)
    idx = 0
    for bi in range(0, h, 8):
        for bj in range(0, w, 8):
            block = image[bi:bi+8, bj:bj+8].astype(float)
            d = F @ block @ FT  # DCT
            if idx < len(bits):
                bit = bits[idx]
                a, b = d[5, 2], d[4, 3]
                if (bit == '1' and a < b) or (bit == '0' and a > b) or (a == b):
                    d[5, 2], d[4, 3] = b, a
                idx += 1
            block_idct = FT @ d @ F  # IDCT
            stego[bi:bi+8, bj:bj+8] = np.clip(block_idct, 0, 255)
            if progress_callback:
                progress_callback(idx, total)
    return stego.astype(np.uint8)

# Heatmap generation for DCT matrices
def heatmap(dmat):
    m = np.abs(dmat)
    norm = ((m - m.min()) / np.ptp(m) * 255).astype(np.uint8)
    img = Image.fromarray(norm).resize((256, 256))
    draw = ImageDraw.Draw(img)
    size = 256 // 8
    for (x, y) in [(5, 2), (4, 3)]:
        draw.rectangle([y * size, x * size, y * size + size - 1, x * size + size - 1], outline='red')
    return img

# Visualization of a single block
def visualize_block_transform(orig_block, bit=None):
    """
    弹窗展示单个 8×8 块的 DCT 前后热力图，以及对应系数的数值。
    """
    # 1. 计算 DCT 前向量和后向量
    d_before = F @ orig_block @ FT
    if bit is not None:
        d_after = d_before.copy()
        a0, b0 = d_before[5,2], d_before[4,3]
        # 嵌入逻辑：如果要嵌入 1，就保证 a>=b；要嵌入 0，就保证 a<=b
        if (bit=='1' and a0 < b0) or (bit=='0' and a0 > b0) or (a0==b0):
            d_after[5,2], d_after[4,3] = b0, a0
        a1, b1 = d_after[5,2], d_after[4,3]
    else:
        d_after = None
        a0=b0=a1=b1 = None

    # 2. 生成热力图 PhotoImage
    img_before = heatmap(d_before)
    photo_before = ImageTk.PhotoImage(img_before)
    img_after  = heatmap(d_after) if d_after is not None else None
    photo_after = ImageTk.PhotoImage(img_after) if img_after else None
    photo_idct  = None
    if d_after is not None:
        idct_arr = FT @ d_after @ F
        img_idct = Image.fromarray(
            np.clip(idct_arr,0,255).astype(np.uint8)
        ).resize((256,256))
        photo_idct = ImageTk.PhotoImage(img_idct)

    # 3. 弹窗显示
    win = tk.Toplevel()
    win.title('Block Visualization（含数值）')

    # DCT Before
    tk.Label(win, text='► DCT Before').pack(pady=(5,0))
    lbl_b = tk.Label(win, image=photo_before)
    lbl_b.image = photo_before
    lbl_b.pack()
    if a0 is not None:
        tk.Label(win,
            text=f'Before: d[5,2]={a0:.2f},  d[4,3]={b0:.2f}'
        ).pack(pady=(0,5))

    # DCT After
    if photo_after:
        tk.Label(win, text='► DCT After').pack(pady=(5,0))
        lbl_a = tk.Label(win, image=photo_after)
        lbl_a.image = photo_after
        lbl_a.pack()
        tk.Label(win,
            text=f' After: d[5,2]={a1:.2f},  d[4,3]={b1:.2f}    (嵌入 bit={bit})'
        ).pack(pady=(0,5))

    # IDCT 重建
    if photo_idct:
        tk.Label(win, text='► IDCT Result').pack(pady=(5,0))
        lbl_i = tk.Label(win, image=photo_idct)
        lbl_i.image = photo_idct
        lbl_i.pack()
        # 计算空间域最大差异，展示给用户看“肉眼损伤”
        diff = orig_block - (FT @ d_after @ F)
        tk.Label(win,
            text=f'空间域重建最大差异：{np.max(np.abs(diff)):.2f}'
        ).pack(pady=(0,5))

    # 让窗口根据内容自动适应大小
    win.geometry('')

# GUI for embedding
class EmbedGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('DCT 水印嵌入')
        self.cover = None
        self.bits  = ''
        # 按钮和控件
        ttk.Button(self, text='加载载体图像',    command=self.load_image).pack(pady=4)
        ttk.Button(self, text='加载水印文件',    command=self.load_bits).pack(pady=4)
        ttk.Button(self, text='执行嵌入',        command=self.do_embed).pack(pady=4)
        ttk.Button(self, text='保存载密图像',    command=self.save_image).pack(pady=4)

        frame_v = ttk.Frame(self); frame_v.pack(pady=4)
        ttk.Label(frame_v, text='可视化块索引:').pack(side=tk.LEFT)
        self.block_idx = tk.IntVar(value=0)
        ttk.Entry(frame_v, textvariable=self.block_idx, width=5).pack(side=tk.LEFT)
        ttk.Button(frame_v, text='可视化该块', command=self.visualize_block).pack(side=tk.LEFT)

        self.pbar = ttk.Progressbar(self, length=300); self.pbar.pack(pady=4)
        # 显示载体与载密图
        self.cv_cover = tk.Canvas(self, width=256, height=256); self.cv_cover.pack(side=tk.LEFT, padx=6)
        self.cv_stego = tk.Canvas(self, width=256, height=256); self.cv_stego.pack(side=tk.RIGHT, padx=6)
        self.stego = None

    def load_image(self):
        p = filedialog.askopenfilename(filetypes=[('Image', '*.png;*.jpg;*.bmp')])
        if not p: return
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror('错误', '无法加载图像')
            return
        self.cover = img
        thumb = Image.fromarray(img).resize((256,256))
        self.cv_cover.image = ImageTk.PhotoImage(thumb)
        self.cv_cover.create_image(0,0,anchor='nw',image=self.cv_cover.image)

    def load_bits(self):
        p = filedialog.askopenfilename(filetypes=[('Text', '*.txt')])
        if not p: return
        with open(p, 'r') as f:
            self.bits = f.read().strip()
        messagebox.showinfo('提示', f'加载水印，共 {len(self.bits)} 比特')

    def do_embed(self):
        if self.cover is None or not self.bits:
            messagebox.showwarning('警告', '请先加载图像和水印')
            return
        def cb(idx, total):
            self.pbar['maximum'] = total
            self.pbar['value']   = idx
            self.update_idletasks()
        self.stego = embed_watermark(self.cover, self.bits, cb)
        thumb = Image.fromarray(self.stego).resize((256,256))
        self.cv_stego.image = ImageTk.PhotoImage(thumb)
        self.cv_stego.create_image(0,0,anchor='nw',image=self.cv_stego.image)
        messagebox.showinfo('完成', '嵌入完成')

    def save_image(self):
        if self.stego is None:
            messagebox.showwarning('警告', '无可保存图像')
            return
        p = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG','*.png')])
        if p:
            cv2.imwrite(p, self.stego)
            messagebox.showinfo('提示', f'已保存到 {p}')

    def visualize_block(self):
        if self.cover is None:
            messagebox.showwarning('警告', '请先加载图像')
            return
        idx = self.block_idx.get()
        h, w = self.cover.shape
        max_idx = (h // 8) * (w // 8)
        if idx < 0 or idx >= max_idx:
            messagebox.showwarning('警告', f'索引超出范围 (0~{max_idx-1})')
            return

        per = w // 8
        bi = (idx // per) * 8
        bj = (idx %  per) * 8
        block = self.cover[bi:bi+8, bj:bj+8].astype(float)
        bit   = self.bits[idx] if idx < len(self.bits) else None
        visualize_block_transform(block, bit)

if __name__ == '__main__':
    EmbedGUI().mainloop()
