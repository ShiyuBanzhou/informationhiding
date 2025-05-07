import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw

# Precompute DCT matrix
def dct_matrix(n=8):
    f = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==0:
                f[i,j]=math.sqrt(1/n)*math.cos((j+0.5)*math.pi*i/n)
            else:
                f[i,j]=math.sqrt(2/n)*math.cos((j+0.5)*math.pi*i/n)
    return f

F  = dct_matrix()
FT = F.T

# Extract bits
def extract_watermark(image, length, progress_callback=None):
    h,w = image.shape
    bits = []
    idx  = 0
    total = (h//8)*(w//8)
    for bi in range(0,h,8):
        for bj in range(0,w,8):
            block = image[bi:bi+8,bj:bj+8].astype(float)
            d = F @ block @ FT
            bits.append('1' if d[5,2]>=d[4,3] else '0')
            idx += 1
            if progress_callback:
                progress_callback(idx, total)
            if idx >= length:
                return ''.join(bits)
    return ''.join(bits)

# Visualization (extract-only)
def visualize_block_extract(orig_block):
    d = F @ orig_block @ FT
    m = np.abs(d)
    norm = ((m - m.min()) / np.ptp(m) * 255).astype(np.uint8)
    img = Image.fromarray(norm).resize((256,256))
    draw = ImageDraw.Draw(img)
    size = 256 // 8
    for (x,y) in [(5,2),(4,3)]:
        draw.rectangle([y*size,x*size,y*size+size-1,x*size+size-1], outline='blue')

    win = tk.Toplevel()
    win.title('Extract Block DCT')
    tk.Label(win, text='DCT Coeff Heatmap').pack(pady=(5,0))

    photo = ImageTk.PhotoImage(img)
    lbl   = tk.Label(win, image=photo)
    lbl.image = photo
    lbl.pack()

class ExtractGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('DCT 水印提取')
        self.stego    = None
        self.length   = tk.IntVar(value=4096)
        self.bits_out = ''

        ttk.Button(self, text='加载载密图像', command=self.load_image).pack(pady=4)
        frame = ttk.Frame(self); frame.pack(pady=4)
        ttk.Label(frame, text='提取长度:').pack(side=tk.LEFT)
        ttk.Entry(frame, textvariable=self.length, width=6).pack(side=tk.LEFT)
        ttk.Button(self, text='执行提取', command=self.do_extract).pack(pady=4)
        ttk.Button(self, text='保存水印文本', command=self.save_bits).pack(pady=4)

        frame2 = ttk.Frame(self); frame2.pack(pady=4)
        ttk.Label(frame2, text='块索引:').pack(side=tk.LEFT)
        self.block_idx = tk.IntVar(value=0)
        ttk.Entry(frame2, textvariable=self.block_idx, width=5).pack(side=tk.LEFT)
        ttk.Button(frame2, text='可视化该块', command=self.visualize_block).pack(side=tk.LEFT)

        self.pbar = ttk.Progressbar(self, length=300); self.pbar.pack(pady=4)
        self.txt  = tk.Text(self, width=40, height=16); self.txt.pack(pady=4)

    def load_image(self):
        p = filedialog.askopenfilename(filetypes=[('Image','*.png;*.jpg')])
        if not p: return
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror('错误','无法加载图像')
            return
        # 保证尺寸能整除8
        h, w = img.shape
        img = cv2.resize(img, ((w//8)*8, (h//8)*8))
        self.stego = img
        messagebox.showinfo('提示','载密图像已加载')

    def update_progress(self, val, total):
        self.pbar['maximum'] = total
        self.pbar['value']   = val
        self.update_idletasks()

    def do_extract(self):
        if self.stego is None:
            messagebox.showwarning('警告','请先加载载密图像')
            return
        bits = extract_watermark(self.stego, self.length.get(), self.update_progress)
        self.bits_out = bits
        self.txt.delete('1.0','end')
        for i in range(0, len(bits), 64):
            self.txt.insert('end', bits[i:i+64] + '\n')
        messagebox.showinfo('完成','提取完成')

    def save_bits(self):
        if not self.bits_out:
            return
        p = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text','*.txt')])
        if p:
            with open(p, 'w') as f:
                f.write(self.bits_out)
            messagebox.showinfo('提示','已保存水印文本')

    def visualize_block(self):
        if self.stego is None:
            messagebox.showwarning('警告','请先加载载密图像')
            return
        idx = self.block_idx.get()
        h, w = self.stego.shape
        per = w // 8
        max_idx = (h // 8) * per
        if idx < 0 or idx >= max_idx:
            messagebox.showwarning('警告', f'索引超出范围 (0~{max_idx-1})')
            return

        bi = (idx // per) * 8
        bj = (idx %  per) * 8
        block = self.stego[bi:bi+8, bj:bj+8].astype(float)
        visualize_block_extract(block)

if __name__ == '__main__':
    ExtractGUI().mainloop()
