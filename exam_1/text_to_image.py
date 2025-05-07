"""
一个 GUI 程序，根据用户输入的两行文字生成 256×256 灰度图。
如果只输入一行，可将第二行留空。
用法：
    python text_to_image_gui.py
"""
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageFont, ImageTk

class TextImageGUI:
    def __init__(self, master):
        self.master = master
        master.title("文本生成灰度图 (256×256)")
        master.resizable(False, False)
        # 输入区域
        tk.Label(master, text="第一行:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.entry1 = tk.Entry(master, width=25)
        self.entry1.grid(row=0, column=1, columnspan=2, padx=5, pady=5)
        tk.Label(master, text="第二行:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
        self.entry2 = tk.Entry(master, width=25)
        self.entry2.grid(row=1, column=1, columnspan=2, padx=5, pady=5)
        # 按钮
        tk.Button(master, text="生成图像", command=self.generate).grid(row=2, column=1, pady=10)
        self.save_btn = tk.Button(master, text="保存图像", command=self.save, state=tk.DISABLED)
        self.save_btn.grid(row=2, column=2, pady=10)
        # 预览区域：自适应大小
        self.preview_label = tk.Label(master, bg="lightgrey")
        self.preview_label.grid(row=3, column=0, columnspan=3, padx=5, pady=5)
        self.image = None

    def generate(self):
        text1 = self.entry1.get().strip()
        text2 = self.entry2.get().strip()
        if not text1 and not text2:
            messagebox.showwarning("输入错误", "请至少在一行输入文字。")
            return
        size = 256
        img = Image.new('L', (size, size), color=255)
        draw = ImageDraw.Draw(img)
        # 动态字体大小
        max_len = max(len(text1), len(text2), 1)
        font_size = max(10, min(64, int(size / max_len * 1.2)))
        try:
            font = ImageFont.truetype("simhei.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        # 计算文本尺寸，使用 textbbox
        if text1:
            bbox1 = draw.textbbox((0, 0), text1, font=font)
            w1 = bbox1[2] - bbox1[0]
            h1 = bbox1[3] - bbox1[1]
        else:
            w1 = h1 = 0
        if text2:
            bbox2 = draw.textbbox((0, 0), text2, font=font)
            w2 = bbox2[2] - bbox2[0]
            h2 = bbox2[3] - bbox2[1]
        else:
            w2 = h2 = 0
        total_h = h1 + h2
        y = (size - total_h) // 2
        # 绘制文本
        if text1:
            x1 = (size - w1) // 2
            draw.text((x1, y), text1, fill=0, font=font)
        if text2:
            x2 = (size - w2) // 2
            draw.text((x2, y + h1 + 10), text2, fill=0, font=font)
        # 更新预览
        self.image = img
        tkimg = ImageTk.PhotoImage(img)
        self.preview_label.config(image=tkimg)
        self.preview_label.image = tkimg
        self.save_btn.config(state=tk.NORMAL)

    def save(self):
        if not self.image:
            return
        path = filedialog.asksaveasfilename(
            title="保存灰度图像",
            defaultextension=".png",
            filetypes=[("PNG 文件", "*.png")]
        )
        if not path:
            return
        self.image.save(path)
        messagebox.showinfo("保存成功", f"已保存到 {path}")

if __name__ == '__main__':
    root = tk.Tk()
    gui = TextImageGUI(root)
    root.mainloop()