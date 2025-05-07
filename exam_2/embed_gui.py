import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np

class EmbedWatermarkApp:
    def __init__(self, master):
        self.master = master
        master.title("嵌入脆弱水印 (奇偶校验)")
        self.frame = tk.Frame(master)
        self.frame.pack(padx=10, pady=10)

        # 嵌入按钮
        self.embed_btn = tk.Button(
            self.frame, text="嵌入水印", width=20, command=self.embed_watermark
        )
        self.embed_btn.grid(row=0, column=0, columnspan=3, pady=5)

        # 显示标签与画布：原图、隐秘载体、奇偶位图
        labels = ["原始图像", "隐秘载体", "校验位图"]
        for idx, txt in enumerate(labels):
            tk.Label(self.frame, text=txt).grid(row=1, column=idx)

        self.canvas_orig = tk.Label(self.frame)
        self.canvas_orig.grid(row=2, column=0, padx=5, pady=5)
        self.canvas_stego = tk.Label(self.frame)
        self.canvas_stego.grid(row=2, column=1, padx=5, pady=5)
        self.canvas_parity = tk.Label(self.frame)
        self.canvas_parity.grid(row=2, column=2, padx=5, pady=5)

        # 进度条和状态栏
        self.progress = ttk.Progressbar(
            self.frame, orient="horizontal", length=400, mode="determinate"
        )
        self.progress.grid(row=3, column=0, columnspan=3, pady=5)
        self.status_label = tk.Label(self.frame, text="等待嵌入...", fg="blue")
        self.status_label.grid(row=4, column=0, columnspan=3, pady=5)

    def embed_watermark(self):
        path = filedialog.askopenfilename(
            title="选择灰度图像",
            filetypes=[("Image", "*.png;*.jpg;*.jpeg"), ("All", "*.*")]
        )
        if not path:
            return
        img = Image.open(path).convert("L")
        arr = np.array(img)
        h, w = arr.shape

        # 初始化
        stego = np.zeros_like(arr)
        parity_map = np.zeros_like(arr)
        total = h * w
        self.progress['maximum'] = total
        self.progress['value'] = 0
        self.status_label.config(text="开始嵌入：0%", fg="black")

        # 逐像素处理
        count1 = 0
        for idx, (i, j) in enumerate(np.ndindex(h, w)):
            p = int(arr[i, j])
            p7 = p >> 1
            parity = bin(p7).count("1") % 2
            stego[i, j] = (p & ~1) | parity
            parity_map[i, j] = parity * 255
            count1 += parity
            # 更新进度
            if idx % (w * 5) == 0:
                pct = idx / total * 100
                self.progress['value'] = idx
                self.status_label.config(
                    text=f"嵌入进度：{pct:.1f}%"
                )
                self.master.update()

        stego_img = Image.fromarray(stego.astype(np.uint8))
        parity_img = Image.fromarray(parity_map.astype(np.uint8))

        # 保存
        save = filedialog.asksaveasfilename(
            title="保存隐秘载体",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")]
        )
        if not save:
            return
        stego_img.save(save)
        self.status_label.config(
            text=f"完成：嵌入位1共有{count1}个 (占比{count1/total*100:.2f}%), 文件：{save}",
            fg="green"
        )
        messagebox.showinfo(
            "嵌入完成",
            f"隐秘载体已保存：{save}\n嵌入位1共有 {count1} / {total} 个 ({count1/total*100:.2f}% )"
        )

        # 显示三幅图
        self._show(img, self.canvas_orig, interp=Image.LANCZOS)
        self._show(stego_img, self.canvas_stego, interp=Image.LANCZOS)
        self._show(parity_img, self.canvas_parity, interp=Image.NEAREST)

    def _show(self, pil_img, widget, interp):
        max_size = 256
        w, h = pil_img.size
        scale = min(max_size / w, max_size / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        img2 = pil_img.resize((nw, nh), interp)
        tk_img = ImageTk.PhotoImage(img2)
        widget.configure(image=tk_img)
        widget.image = tk_img

if __name__ == "__main__":
    root = tk.Tk()
    app = EmbedWatermarkApp(root)
    root.mainloop()