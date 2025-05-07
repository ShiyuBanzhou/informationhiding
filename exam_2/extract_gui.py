import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

class ExtractWatermarkApp:
    def __init__(self, master):
        self.master = master
        master.title("提取并检测脆弱水印 (奇偶校验)")
        self.frame = tk.Frame(master)
        self.frame.pack(padx=10, pady=10)

        # 提取按钮
        self.extract_btn = tk.Button(
            self.frame, text="提取并检测", width=20, command=self.extract_and_detect
        )
        self.extract_btn.grid(row=0, column=0, columnspan=3, pady=5)

        # 显示标签与画布：隐秘载体、篡改覆盖、篡改地图
        labels = ["隐秘载体", "篡改覆盖 (红) , 透明度可调", "篡改地图"]
        for idx, txt in enumerate(labels):
            tk.Label(self.frame, text=txt).grid(row=1, column=idx)

        self.canvas_stego = tk.Label(self.frame)
        self.canvas_stego.grid(row=2, column=0, padx=5, pady=5)
        self.canvas_overlay = tk.Label(self.frame)
        self.canvas_overlay.grid(row=2, column=1, padx=5, pady=5)
        self.canvas_map = tk.Label(self.frame)
        self.canvas_map.grid(row=2, column=2, padx=5, pady=5)

        # 透明度滑块
        self.alpha_scale = ttk.Scale(
            self.frame, from_=0, to=1, orient=tk.HORIZONTAL,
            length=200, command=self.update_alpha
        )
        self.alpha_scale.set(0.5)
        tk.Label(self.frame, text="覆盖透明度").grid(row=3, column=1)
        self.alpha_scale.grid(row=4, column=1)

        # 状态栏
        self.status_label = tk.Label(self.frame, text="等待检测...", fg="blue")
        self.status_label.grid(row=5, column=0, columnspan=3, pady=5)

        # 存储中间图像
        self._stego = None
        self._overlay = None

    def extract_and_detect(self):
        path = filedialog.askopenfilename(
            title="选择隐秘载体",
            filetypes=[("PNG", "*.png"), ("Image", "*.jpg;*.jpeg"), ("All", "*.*")]
        )
        if not path:
            return
        img = Image.open(path).convert("L")
        arr = np.array(img)
        h, w = arr.shape
        total = h * w

        # 计算篡改地图和覆盖层
        tamper = np.ones_like(arr, dtype=np.uint8) * 255
        mask = np.zeros((h, w), dtype=bool)
        for (i, j) in np.ndindex(h, w):
            p = int(arr[i, j])
            parity = bin(p >> 1).count("1") % 2
            lsb = p & 1
            if lsb != parity:
                tamper[i, j] = 0
                mask[i, j] = True

        # 生成覆盖层：红色通道 tinted
        stego_rgb = img.convert("RGB")
        overlay = np.array(stego_rgb).astype(float)
        # mask 匹配处涂红
        overlay[mask] = [255, 0, 0]
        self._stego = stego_rgb
        self._overlay = Image.fromarray(overlay.astype(np.uint8))

        # 保存 map 图
        map_img = Image.fromarray(tamper)

        # 状态输出
        tampered_count = int(mask.sum())
        pct = tampered_count / total * 100
        if tampered_count:
            self.status_label.config(
                text=f"检测完成：共{tampered_count}/{total}像素被篡改 ({pct:.2f}%)", fg="red"
            )
        else:
            self.status_label.config(
                text="检测完成：未检测到篡改。", fg="green"
            )

        # 初始渲染
        self._render(img, self.canvas_stego, interp=Image.LANCZOS)
        self._render(self._overlay, self.canvas_overlay, interp=Image.LANCZOS)
        self._render(map_img, self.canvas_map, interp=Image.NEAREST)

    def update_alpha(self, val):
        if self._stego is None or self._overlay is None:
            return
        # 混合原图与 overlay
        alpha = float(val)
        base = np.array(self._stego).astype(float)
        over = np.array(self._overlay).astype(float)
        blended = (1 - alpha) * base + alpha * over
        blended_img = Image.fromarray(blended.astype(np.uint8))
        self._render(blended_img, self.canvas_overlay, interp=Image.LANCZOS)

    def _render(self, pil_img, widget, interp):
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
    app = ExtractWatermarkApp(root)
    root.mainloop()
