import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

# --- LSB 位平面提取函数 ---
def lsb_bitplanes(embedded_np):
    """
    返回 8 个位平面数组（索引 0 为最低位，7 为最高位）
    """
    return [((embedded_np >> k) & 1).astype(np.uint8) * 255 for k in range(8)]

# --- 提取详情函数 ---
def extract_changes(original_np, embedded_np):
    """
    对比原始最低位与嵌入后最低位，返回变化列表
    每条：(i, j, orig_bit, emb_bit)
    """
    changes = []
    rows, cols = original_np.shape
    for i in range(rows):
        for j in range(cols):
            orig_bit = original_np[i, j] & 1
            emb_bit = embedded_np[i, j] & 1
            if orig_bit != emb_bit:
                changes.append((i, j, orig_bit, emb_bit))
    return changes

class ExtractGUI:
    def __init__(self, master):
        self.master = master
        master.title("LSB 水印提取工具")
        master.resizable(False, False)
        self.orig_np = None
        self.emb_np = None
        self.planes = None
        self.changes = None

        # 按钮区
        btn_frame = tk.Frame(master)
        btn_frame.grid(row=0, column=0, columnspan=4, pady=10)
        tk.Button(btn_frame, text="加载原始图像", command=self.load_orig).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="加载嵌入图像", command=self.load_emb).grid(row=0, column=1, padx=5)
        self.btn_extract = tk.Button(btn_frame, text="执行提取", command=self.extract, state=tk.DISABLED)
        self.btn_extract.grid(row=0, column=2, padx=5)
        self.btn_save = tk.Button(btn_frame, text="保存结果", command=self.save, state=tk.DISABLED)
        self.btn_save.grid(row=0, column=3, padx=5)

        # Notebook 视图
        self.notebook = ttk.Notebook(master)
        self.notebook.grid(row=1, column=0, columnspan=4)

        # Tab1: 位平面视图
        self.tab_planes = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_planes, text='位平面预览')
        self.plane_labels = []
        for k in range(8):
            lbl = tk.Label(self.tab_planes, text=f"Bit {k}", bg='lightgrey', compound='top')
            lbl.grid(row=k//4, column=k%4, padx=5, pady=5)
            self.plane_labels.append(lbl)

        # Tab2: 变化详情
        self.tab_changes = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_changes, text='变化详情')
        cols = ('pos','orig','emb')
        self.tree = ttk.Treeview(self.tab_changes, columns=cols, show='headings', height=15)
        for col, hd in zip(cols, ['位置(i,j)','原始LSB','嵌入LSB']):
            self.tree.heading(col, text=hd)
            self.tree.column(col, width=100, anchor='center')
        vsb = ttk.Scrollbar(self.tab_changes, orient='vertical', command=self.tree.yview)
        hsb = ttk.Scrollbar(self.tab_changes, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscroll=vsb.set, xscroll=hsb.set)
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        self.tab_changes.grid_rowconfigure(0, weight=1)
        self.tab_changes.grid_columnconfigure(0, weight=1)

    def _update_preview(self, label, arr):
        pil = Image.fromarray(arr).resize((200,200), Image.NEAREST)
        tkimg = ImageTk.PhotoImage(pil)
        label.config(image=tkimg)
        label.image = tkimg

    def load_orig(self):
        path = filedialog.askopenfilename(title='选择原始图像', filetypes=[('图像','*.png;*.jpg;*.bmp')])
        if not path: return
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror('加载错误','无法读取原始图像')
            return
        self.orig_np = img
        self.check_ready()

    def load_emb(self):
        path = filedialog.askopenfilename(title='选择嵌入图像', filetypes=[('图像','*.png;*.jpg;*.bmp')])
        if not path: return
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror('加载错误','无法读取嵌入图像')
            return
        self.emb_np = img
        self.check_ready()

    def check_ready(self):
        if self.orig_np is not None and self.emb_np is not None:
            self.btn_extract.config(state=tk.NORMAL)

    def extract(self):
        # 位平面
        self.planes = lsb_bitplanes(self.emb_np)
        for k, plane in enumerate(self.planes):
            self._update_preview(self.plane_labels[k], plane)
        # 变化详情
        self.changes = extract_changes(self.orig_np, self.emb_np)
        for item in self.tree.get_children():
            self.tree.delete(item)
        for idx, (i,j,ob,eb) in enumerate(self.changes[:500]):
            self.tree.insert('', 'end', values=(f"({i},{j})", ob, eb))
        if len(self.changes) > 500:
            self.tree.insert('', 'end', values=('...','共 %d 条' % len(self.changes),''))
        self.btn_save.config(state=tk.NORMAL)
        messagebox.showinfo('完成','提取完成！可在“位平面预览”和“变化详情”查看结果')

    def save(self):
        # 保存所有位平面
        dir_ = filedialog.askdirectory(title='选择保存目录')
        if not dir_: return
        for k, plane in enumerate(self.planes):
            cv2.imwrite(f"{dir_}/bitplane_{k}.png", plane)
        # 保存详情表CSV
        import csv
        csv_path = f"{dir_}/changes.csv"
        with open(csv_path,'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['i','j','orig_bit','emb_bit'])
            for i,j,ob,eb in self.changes:
                writer.writerow([i,j,ob,eb])
        messagebox.showinfo('保存','位平面图及变化详情已保存')

if __name__ == '__main__':
    root = tk.Tk()
    ExtractGUI(root)
    root.mainloop()