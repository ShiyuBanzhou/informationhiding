import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

# --- 多位平面嵌入函数（记录位平面变化） ---
def embed_multiple_planes(original_np, watermark_list):
    """
    watermark_list: list of tuples (wm_np, plane_index)
    返回嵌入后的图像和变化列表，每条变化：(i, j, plane, orig_bit, new_bit)
    """
    embedded = original_np.copy()
    changes = []
    for wm_np, plane in watermark_list:
        rows_w, cols_w = wm_np.shape
        wm_bin = np.where(wm_np > 128, 1, 0).astype(np.uint8)
        for i in range(rows_w):
            for j in range(cols_w):
                orig_pixel = int(embedded[i, j])
                orig_bit = (orig_pixel >> plane) & 1
                bit = int(wm_bin[i, j])
                # 清除目标位，再设置新位
                mask_clear = ~(1 << plane)
                new_pixel = (orig_pixel & mask_clear) | (bit << plane)
                new_bit = (new_pixel >> plane) & 1
                if orig_bit != new_bit:
                    changes.append((i, j, plane, orig_bit, new_bit))
                embedded[i, j] = new_pixel
    return embedded, changes

class EmbedGUI:
    def __init__(self, master):
        self.master = master
        master.title("多位平面水印嵌入工具")
        master.resizable(False, False)
        self.orig_np = None
        self.watermark_list = []
        self.changes = None

        # 原图面板
        tk.Label(master, text="载体图像").grid(row=0, column=0)
        self.lbl_orig = tk.Label(master, bg='lightgrey')
        self.lbl_orig.grid(row=1, column=0, padx=5, pady=5)

        # 控制区
        ctrl = tk.Frame(master)
        ctrl.grid(row=1, column=1, padx=10)
        tk.Button(ctrl, text="加载载体图像", command=self.load_orig).grid(row=0, column=0, pady=5)
        tk.Label(ctrl, text="选择位平面:").grid(row=1, column=0, pady=5)
        self.plane_var = tk.IntVar(value=0)
        self.plane_menu = ttk.Combobox(ctrl, textvariable=self.plane_var, values=list(range(8)), width=5)
        self.plane_menu.grid(row=2, column=0, pady=5)
        tk.Button(ctrl, text="添加水印图像", command=self.add_watermark).grid(row=3, column=0, pady=5)
        self.wm_listbox = tk.Listbox(ctrl, height=6, width=25)
        self.wm_listbox.grid(row=4, column=0, pady=5)
        tk.Button(ctrl, text="移除选中水印", command=self.remove_watermark).grid(row=5, column=0, pady=5)
        self.btn_embed = tk.Button(ctrl, text="执行嵌入", command=self.embed, state=tk.DISABLED)
        self.btn_embed.grid(row=6, column=0, pady=5)

        # 结果面板
        tk.Label(master, text="嵌入结果").grid(row=0, column=2)
        self.lbl_res = tk.Label(master, bg='lightgrey')
        self.lbl_res.grid(row=1, column=2, padx=5, pady=5)

        # 查看与保存
        btn_frame = tk.Frame(master)
        btn_frame.grid(row=2, column=0, columnspan=3, pady=10)
        self.btn_view = tk.Button(btn_frame, text="查看嵌入详情", command=self.view_details, state=tk.DISABLED)
        self.btn_view.grid(row=0, column=0, padx=5)
        self.btn_save = tk.Button(btn_frame, text="保存结果", command=self.save, state=tk.DISABLED)
        self.btn_save.grid(row=0, column=1, padx=5)

    def _update_img(self, label, arr):
        pil = Image.fromarray(arr)
        pil.thumbnail((200,200), Image.LANCZOS)
        tkimg = ImageTk.PhotoImage(pil)
        label.config(image=tkimg, width=pil.width, height=pil.height)
        label.image = tkimg

    def load_orig(self):
        path = filedialog.askopenfilename(title='选择载体图像', filetypes=[('图像','*.png;*.jpg;*.bmp')])
        if not path: return
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror('错误','无法读取载体图像')
            return
        self.orig_np = img
        self._update_img(self.lbl_orig, img)
        self.check_ready()

    def add_watermark(self):
        if self.orig_np is None:
            messagebox.showwarning('提示','请先加载载体图像')
            return
        path = filedialog.askopenfilename(title='选择水印图像', filetypes=[('图像','*.png;*.jpg;*.bmp')])
        if not path: return
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror('错误','无法读取水印图像')
            return
        plane = self.plane_var.get()
        self.watermark_list.append((img, plane))
        self.wm_listbox.insert('end', f"{path.split('/')[-1]} -> 位 {plane}")
        self.check_ready()

    def remove_watermark(self):
        sel = self.wm_listbox.curselection()
        if not sel: return
        idx = sel[0]
        self.wm_listbox.delete(idx)
        self.watermark_list.pop(idx)
        self.check_ready()

    def check_ready(self):
        state = tk.NORMAL if self.orig_np is not None and self.watermark_list else tk.DISABLED
        self.btn_embed.config(state=state)

    def embed(self):
        emb, self.changes = embed_multiple_planes(self.orig_np, self.watermark_list)
        self.res_np = emb
        self._update_img(self.lbl_res, emb)
        self.btn_view.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.NORMAL)
        messagebox.showinfo('完成', '已成功嵌入所有水印')

    def view_details(self):
        top = tk.Toplevel(self.master)
        top.title('嵌入详情')
        notebook = ttk.Notebook(top)
        notebook.pack(expand=True, fill='both')

        # Tab1: 位平面图像对比
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text='位平面图像')
        # 收集受影响的位平面
        planes = sorted(set(c[2] for c in self.changes))
        for idx, plane in enumerate(planes):
            orig_plane = ((self.orig_np >> plane) & 1).astype(np.uint8) * 255
            new_plane = ((self.res_np >> plane) & 1).astype(np.uint8) * 255
            pil_o = Image.fromarray(orig_plane).resize((200,200), Image.NEAREST)
            pil_n = Image.fromarray(new_plane).resize((200,200), Image.NEAREST)
            lbl_o = tk.Label(tab1, text=f'Plane {plane} 原始', compound='top')
            lbl_o.grid(row=0, column=2*idx, padx=5, pady=5)
            tkimg_o = ImageTk.PhotoImage(pil_o)
            lbl_o.config(image=tkimg_o)
            lbl_o.image = tkimg_o
            lbl_n = tk.Label(tab1, text=f'Plane {plane} 新', compound='top')
            lbl_n.grid(row=0, column=2*idx+1, padx=5, pady=5)
            tkimg_n = ImageTk.PhotoImage(pil_n)
            lbl_n.config(image=tkimg_n)
            lbl_n.image = tkimg_n

        # Tab2: 逐像素列表
        tab2 = ttk.Frame(notebook)
        notebook.add(tab2, text='逐像素详情')
        cols = ('pos','plane','orig_bit','new_bit')
        tree = ttk.Treeview(tab2, columns=cols, show='headings', height=10)
        for col, hd in zip(cols, ['位置','平面','原始比特','新比特']):
            tree.heading(col, text=hd)
            tree.column(col, width=100, anchor='center')
        vs = ttk.Scrollbar(tab2, orient='vertical', command=tree.yview)
        tree.configure(yscroll=vs.set)
        tree.grid(row=0,column=0,sticky='nsew')
        vs.grid(row=0,column=1,sticky='ns')
        tab2.grid_rowconfigure(0,weight=1); tab2.grid_columnconfigure(0,weight=1)
        for idx, (i,j,plane,ob,nb) in enumerate(self.changes[:500]):
            tree.insert('','end',values=(f'({i},{j})',plane,ob,nb))
        if len(self.changes)>500:
            tree.insert('','end',values=('...','共 %d 条'%len(self.changes),'',''))

    def save(self):
        path = filedialog.asksaveasfilename(title='保存结果', defaultextension='.png', filetypes=[('PNG','*.png')])
        if not path: return
        cv2.imwrite(path, self.res_np)
        messagebox.showinfo('保存','已保存嵌入结果')

if __name__ == '__main__':
    root = tk.Tk()
    EmbedGUI(root)
    root.mainloop()