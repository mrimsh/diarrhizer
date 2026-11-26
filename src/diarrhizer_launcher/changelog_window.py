"""Tk release history for first launch, before the processing environment exists."""

import tkinter as tk
import webbrowser
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from diarrhizer.changelog import NEWER, OLDER, TITLE, UNAVAILABLE, Entry, blocks, inline


# [SEMANTIC-BEGIN] LAUNCHER:CHANGELOG
# @purpose: Browse offline release pages using only the launcher's bundled Tk and standard library
# @description: Render the document's headings, paragraphs, lists, emphasis, code and source links.
# @see: APP:CHANGELOG, LAUNCHER:UI
class ChangelogWindow(tk.Toplevel):
    def __init__(self, parent, entries: tuple[Entry, ...]):
        super().__init__(parent)
        self.title(TITLE)
        self.geometry("720x520")
        self.transient(parent)
        self.entries = entries
        frame = ttk.Frame(self, padding=12)
        frame.pack(fill="both", expand=True)
        self.selector = ttk.Combobox(frame, state="readonly", values=[e.title for e in entries])
        self.selector.pack(fill="x")
        self.selector.bind("<<ComboboxSelected>>", lambda _: self.show_page(self.selector.current()))
        self.browser = ScrolledText(frame, wrap="word", font=("Segoe UI", 10), padx=12, pady=8)
        self.browser.pack(fill="both", expand=True, pady=8)
        for tag, size in (("title", 16), ("heading", 12), ("bold", 10)):
            self.browser.tag_configure(tag, font=("Segoe UI", size, "bold"))
        self.browser.tag_configure("code", font=("Consolas", 10))
        self.browser.tag_configure("link", foreground="#1859a8", underline=True)
        self.browser.tag_configure("bullet", lmargin2=16)
        bar = ttk.Frame(frame)
        bar.pack(fill="x")
        self.newer = ttk.Button(bar, text=NEWER, command=lambda: self.show_page(self.selector.current() - 1))
        self.older = ttk.Button(bar, text=OLDER, command=lambda: self.show_page(self.selector.current() + 1))
        self.newer.pack(side="left")
        self.older.pack(side="right")
        self.page = ttk.Label(bar)
        self.page.pack(expand=True)
        self.show_page(0)
        self.bind("<Escape>", lambda _: self.destroy())
        self.selector.focus_set()

    def show_page(self, index: int):
        total = len(self.entries)
        self.newer.state(["!disabled" if index > 0 else "disabled"])
        self.older.state(["!disabled" if index < total - 1 else "disabled"])
        self.page.configure(text=f"{index + 1} / {total}" if total else "")
        self.browser.configure(state="normal")
        self.browser.delete("1.0", "end")
        if total:
            self.selector.current(index)
            entry = self.entries[index]
            self.browser.insert("end", entry.title + "\n\n", "title")
            for kind, line in blocks(entry.body):
                if kind == "bullet":
                    self.browser.insert("end", "• ", kind)
                for style, label, url in inline(line):
                    tags = [kind, style] if style else [kind]
                    if url:
                        tags.append(url)
                        self.browser.tag_bind(url, "<Button-1>", lambda _, target=url: webbrowser.open(target))
                    self.browser.insert("end", label, tuple(tags))
                self.browser.insert("end", "\n" if kind in ("bullet", "numbered") else "\n\n")
        else:
            self.selector.state(["disabled"])
            self.browser.insert("end", UNAVAILABLE)
        self.browser.configure(state="disabled")
        self.browser.yview_moveto(0)
# [SEMANTIC-END] LAUNCHER:CHANGELOG
