"""Paged release notes in the main application."""

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)

from diarrhizer.changelog import NEWER, OLDER, TITLE, UNAVAILABLE, load_history


# [SEMANTIC-BEGIN] GUI:CHANGELOG
# @purpose: Open the local release notes at the newest page from the sidebar
# @description: The version selector controls both adjacent-page buttons and the formatted document.
# @see: APP:CHANGELOG, GUI:MAIN_WINDOW
class ChangelogDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(TITLE)
        self.resize(720, 520)
        self.entries = load_history()
        layout = QVBoxLayout(self)
        self.selector = QComboBox()
        self.selector.setAccessibleName("Перейти к версии")
        self.selector.addItems([entry.title for entry in self.entries])
        layout.addWidget(self.selector)
        self.browser = QTextBrowser()
        self.browser.setOpenLinks(False)
        self.browser.anchorClicked.connect(self._open_source)
        layout.addWidget(self.browser)
        bar = QHBoxLayout()
        layout.addLayout(bar)
        self.newer = QPushButton(NEWER)
        self.older = QPushButton(OLDER)
        self.page = QLabel()
        bar.addWidget(self.newer)
        bar.addStretch()
        bar.addWidget(self.page)
        bar.addStretch()
        bar.addWidget(self.older)
        self.newer.clicked.connect(lambda: self.selector.setCurrentIndex(self.selector.currentIndex() - 1))
        self.older.clicked.connect(lambda: self.selector.setCurrentIndex(self.selector.currentIndex() + 1))
        self.selector.currentIndexChanged.connect(self._show_page)
        self._show_page(self.selector.currentIndex())

    @staticmethod
    def _open_source(url: QUrl):
        if url.scheme() == "https":
            QDesktopServices.openUrl(url)

    def _show_page(self, index: int):
        total = len(self.entries)
        self.selector.setEnabled(bool(total))
        self.newer.setEnabled(index > 0)
        self.older.setEnabled(0 <= index < total - 1)
        self.page.setText(f"{index + 1} / {total}" if total else "")
        if total:
            entry = self.entries[index]
            self.browser.setMarkdown(f"# {entry.title}\n\n{entry.body}")
            self.browser.verticalScrollBar().setValue(0)
        else:
            self.browser.setPlainText(UNAVAILABLE)
# [SEMANTIC-END] GUI:CHANGELOG
