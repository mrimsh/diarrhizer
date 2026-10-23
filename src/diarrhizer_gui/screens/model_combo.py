"""Editable list of the models a provider offers.

The list shows names (and ids), the value is the id the provider wants. The user can still type an id the list
does not have - a provider's list is a convenience, not a gate (a new model, a private deployment). The first
entry stands for "the provider's own default" where that is a valid choice (the subscription tools).
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox

from diarrhizer.summary.provider_info import ModelOption

DEFAULT_LABEL = "(по умолчанию у провайдера)"


def option_label(option: ModelOption) -> str:
    """"GPT-6-Astra (gpt-6-astra)"; the provider's default entry keeps its own name."""
    if option.id == "":
        return f"{DEFAULT_LABEL} — {option.name}" if option.name else DEFAULT_LABEL
    label = f"{option.name} ({option.id})" if option.name and option.name != option.id else option.id
    return f"{label} — по умолчанию" if option.default else label


class ModelCombo(QComboBox):
    def __init__(self) -> None:
        super().__init__()
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._options: dict[str, ModelOption] = {}
        self.set_models([], allow_default=True)

    def set_models(self, models: list[ModelOption], allow_default: bool) -> None:
        """Replace the list, keeping the current value. `allow_default`: offer "the provider's default"
        (unless the provider's own list has such an entry already)."""
        current = self.text()
        self.blockSignals(True)
        self.clear()
        self._options = {}
        entries = list(models)
        if allow_default and not any(m.id == "" for m in entries):
            # "The provider's default" - and, when its list says which model that is, say it.
            current_default = next((m for m in entries if m.default), None)
            entries.insert(0, ModelOption(
                id="", description=f"Сейчас по умолчанию: {current_default.label}" if current_default else ""
            ))
        elif not allow_default:
            entries = [m for m in entries if m.id != ""]
        for option in entries:
            self.addItem(option_label(option), option.id)
            self.setItemData(self.count() - 1, option.description, Qt.ItemDataRole.ToolTipRole)
            self._options[option.id] = option
        self.blockSignals(False)
        self.setText(current)

    def text(self) -> str:
        """The model id: the list entry's id if the shown text is one of the entries, else what was typed."""
        shown = self.currentText().strip()
        index = self.findText(shown)
        if index >= 0:
            return self.itemData(index) or ""
        return shown

    def setText(self, value: str) -> None:
        index = self.findData(value) if self.count() else -1
        if index >= 0:
            self.setCurrentIndex(index)
        else:
            self.setEditText(value)

    def option(self) -> Optional[ModelOption]:
        """The list entry the current value is, None for a typed id the provider's list lacks."""
        return self._options.get(self.text())
