"""Shared QSettings key names, so screens that read/write the same setting
(e.g. the results folder, used by both History and New Job) don't drift.
"""

OUT_DIR = "app/out_dir"
DEFAULT_DEVICE = "app/default_device"
DEFAULT_ASR_MODEL = "app/default_asr_model"
CUSTOM_ASR_MODELS = "app/custom_asr_models"

# History: the column the table is sorted by ("history/sort_column": index, "history/sort_order": "asc"
# or "desc") and the widths the user dragged ("history/column_widths": JSON, see HistoryScreen).
HISTORY_SORT_COLUMN = "history/sort_column"
HISTORY_SORT_ORDER = "history/sort_order"
HISTORY_COLUMN_WIDTHS = "history/column_widths"

# New Job remembers the parameters of the last started job under this prefix
# (mode, language, speakers, audio profile, advanced ASR params, glossary).
NEW_JOB_PREFIX = "new_job/"
