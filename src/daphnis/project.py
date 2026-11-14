"""Named workspaces for grouping sessions."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4


def validate_project_id(project_id: str) -> None:
    if not isinstance(project_id, str) or re.fullmatch(r"[0-9a-f]{32}", project_id) is None:
        raise ValueError("无效的项目 ID。")


@dataclass(frozen=True)
class Project:
    name: str
    workspace: Path
    id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        validate_project_id(self.id)
        name = self.name.strip() if isinstance(self.name, str) else ""
        if not 0 < len(name) <= 120 or not name.isprintable():
            raise ValueError("项目名称必须包含 1–120 个可打印字符。")
        object.__setattr__(self, "name", name)
