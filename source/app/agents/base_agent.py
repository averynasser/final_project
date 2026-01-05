from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """
    Base class untuk semua agent.
    """

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    @abstractmethod
    def run(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        task: deskripsi tugas dalam natural language
        context: informasi tambahan antar agent
        return: dict yang berisi hasil dan context baru
        """
        pass
