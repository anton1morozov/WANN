from typing import Optional


class BaseNode:
    """
    Base class of all other nodes
    """
    def __init__(self, name: str, level: Optional[int] = None):
        assert isinstance(name, str), f"Name must be of type 'str', got {type(name)}"
        # assert isinstance(level, int), f"Level must be of type 'int', got {type(level)}"
        self.name = name
        self.level = level
