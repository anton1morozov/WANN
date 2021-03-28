from typing import Optional


class BaseNode:
    """
    Base class of all other nodes
    """
    def __init__(self, name: str, level: Optional[int] = None):
        """
        :param name: name of the node
        :param level: level of the node in WANN
        """
        assert isinstance(name, str), f"Name must be of type 'str', got {type(name)}"
        self.name = name
        self.level = level
