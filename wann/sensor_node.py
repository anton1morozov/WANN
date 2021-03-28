import tensorflow as tf
import json
from .base_node import BaseNode


class SensorNode(BaseNode):
    """
    Input node of wann
    """

    def __init__(self, name: str, input_pl, idx: int):
        """
        :param name: name of the node
        :param input_pl: placeholder, from which current sensor data must be taken
        :param idx: index by which data for this sensor node must be taken
        """
        super().__init__(name, level=0)
        self.placeholder = tf.placeholder(dtype=tf.float32, shape=[], name=name)
        self.input_pl = input_pl
        self.idx = idx
        self.out_connections = []
        self.out = None

    def build(self):
        """
        Building output connections
        """
        self.out = self.input_pl[:, self.idx]
        for connection in self.out_connections:
            if connection.is_enabled:
                connection.build(self.out)

    def add_out_connection(self, connection):
        """
        Adding output connection
        :param connection: connection to add
        """
        self.out_connections.append(connection)

    def no_out_connections(self) -> bool:
        """
        Check if there are no output connections from this node
        :return: True, if there are no output connections, False otherwise
        """
        return not self.out_connections

    def get_pl(self):
        """
        Get placeholder of this sensor node
        """
        return self.placeholder

    def to_json(self) -> str:
        """
        Save this node as a json string
        """
        return json.dumps(dict(name=self.name), indent=4)
