import tensorflow as tf
import json

# from wann.connection import Connection
from wann.base_node import BaseNode


class OutputNode(BaseNode):
    """
    Output node of WANN
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.in_connections = []
        self.out = None

    def build(self):
        """
        Building node output
        """

        # DEBUG
        # for connection in self.in_connections:
        #     if connection.is_enabled:
        #         print(f"connection.in_node.name: {connection.in_node.name}")
        #         print(f"connection.in_node.out: {connection.in_node.out}")
        #         print(f"connection.out: {connection.out}")

        self.out = tf.reduce_sum(tf.stack([_.out for _ in self.in_connections if _.is_enabled], axis=1), axis=1)

    def all_in_connections_built(self) -> bool:
        """
        Check if all input connections are build
        :return: check result
        """
        for connection in self.in_connections:
            if connection.is_enabled:
                if connection.out is None:
                    return False
        return True

    def add_in_connection(self, connection):
        # assert isinstance(connection, Connection)
        self.in_connections.append(connection)

    def no_in_connections(self):
        return not self.in_connections

    def to_json(self):
        return json.dumps({'name': self.name}, indent=4)

    def __repr__(self):
        return f"{self.name}:\n" \
               f"\tLevel: {self.level}\n" \
               f"\tNumber of input connections: {len(self.in_connections)}\n" \
               f"\tOutput tensor: {self.out}"
