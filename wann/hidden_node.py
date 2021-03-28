import tensorflow as tf
import json

# from wann.connection import Connection
from wann.activation import Activation
from wann.base_node import BaseNode


class HiddenNode(BaseNode):
    """
    Inner node of WANN, performing mathematical operation on its input
    """

    def __init__(self, name: str, activation: Activation):
        """

        :param name:
        :param activation:
        """
        super().__init__(name)
        # assert isinstance(activation, NodeActivation)
        self.activation = activation
        self.in_connections = []
        self.out_connections = []
        self.out = None

    def build(self):
        """
        Building tensorflow graph through this node
        """

        # Summing all input connections
        self.out = tf.reduce_sum(tf.stack([_.out for _ in self.in_connections if _.is_enabled], axis=1), axis=1)

        # Building node output
        if Activation.RELU == self.activation:
            self.out = tf.nn.relu(self.out)
        elif Activation.SIGMOID == self.activation:
            self.out = tf.nn.sigmoid(self.out)
        elif Activation.TANH == self.activation:
            self.out = tf.nn.tanh(self.out)
        elif Activation.INVERSE == self.activation:
            self.out = -self.out
        elif Activation.STEP == self.activation:
            self.out = tf.cast(tf.math.greater(self.out, tf.constant(0, dtype=tf.float32)), dtype=tf.float32)
        elif Activation.SIN == self.activation:
            self.out = tf.math.sin(self.out)
        elif Activation.COS == self.activation:
            self.out = tf.math.cos(self.out)
        elif Activation.GAUSSIAN == self.activation:
            self.out = tf.math.exp(-tf.math.pow(self.out, 2))
        elif Activation.ABS == self.activation:
            self.out = tf.math.abs(self.out)

        # Building output connections
        for connection in self.out_connections:
            if connection.is_enabled:
                connection.build(self.out)

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
        """
        Adding input connection to this node
        :param connection:
        """
        # assert isinstance(connection, Connection)
        self.in_connections.append(connection)

    def add_out_connection(self, connection):
        # assert isinstance(connection, Connection)
        self.out_connections.append(connection)

    def no_in_connections(self):
        """

        :return:
        """
        return not self.in_connections

    def no_out_connections(self):
        """

        :return:
        """
        return not self.out_connections

    def to_json(self):
        json_d = dict()
        json_d['name'] = self.name
        json_d['level'] = self.level
        if Activation.RELU == self.activation:
            json_d['activation'] = 'relu'
        elif Activation.SIGMOID == self.activation:
            json_d['activation'] = 'sigmoid'
        elif Activation.TANH == self.activation:
            json_d['activation'] = 'tanh'
        elif Activation.INVERSE == self.activation:
            json_d['activation'] = 'inverse'
        elif Activation.STEP == self.activation:
            json_d['activation'] = 'step'
        elif Activation.SIN == self.activation:
            json_d['activation'] = 'sin'
        elif Activation.COS == self.activation:
            json_d['activation'] = 'cos'
        elif Activation.GAUSSIAN == self.activation:
            json_d['activation'] = 'gaussian'
        elif Activation.ABS == self.activation:
            json_d['activation'] = 'abs'
        return json.dumps(json_d, indent=4)

    def __repr__(self):
        return f"{self.name}:\n" \
               f"\tActivation: {self.activation}\n" \
               f"\tLevel: {self.level}\n" \
               f"\tNumber of input connections: {len(self.in_connections)}\n" \
               f"\tNumber of output connections: {len(self.out_connections)}\n" \
               f"\tOutput tensor: {self.out}"

