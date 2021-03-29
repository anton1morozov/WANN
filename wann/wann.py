import tensorflow as tf
from random import choice, randint
from enum import Enum
import json
import networkx as nx
import logging
from matplotlib import pyplot as plt
from copy import copy
from typing import List, Optional, Iterator, Union, Tuple, Dict

from wann.connection import Connection
from wann.activation import Activation
from wann.sensor_node import SensorNode
from wann.hidden_node import HiddenNode
from wann.output_node import OutputNode


class NodeBuildState(Enum):
    NOT_BUILT = 0,
    BUILT = 1,
    INACCESSIBLE = 2

# wann
# Possible mutations:
# 1) insert node (activation is randomly assigned)
# 2) add connection ("New connections are added between previously unconnected nodes,
#                       respecting the feed-forward property of the network.")
# 3) change activation


# NEAT
# "In NEAT, mutation can either mutate existing connections or can add new structure to a network."
# "If a new node is added, it is placed between two nodes that are already connected.
#  The previous connection is disabled (though still present in the genome).
#  The previous start node is linked to the new node with the weight of the old connection
#  and the new node is linked to the previous end node with a weight of 1."


# Weight values from the paper: [-2, -1, -0.5, +0.5, +1, +2],
# in later experiments: [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]


Node = Union[SensorNode, HiddenNode, OutputNode]


class Mutation(Enum):
    INSERT_NODE = 0,
    ADD_CONNECTION = 1,
    CHANGE_ACTIVATION = 2


class WANN:
    """
    Weight Agnostic Neural Network (WANN, https://arxiv.org/abs/1906.04358)
    """

    supported_output_activations = ['linear', 'softmax']

    # Networks topologies are judged based on three criteria:
    # 1) mean performance over all weight values,
    # 2) max performance of the single best weight value,
    # 3) and the number of connections in the network.

    def __init__(self, sensor_nodes: Optional[List[str]] = None,
                 output_nodes: Optional[List[str]] = None,
                 input_dict: Dict = None,
                 json_filename: str = None,
                 output_activation: str = 'linear'):
        """
        :param sensor_nodes: names to be assigned to all sensor nodes
        :param output_nodes: names to be assigned to all output nodes
        :param input_dict: dictionary, containing all required information for the WANN
        :param json_filename: json-file, from which the WANN must be loaded
        :param output_activation: activation, that must be applied to all output nodes
        """
        if output_activation:
            assert output_activation in self.supported_output_activations,\
                f"Unsupported output activation '{output_activation}'"

        # Genome initialization
        self.nodes_genes = []
        self.connections_genes = []

        # Mutation history
        self.mutation_history = []

        # Setting tensorflow config
        self.config = tf.ConfigProto(device_count={'GPU': 0})
        # self.config = tf.ConfigProto()

        # Creating lists for sensor, motor and hidden nodes to keep them separately
        self.sensor_nodes = []
        self.hidden_nodes = []
        self.output_nodes = []

        # Output activation, applied to all output nodes
        self.output_activation = output_activation

        # Creating 'sensor: placeholder' dictionary
        self.sensor_pl_d = dict()

        # Creating 'built' flag
        self.is_built = False

        # Tensorflow variables
        self.s = None
        self.input_pl = None
        self.shared_weight_pl = tf.placeholder(tf.float32, shape=(), name="shared_weight")
        self.out = None

        # FILLING

        # if json_filename is set, then load existing model from json-file
        if json_filename:
            self.load_json(filename=json_filename)

        # else if input dictionary is defined
        elif input_dict:
            self.from_dict(input_dict)

        # otherwise sensor_names and motor_names might be set
        else:

            # if sensor names are specified, add them to the structure
            if sensor_nodes:
                # print(f"len(sensor_nodes) = {len(sensor_nodes)}")
                self.input_pl = tf.placeholder(tf.float32, shape=(None, len(sensor_nodes)), name='X')
                self.create_sensors(sensor_nodes)

            # if motor names are specified, add them to the structure
            if output_nodes:
                self.create_motors(output_nodes)

    def create_sensors(self, sensor_names: List[str]) -> None:
        """
        Creating sensor nodes
        :param sensor_names: names for sensor nodes
        """
        for i, name in enumerate(sensor_names):
            node = SensorNode(name, self.input_pl, i)
            self.nodes_genes.append(node)
            self.sensor_nodes.append(node)
            self.sensor_pl_d.update({name: node.get_pl()})

    def create_motors(self, motor_names: List[str]) -> None:
        """
        Creating output nodes
        :param motor_names: names for output nodes
        """
        for name in motor_names:
            node = OutputNode(name)
            self.nodes_genes.append(node)
            self.output_nodes.append(node)

    def __eq__(self, other: 'WANN') -> bool:
        """
        Check if two WANNs are equal
        :param other: another WANN
        :return: comparison result, True if both WANNs have similar structure, False otherwise
        """

        # Checking if node counts are equal
        if len(self.sensor_nodes) != len(other.sensor_nodes):
            return False
        if len(self.hidden_nodes) != len(other.hidden_nodes):
            return False
        if len(self.output_nodes) != len(other.output_nodes):
            return False

        # Checking if all nodes and their activation functions are equal
        for i in range(len(self.nodes_genes)):

            # Getting corresponding nodes
            a_node = self.nodes_genes[i]
            b_node = other.nodes_genes[i]

            # Checking that nodes are of the same type
            if type(a_node) != type(b_node):
                return False

            # Checking names
            if a_node.name != b_node.name:
                return False

            # Checking count of the input or output connections
            if SensorNode == type(a_node) == type(b_node):
                if len(a_node.out_connections) != len(b_node.out_connections):
                    return False
            elif HiddenNode == type(a_node) == type(b_node):
                if len(a_node.in_connections) != len(b_node.in_connections):
                    return False
                if a_node.activation != b_node.activation:
                    return False
                if len(a_node.out_connections) != len(b_node.out_connections):
                    return False
            elif OutputNode == type(a_node) == type(b_node):
                if len(a_node.in_connections) != len(b_node.in_connections):
                    return False
            else:
                raise Exception(f"Unknown node types: '{type(a_node)}' and '{type(b_node)}'")

        # Checking if connections counts are equal
        if len(self.connections_genes) != len(other.connections_genes):
            return False

        # Checking all connections
        for i in range(len(self.connections_genes)):

            # Getting corresponding connections
            a_connection = self.connections_genes[i]
            b_connection = other.connections_genes[i]

            # Getting indexes for every node of each connection
            a_in_node_i = self.nodes_genes.index(a_connection.in_node)
            a_out_node_i = self.nodes_genes.index(a_connection.out_node)
            b_in_node_i = other.nodes_genes.index(b_connection.in_node)
            b_out_node_i = other.nodes_genes.index(b_connection.out_node)

            # Checking if nodes are equal
            if a_in_node_i != b_in_node_i or a_out_node_i != b_out_node_i:
                return False

            # Additional: checking if both connections are enabled/disabled
            if a_connection.is_enabled != b_connection.is_enabled:
                return False

        # If all checks have passed, WANNs are equal
        return True

    def init_randomly(self) -> 'WANN':
        """
        Randomly initializing minimal structure for the WANN
        """

        # if some structure exists already, erasing it
        self.connections_genes.clear()
        self.nodes_genes = [node for node in self.nodes_genes if isinstance(node, (SensorNode, OutputNode))]

        # Every motor should have at least one connection
        # Creating random connections between sensors and motors
        for output_node in self.output_nodes:

            # Creating several random connections of this motor node with some sensor nodes
            connected_sensor_nodes = []  # list to track already connected sensor nodes
            for _ in range(randint(1, 4)):  # we create from 1 to 4 random connections

                # Check if motor is not already connected to all possible sensor nodes
                if len(connected_sensor_nodes) == len(self.sensor_nodes):
                    # print("\tlen(connected_sensor_nodes) == len(self.sensor_nodes)")  # DEBUG
                    break

                # Getting new sensor node which was not connected to this motor
                sensor_node = choice([node for node in self.sensor_nodes if node not in connected_sensor_nodes])

                # Adding current sensor node to the list of already connected sensors
                connected_sensor_nodes.append(sensor_node)

                # Creating new connection
                connection = Connection(in_node=sensor_node, out_node=output_node, weight_pl=self.shared_weight_pl)

                # Adding created connection to sensor and motor nodes
                sensor_node.add_out_connection(connection)
                output_node.add_in_connection(connection)

                # Adding created connection to the genome
                self.connections_genes.append(connection)

        return self.set_levels()

    @property
    def complexity(self) -> int:
        """
        Get enabled connections count of the WANN
        :return: number of working connections in this WANN
        """
        return len([connection for connection in self.connections_genes if connection.is_enabled])

    def __copy__(self):
        return WANN(input_dict=self.to_dict())

    def __del__(self):
        pass

    def _walk_forward(self, input_node: Node) -> Iterator[Node]:
        """
        Helper function, used to get all nodes on top of this one
        :param input_node: node, from which iterator should walk
        :return: iterator, walking forward through nodes
        """

        # Creating a stack to hold all the nodes after this very node
        stack = [input_node]

        # While stack is not empty
        while stack:

            # Getting last element of the list with its removal
            node = stack.pop()

            if not isinstance(node, OutputNode):  # output nodes have no output connections
                stack = [connection.out_node for connection in node.out_connections] + stack

            # Releasing node to do whatever we want to do
            yield node

    def _walk_backward(self, input_node: Node, return_input_node: bool = True) -> Iterator[Node]:
        """
        Helper function, used to get all nodes on top of this one
        :param input_node: node, from which iterator should walk
        :param return_input_node: whether iterator of this method must return input node or not
        :return: iterator, walking forward through nodes
        """

        # Creating a stack to hold all the nodes after this very node
        stack = [input_node]

        # While stack is not empty
        while stack:

            # Getting last element of the list with its removal
            node = stack.pop()

            if not isinstance(node, SensorNode):  # sensor nodes have no input connections
                stack = [connection.in_node for connection in node.in_connections if connection.is_enabled] + stack

            # Releasing node to do whatever
            if node is input_node:
                if return_input_node:
                    yield node
            else:
                yield node

    def walk_by_levels(self) -> Iterator[List[Node]]:
        """
        Walk through all nodes in WANN in a level-by-level fashion
        :return:
        """

        max_level = max([node.level for node in self.output_nodes])

        level = 0
        stack = []
        while level <= max_level:
            if level == 0:
                to_yield = []
                for node in self.sensor_nodes:
                    to_yield.append(node)
                    stack.extend([_.out_node for _ in node.out_connections
                                  if _.out_node.level == 1 and _.out_node not in stack])
                yield to_yield
            elif level == max_level:
                yield [node for node in self.output_nodes]
            else:
                new_stack = []
                for node in stack:
                    new_stack.extend([_.out_node for _ in node.out_connections
                                      if _.out_node.level == level + 1 and not isinstance(_.out_node, OutputNode)
                                      and _.out_node not in new_stack])
                yield stack
                stack = new_stack
            level += 1

    def _is_connection_possible(self, in_node: Node, out_node: Node) -> bool:
        """
        Check if suggested connection is possible
        :param in_node: input node of suggested connection
        :param out_node: output node of suggested connection
        :return: True, if connection can be created, False otherwise
        """

        # input node can't exist before out_node
        if in_node in list(self._walk_backward(out_node, return_input_node=False)):
            return False

        # reverse connection can't exist either
        if out_node in list(self._walk_backward(in_node, return_input_node=False)):
            return False

        return True

    def _get_all_possible_connections(self) -> List[Tuple[Node, Node]]:
        """
        Get all possible new connections with the current WANN structure
        :return: list of tuples (input node, output node), containing nodes for possible new connections
        """

        # Getting list of all possible connections
        possible_connections = []

        for in_node in self.sensor_nodes + self.hidden_nodes:
            for out_node in self.hidden_nodes + self.output_nodes:
                if out_node is not in_node and self._is_connection_possible(in_node, out_node):
                    possible_connections.append((in_node, out_node))

        return possible_connections

    def mutate(self) -> 'WANN':
        """
        Add random change to this WANN
        :return: self
        """

        # Getting all possible connections
        possible_connections = self._get_all_possible_connections()

        # Getting available mutations for current architecture
        available_mutations = list(Mutation)

        # if not hidden nodes in the structure
        if not self.hidden_nodes:
            available_mutations.remove(Mutation.CHANGE_ACTIVATION)

        # if no possible connections left
        if not possible_connections:
            available_mutations.remove(Mutation.ADD_CONNECTION)

        # Choosing type of mutation
        mutation = choice(available_mutations)

        # Performing mutation
        if Mutation.INSERT_NODE == mutation:  # HERE LEVEL OF THE NODE AND ALL THE FOLLOWING NODES CAN BE CHANGED

            # Choosing random enabled connection
            connection = choice([connection for connection in self.connections_genes if connection.is_enabled])

            # Disabling connection
            connection.disable()

            # Getting input and output nodes of this connection
            in_node = connection.in_node
            out_node = connection.out_node

            # Creating a new node with random activation and adding it to hidden nodes list and to the genome
            new_node = HiddenNode(name=f"H_{len(self.hidden_nodes)}", activation=choice(list(Activation)))
            self.hidden_nodes.append(new_node)
            self.nodes_genes.append(new_node)

            # Creating new connections and adding them to the genome
            new_in_connection = Connection(in_node, new_node, weight_pl=self.shared_weight_pl)
            new_out_connection = Connection(new_node, out_node, weight_pl=self.shared_weight_pl)
            self.connections_genes.append(new_in_connection)
            self.connections_genes.append(new_out_connection)

            # Adding created connections to the hidden node
            new_node.add_in_connection(new_in_connection)
            new_node.add_out_connection(new_out_connection)

            # Adding new connections to in_node and out_node
            in_node.add_out_connection(new_in_connection)
            out_node.add_in_connection(new_out_connection)

            # Calculating the level of the new node
            new_node.level = new_in_connection.in_node.level + 1

            # Changing level of all the following nodes
            self.set_levels()

            # Adding mutation to the history
            history_entry = f"INSERT_NODE(name='{new_node.name}', activation={new_node.activation.name}"

        elif Mutation.ADD_CONNECTION == mutation:  # HERE LEVEL OF THE NODE CAN BE CHANGED

            in_node, out_node = choice(possible_connections)

            connection = Connection(in_node, out_node, weight_pl=self.shared_weight_pl)
            in_node.add_out_connection(connection)
            out_node.add_in_connection(connection)
            self.connections_genes.append(connection)

            # Changing level of all the following nodes
            self.set_levels()

            # Adding mutation to the history
            history_entry = f"ADD_CONNECTION(in_node='{in_node.name}', out_node='{out_node.name}')"

        elif Mutation.CHANGE_ACTIVATION == mutation:

            # Choosing random node from hidden ones to change its activation
            node = choice(self.hidden_nodes)

            # Getting all available activations without activation from the node
            available_activations = [activation for activation in Activation if activation != node.activation]

            # Changing activation
            node.activation = choice(available_activations)

            # Adding mutation to the history
            history_entry = f"CHANGE_ACTIVATION(node='{node.name}', new_activation={node.activation.name})"

        else:
            raise NotImplementedError(f"Mutation '{mutation}' is not implemented")

        self.mutation_history.append(history_entry)

        return self

    def set_levels(self) -> 'WANN':
        """
        Set correct levels for all nodes in the WANN
        :return: this WANN with correct level in every node
        """

        states = [False] * len(self.nodes_genes)

        while not all(states):

            for i, node in enumerate(self.nodes_genes):

                if isinstance(node, SensorNode):
                    node.level = 0
                    states[i] = True
                else:
                    levels = [connection.in_node.level for connection in node.in_connections]
                    if None not in levels:
                        node.level = max(levels) + 1
                        states[i] = True

        # Output node can't have level less than maximum level
        max_level = max([node.level for node in self.output_nodes])
        for node in self.output_nodes:
            node.level = max_level

        return self

    def clear_tf_graph(self) -> None:
        """
        Clear built tensorflow graph
        """
        self.is_built = False

    def build_tf_graph(self) -> None:
        """
        Build tensorflow graph of this WANN
        """

        self.s = tf.Session(config=self.config)

        # Building nodes build boolean list
        nodes_build_status = [NodeBuildState.NOT_BUILT for _ in self.nodes_genes]

        def build_tf_node(node):

            logging.info("Building node '%s' of type '%s'" % (node.name, str(type(node))))

            if isinstance(node, SensorNode):
                # if there are no output connections from this sensor node
                if not node.out_connections:
                    nodes_build_status[self.nodes_genes.index(node)] = NodeBuildState.INACCESSIBLE  # DEBUG
                    return
            else:
                # Check if all input connections are built
                if not node.all_in_connections_built():
                    logging.debug("Not all input connections are built")
                    return

            # Building current node
            node.build()
            nodes_build_status[self.nodes_genes.index(node)] = NodeBuildState.BUILT

            # Recursively calling building of all output nodes
            if not isinstance(node, OutputNode):
                for connection in node.out_connections:
                    if connection.is_enabled:
                        build_tf_node(connection.out_node)
            else:
                return

        # Walking through all sensor nodes
        for node in self.sensor_nodes:
            build_tf_node(node)

        # Checking if all nodes are built
        if NodeBuildState.NOT_BUILT in nodes_build_status:
            logging.debug("Not all nodes are built after build_tf_graph_v2 function")
            logging.debug("nodes_build_status: %s" % str([_.name for _ in nodes_build_status]))

        # making tf-operations to call with s.run(...)
        self.out = [node.out for node in self.output_nodes]  # fixme: problem if output node is inaccessible

        # Applying output activation to all output nodes
        if self.output_activation == 'linear':
            self.out = tf.stack(self.out, axis=1)
        if self.output_activation == 'softmax':
            self.out = tf.nn.softmax(tf.stack(self.out, axis=1), axis=1)

        self.is_built = True

    def run(self, data, weight):
        """
        Run computation through WANN using built tensorflow graph and provided weight value
        :param data: data to be processed
        :param weight: shared weight to use
        :return: numpy array, containing results of graph processing
        """
        assert self.s is not None
        assert self.out is not None
        return self.s.run(self.out, feed_dict={self.input_pl: data, self.shared_weight_pl: weight})

    def to_dict(self) -> Dict:
        """
        Return all infomation about this WANN as a dictionary object
        """

        # Creating dictionary
        d = dict()

        # Writing sensors to dictionary
        d['sensors'] = [{'name': node.name} for node in self.sensor_nodes]

        # Writing motors to dictionary
        d['motors'] = [{'name': node.name} for node in self.output_nodes]

        # Writing hidden nodes to dictionary
        d['hidden'] = [dict(name=node.name, activation=node.activation.name) for node in self.hidden_nodes]

        # Writing output activation to dictionary
        d['output_activation'] = self.output_activation

        # Writing connections to dictionary
        d['connections'] = []
        for connection in self.connections_genes:
            connection_d = dict()
            connection_d['in_node'] = connection.in_node.name
            connection_d['out_node'] = connection.out_node.name
            connection_d['innov'] = connection.innov
            connection_d['is_enabled'] = connection.is_enabled
            d['connections'].append(connection_d)

        # Writing mutation history to dictionary
        d['history'] = copy(self.mutation_history)

        return d

    def from_dict(self, d: Dict) -> None:
        """
        Restore this WANN, using information from dictionary
        :param d: dictionary, containing all required information for this WANN
        """

        # Creating sensors
        # print(json.dumps(d, indent=4))
        sensor_names = [_['name'] for _ in d['sensors']]
        self.input_pl = tf.placeholder(tf.float32, shape=(None, len(sensor_names)), name='X')
        self.create_sensors(sensor_names)
        sensor_name_node_d = dict(zip(sensor_names, self.sensor_nodes))

        # Creating motors
        motor_names = [_['name'] for _ in d['motors']]
        self.create_motors(motor_names)
        motor_name_node_d = dict(zip(motor_names, self.output_nodes))

        # Setting output activation
        self.output_activation = d['output_activation']

        # Creating hidden nodes
        hidden_name_node_d = dict()
        for hidden_node_d in d['hidden']:
            node = HiddenNode(name=hidden_node_d['name'], activation=Activation[hidden_node_d['activation']])
            self.hidden_nodes.append(node)
            self.nodes_genes.append(node)
            hidden_name_node_d[node.name] = node

        # Creating dictionary for all nodes
        name_node_d = {**sensor_name_node_d, **motor_name_node_d, **hidden_name_node_d}

        # Creating connections
        for connection_d in d['connections']:
            in_node = name_node_d[connection_d['in_node']]
            out_node = name_node_d[connection_d['out_node']]
            innov = connection_d['innov']
            is_enabled = connection_d['is_enabled']
            connection = Connection(in_node, out_node, innov=innov, is_enabled=is_enabled,
                                    weight_pl=self.shared_weight_pl)
            self.connections_genes.append(connection)
            in_node.add_out_connection(connection)
            out_node.add_in_connection(connection)

        # Loading mutation history
        self.mutation_history = d['history']

        # Setting levels of nodes
        self.set_levels()

    def save_json(self, filename: str, additional_info=None) -> None:
        """
        Save WANN to json file
        :param filename: name of the file, where WANN must be storred
        :param additional_info: additional information, that should be included in the file
        """

        d = self.to_dict()
        d.update(additional_info)
        with open(filename, 'w') as f:
            json.dump(d, f, indent=4)

        logging.info("wann saved in '%s'" % filename)

    def load_json(self, filename: str) -> None:
        """
        Load WANN from json file
        :param filename: file, where WANN is stored
        """

        with open(filename, 'r') as f:
            d = json.load(f)

        self.from_dict(d)

        logging.info("wann is loaded from json-file '%s'" % filename)

    def save_tf(self, filename: str) -> bool:
        """
        Save tensorflow graph separately
        :param filename: file, where tensorflow graph must be saved
        :return: True, if graph is saved, False otherwise
        """
        if not self.is_built:
            logging.error("TF graph is not built!")
            return False
        else:
            with tf.gfile.GFile(filename, 'wb') as f:
                f.write(self.s.graph.as_graph_def().SerializeToString())
                return True

    def draw(self) -> None:
        """
        Draw current WANNs structure
        """

        # Creating networkX graph object
        graph = nx.DiGraph()

        # Creating color_map for correct drawing
        color_map = []

        # Creating fixed_positions object to fix positions of sensor and motor nodes
        fixed_positions = dict()

        # Getting max level of all nodes to understand how to draw wann properly
        max_level = max([node.level for node in self.output_nodes])

        # Calculating level step
        level_step = 2. / max_level

        # Adding sensor nodes
        for i, node in enumerate(self.sensor_nodes):
            graph.add_node(node.name)
            color_map.append('green')
            fixed_positions.update({node.name: (-1.0, 1.0 - (i * 2 / len(self.sensor_nodes)))})

        # Adding motor nodes
        for i, node in enumerate(self.output_nodes):
            graph.add_node(node.name, type=1)
            color_map.append('skyblue')
            fixed_positions.update({node.name: (1.0, 1.0 - (i * 2 / len(self.output_nodes)))})

        # DEBUG
        # for nodes in self.walk_by_levels():
            # print("LEVEL %d: " % nodes[0].level + " ".join([_.name for _ in nodes]))

        # Walking through all levels
        hidden_nodes_count = 0
        for nodes in self.walk_by_levels():
            if 0 < nodes[0].level < max_level:
                hidden_nodes_count += len(nodes)
                for i, node in enumerate(nodes):
                    graph.add_node(node.name, type=2, activation=node.activation.name)
                    fixed_positions.update({node.name: (-1.0 + level_step * node.level, 1.0 - (i * 2 / len(nodes)))})
                    if Activation.LINEAR == node.activation:
                        color_map.append('#FFFFFF')
                    elif Activation.RELU == node.activation:
                        color_map.append((117 / 255, 218 / 255, 255 / 255))
                    elif Activation.SIGMOID == node.activation:
                        color_map.append((31 / 255, 190 / 255, 214 / 255))
                    elif Activation.TANH == node.activation:
                        color_map.append((0 / 255, 80 / 255, 0 / 255))
                    elif Activation.INVERSE == node.activation:
                        color_map.append((100 / 255, 100 / 255, 100 / 255))
                    elif Activation.STEP == node.activation:
                        color_map.append((125 / 255, 194 / 255, 75 / 255))
                    elif Activation.SIN == node.activation:
                        color_map.append((230 / 255, 243 / 255, 4 / 255))
                    elif Activation.COS == node.activation:
                        color_map.append((254 / 255, 209 / 255, 48 / 255))
                    elif Activation.GAUSSIAN == node.activation:
                        color_map.append((250 / 255, 175 / 255, 0 / 255))
                    elif Activation.ABS == node.activation:
                        color_map.append((253 / 255, 217 / 255, 181 / 255))
        # print("hidden_nodes_count: %d" % hidden_nodes_count)

        # DEBUG
        # print("FIXED POSITIONS:\n\t" + "\n\t".join(["%s: %s" % (_, str(fixed_positions[_])) for _ in fixed_positions]))

        # Adding edges
        for connection in self.connections_genes:
            if connection.is_enabled:
                graph.add_edge(connection.in_node.name, connection.out_node.name)

        # Creating custom layout
        pos = nx.spring_layout(graph, pos=fixed_positions, fixed=fixed_positions.keys())

        fig = plt.figure()
        nx.draw(graph,
                with_labels=True,
                pos=pos,
                node_size=800,
                node_color=color_map,
                edge_color='#F0F0F0',
                alpha=1.0,
                arrows=True)
        fig.set_facecolor('#252525')

        plt.show()
