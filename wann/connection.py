from typing import Union
import json

from wann.sensor_node import SensorNode
from wann.hidden_node import HiddenNode
from wann.output_node import OutputNode


class Connection:
    """
    Connection between nodes in WANN
    """

    def __init__(self, in_node: Union[SensorNode, HiddenNode],
                 out_node: Union[HiddenNode, OutputNode],
                 weight_pl,
                 is_enabled: bool = True,
                 innov: int = 0):
        """
        :param in_node: input node of this connection
        :param out_node: output node of this connection
        :param weight_pl: weight placeholder
        :param is_enabled: whether this connection is enabled of not
        :param innov:
        """
        assert isinstance(in_node, (SensorNode, HiddenNode)), f"Input node must be either SensorNode or HiddenNode"
        assert isinstance(out_node, (HiddenNode, OutputNode)), f"Output node must be either HiddenNode or OutputNode"
        assert isinstance(is_enabled, bool)
        assert isinstance(innov, int)

        self.in_node = in_node
        self.out_node = out_node
        self.is_enabled = is_enabled
        self.innov = innov
        self.weight_pl = weight_pl
        self.out = None

    def build(self, in_op):
        """
        Building connection output
        :param in_op:
        """
        self.out = in_op * self.weight_pl

    def enable(self):
        """
        Enable this connection
        """
        self.is_enabled = True

    def disable(self):
        """
        Disable this connection
        """
        self.is_enabled = False

    def to_json(self):
        """

        :return:
        """
        json_d = dict()
        json_d['in_node'] = self.in_node.name
        json_d['out_node'] = self.out_node.name
        json_d['innov'] = self.innov
        json_d['is_enabled'] = self.is_enabled
        return json.dumps(json_d, indent=4)
