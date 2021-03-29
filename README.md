# Weight Agnostic Neural Networks (WANN) on Tensorflow 1.14

Original paper: https://arxiv.org/abs/1906.04358 <br/>
Also cool interactive article: https://weightagnostic.github.io/

Main idea short: there are neural network structures that can perform good on certain task without any specific weight. More than that, a weight shared between all connections can be used.

To do:
- add loss function and target metric plotting
- add custom loss function and metric support
- add distributed training using MPI
- make article-like WANN structure drawing
- add shared weight fitting (choose single value, that gives WANN with current structure best target metric)
- add fitting for WANN with different weights
- add unittests
