# Learning to optimize

[Intro](#intro) - [Usage](#usage) - [Results](#results) - [Acknowledgements](#acknowledgements)

---

## Intro
Combining Learning-to-Optimize (L2O) with insights about symmetries in network architectures and Deep Learning training dynamics.
- Related to:
  - [Learning to Optimize: A Primer and A Benchmark](https://arxiv.org/abs/2103.12828)
  - [Learning to learn by gradient descent by gradient descent](https://arxiv.org/abs/1606.04474)
  - [Neural Mechanics: Symmetry and Broken Conservation Laws in Deep Learning Dynamics](https://arxiv.org/abs/2012.04728)


## Usage
The most important pieces of the code are in the Jupyter Notebook `main.ipynb`. L2O Optimizer is implemented in `optimizer.py`, helper meta-modules are in `meta_modules.py`, and optimizee models are in `optimizee.py`. Both meta-training and meta-testing scripts are in `training.py` (meta-testing <=> pretrained L2O optimizer *trains/optimizes* the optimizee) and called from `main.ipynb`. Additionally, `target.py` contains the task on which optimizee is trained (MNIST).


## Results

List of available results:
- [Initial comparison](#initial-comparison)
- [Deviations from symmetry constraints](#deviations-from-symmetry-constraints)
    - [Rescale symmetry (ReLU, Leaky ReLU, Linear, etc.)](#rescale-symmetry-relu-leaky-relu-linear-etc)
    - [Translation symmetry (Softmax)](#translation-symmetry-softmax)
    - [Scale symmetry (Batch normalization)](#scale-symmetry-batch-normalization)
- [Comparion after regularizing the L2O optimizer against symmetry constraint deviations](#comparison-after-regularizing-the-l2o-optimizer-against-symmetry-constraint-deviations)

### Initial comparison
Comparisons of L2O optimizers with Adam(lr=0.03) and SGD(lr=0.1, momentum=0.9) - from left to right: Optimizee with Sigmoid, ReLU, and ReLU with Batch normalization (L2O optimizer trained for each separately). All optimizees have 1 hidden layer of 20 neurons and Softmax in the output layer. Batch normalization - affine=True, track_running_stats=False.
<p align="center">
  <img src="./results/imgs/MNISTNet_50e.png" width="32%" alt="MNISTNet" />
  <img src="./results/imgs/MNISTRelu_100e.png" width="32%" alt="MNISTRelu" />
  <img src="./results/imgs/MNISTReluBatchNorm_affine_no_stats_tracking_75e.png" width="32%" alt="MNISTReluBatchNorm" />
</p>

### Deviations from symmetry constraints
Below are shown deviations from the geometric constraints on gradients of the loss wrt the optimizee's parameters that arise from symmetries in the network architecture. All plots are for optimizees (MLPs) with 1 hidden layer of 20 neurons (ReLU if no other is specified) and Softmax in the output layer. For the scale symmetry, additional Batch normalization is added (affine=True, track_running_stats=False) before the ReLU activation function. Theory behind it can be found in [Neural Mechanics: Symmetry and Broken Conservation Laws in Deep Learning Dynamics](https://arxiv.org/abs/2012.04728).

*Note: All plots have different scales on the y-axis.*

#### Rescale symmetry (ReLU, Leaky ReLU, Linear, etc.)

![Rescale symmetry (ReLU) - during meta-testing of the L2O optimizer](./results/imgs/rescale_sym_relu_metatesting.png)
![Rescale symmetry (ReLU) - during meta-training of the L2O optimizer](./results/imgs/rescale_sym_relu_metatraining.png)


#### Translation symmetry (Softmax)

![Translation symmetry (Softmax) - during meta-testing of the L2O optimizer](./results/imgs/translation_sym_softmax_relu_optee_metatesting.png)
![Translation symmetry (Softmax) - during meta-training of the L2O optimizer](./results/imgs/translation_sym_softmax_relu_optee_metatraining.png)


#### Scale symmetry (Batch normalization)

![Scale symmetry (Batch normalization) - during meta-testing of the L2O optimizer](./results/imgs/scale_sym_batchnorm_relu_optee_metatesting.png)
![Scale symmetry (Batch normalization) - during meta-training of the L2O optimizer](./results/imgs/scale_sym_batchnorm_relu_optee_metatraining.png)


### Comparison after regularizing the L2O optimizer against symmetry constraint deviations

Below is shown the influence of regularizing the L2O optimizer during meta-training against deviations from the geometric constraints on gradients of the loss wrt the optimizee's parameters. In the legend of each plot, one can see the regularization strength and the target of regularization during the meta-training phase.

*Click to enlarge the plots*
<table style="padding: 10px">
    <tr>
        <th>Meta-testing optimizee architecture</th>
        <th>Meta-trained on MLP w/ 20 hidden units, sigmoid</th>
        <th>Meta-trained on MLP w/ 20 hidden units, leaky ReLU</th>
        <th>Meta-trained on MLP w/ 20 hidden units, ReLU, batch normalization</th>
    </tr>
    <tr>
        <td>
            <p>20 hidden units, sigmoid</p>
        </td>
        <td>
            <img src="./results/imgs/regularization/MNISTNet_meta_training/losses_comparison_MNISTNet_{}_meta_trained_on_MNISTNet.png" width="100%" alt="MNISTNet to MNISTNet" />
        </td>
        <td>
            <img src="./results/imgs/regularization/MNISTLeakyRelu_meta_training/losses_comparison_MNISTRelu_{}_meta_trained_on_MNISTLeakyRelu.png" width="100%" alt="MNISTLeakyRelu to MNISTNet" />
        </td>
        <td>
            <img src="./results/imgs/regularization/MNISTReluBatchNorm_meta_training/losses_comparison_MNISTNet_{}_meta_trained_on_MNISTReluBatchNorm.png" width="100%" alt="MNISTReluBatchNorm to MNISTNet" />
        </td>
    </tr>
    <tr>
        <td>
            <p>20 hidden units, leaky ReLU</p>
        </td>
        <td>
            <img src="./results/imgs/regularization/MNISTNet_meta_training/losses_comparison_MNISTLeakyRelu_{}_meta_trained_on_MNISTNet.png" width="100%" alt="MNISTNet to MNISTLeakyRelu" />
        </td>
        <td>
            <img src="./results/imgs/regularization/MNISTLeakyRelu_meta_training/losses_comparison_MNISTLeakyRelu_{}_meta_trained_on_MNISTLeakyRelu.png" width="100%" alt="MNISTLeakyRelu to MNISTLeakyRelu" />
        </td>
        <td>
            <img src="./results/imgs/regularization/MNISTReluBatchNorm_meta_training/losses_comparison_MNISTLeakyRelu_{}_meta_trained_on_MNISTReluBatchNorm.png" width="100%" alt="MNISTReluBatchNorm to MNISTLeakyRelu" />
        </td>
    </tr>
    <tr>
        <td>
            <p>20 hidden units, ReLU, batch normalization</p>
        </td>
        <td>
            <img src="./results/imgs/regularization/MNISTNet_meta_training/losses_comparison_MNISTReluBatchNorm_{'affine': True, 'track_running_stats': False}_meta_trained_on_MNISTNet.png" width="100%" alt="MNISTNet to MNISTReluBatchNorm" />
        </td>
        <td>
            <img src="./results/imgs/regularization/MNISTLeakyRelu_meta_training/losses_comparison_MNISTReluBatchNorm_{'affine': True, 'track_running_stats': False}_meta_trained_on_MNISTLeakyRelu.png" width="100%" alt="MNISTLeakyRelu to MNISTReluBatchNorm" />
        </td>
        <td>
            <img src="./results/imgs/regularization/MNISTReluBatchNorm_meta_training/losses_comparison_MNISTReluBatchNorm_{'affine': True, 'track_running_stats': False}_meta_trained_on_MNISTReluBatchNorm.png" width="100%" alt="MNISTReluBatchNorm to MNISTReluBatchNorm" />
        </td>
    </tr>
    <tr>
        <td>
            <p>two layers with 20 hidden units each, sigmoid</p>
        </td>
        <td>
            <img src="./results/imgs/regularization/MNISTNet_meta_training/losses_comparison_MNISTNet2Layer_{}_meta_trained_on_MNISTNet.png" width="100%" alt="MNISTNet to MNISTNet2Layer" />
        </td>
        <td>
            <img src="./results/imgs/regularization/MNISTLeakyRelu_meta_training/losses_comparison_MNISTNet2Layer_{}_meta_trained_on_MNISTLeakyRelu.png" width="100%" alt="MNISTLeakyRelu to MNISTNet2Layer" />
        </td>
        <td>
            <img src="./results/imgs/regularization/MNISTReluBatchNorm_meta_training/losses_comparison_MNISTNet2Layer_{}_meta_trained_on_MNISTReluBatchNorm.png" width="100%" alt="MNISTReluBatchNorm to MNISTNet2Layer" />
        </td>
    </tr>
    <tr>
        <td>
            <p>40 hidden units, sigmoid</p>
        </td>
        <td>
            <img src="./results/imgs/regularization/MNISTNet_meta_training/losses_comparison_MNISTNetBig_{}_meta_trained_on_MNISTNet.png" width="100%" alt="MNISTNet to MNISTNetBig" />
        </td>
        <td>
            <img src="./results/imgs/regularization/MNISTLeakyRelu_meta_training/losses_comparison_MNISTNetBig_{}_meta_trained_on_MNISTLeakyRelu.png" width="100%" alt="MNISTLeakyRelu to MNISTNetBig" />
        </td>
        <td>
            <img src="./results/imgs/regularization/MNISTReluBatchNorm_meta_training/losses_comparison_MNISTNetBig_{}_meta_trained_on_MNISTReluBatchNorm.png" width="100%" alt="MNISTReluBatchNorm to MNISTNetBig" />
        </td>
</table>


## Acknowledgements
* Pytorch version of NIPS'16 "Learning to learn by gradient descent by gradient descent" [chenwydj/learning-to-learn-by-gradient-descent-by-gradient-descent](https://github.com/chenwydj/learning-to-learn-by-gradient-descent-by-gradient-descent).
* Original L2O code from [AdrienLE/learning_by_grad_by_grad_repro](https://github.com/AdrienLE/learning_by_grad_by_grad_repro).
* Meta modules from [danieltan07/learning-to-reweight-examples](https://github.com/danieltan07/learning-to-reweight-examples).