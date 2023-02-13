import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import clear_output, display
from torchviz import make_dot


def resize_dot_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.
    Modify the graph in place.

    Author: ucalyptus (https://github.com/ucalyptus): https://github.com/szagoruyko/pytorchviz/issues/41#issuecomment-699061964
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)
    return dot


def get_model_dot(model, model_out, show_detailed_grad_info=True, output_filepath=None):
    if show_detailed_grad_info:
        dot = make_dot(
            model_out,
            params=dict(model.named_parameters()),
            show_attrs=True,
            show_saved=True,
        )
    else:
        dot = make_dot(model_out, params=dict(model.named_parameters()))
    resize_dot_graph(dot, size_per_element=1, min_size=20)

    if output_filepath:
        dot.format = "png"
        dot.render(output_filepath)

    return dot


class LivePlot:
    """
    A class to plot live data in a Jupyter notebook.
    - How to use:
        import numpy as np
        live_plot = LivePlot(figsize=(10,5), groups=["train", "val"], use_seaborn=True)
        for i in range(10):
            y_train = np.random.rand()
            y_val = np.random.rand()
            live_plot.update({"train": y_train, "val": y_val}, display=True) ### or display=False and call live_plot.draw() later
    """

    def __init__(self, figsize=(10, 5), use_seaborn=False, groups=["default"]):
        """
        Params:
            figsize: tuple of int - example: (10, 5); default: (10, 5)
            use_seaborn: bool - example: True; default: False
            groups: list of str - example: ["train", "val"]; default: None
        """
        self.fig = None
        self.figsize = figsize
        self.groups = groups
        self.use_seaborn = use_seaborn
        self.history = {group_name: [] for group_name in self.groups}

    def setup_fig(self):
        self.fig = plt.figure(figsize=self.figsize)
        if self.groups is not None:
            n_rows = (
                len(self.groups) // 2 + len(self.groups) % 2
                if len(self.groups) > 1
                else 1
            )
            n_cols = 2 if len(self.groups) > 1 else 1
            self.group_axes = {
                group_name: self.fig.add_subplot(n_rows, n_cols, i + 1)
                for i, group_name in enumerate(self.groups)
            }
        else:
            self.group_axes = {"default": self.fig.add_subplot(111)}

    def draw(self):
        if self.fig is None:
            self.setup_fig()
        for group_name, ax in self.group_axes.items():
            ax.cla()
            if self.use_seaborn:
                sns.lineplot(self.history[group_name], ax=ax)
            else:
                ax.plot(self.history[group_name])
            if group_name != "default":
                ax.set_title(group_name, fontsize=14, fontweight="bold")
            ax.set_xlabel("Timestep", fontsize=12)
        display(self.fig)
        clear_output(wait=True)

    def update(self, values, display=True, reset=False):
        """
        Params:
            values: dict of {group_name: y} OR float OR int - example: {"train": 0.5, "val": 0.6} OR 0.5 OR 5
                float or int will be added to the default group
            display: bool
        """
        if type(values) in (float, int):
            values = {"default": values}
        for group, y in values.items():
            if hasattr(y, "__iter__"):
                if reset:
                    self.history[group] = y
                else:
                    self.history[group].extend(y)
            else:
                if reset:
                    self.history[group] = [y]
                else:
                    self.history[group].append(y)
        if display:
            self.draw()

    def reset(self):
        self.history = {group_name: [] for group_name in self.history.keys()}
