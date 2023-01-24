import matplotlib.pyplot as plt
import seaborn as sns
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
        dot = make_dot(model_out, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    else:
        dot = make_dot(model_out, params=dict(model.named_parameters()))
    resize_dot_graph(dot, size_per_element=1, min_size=20)

    if output_filepath:
        dot.format = "png"
        dot.render(output_filepath)

    return dot
