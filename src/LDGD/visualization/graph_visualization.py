import plotly.graph_objects as go
import numpy as np
import networkx as nx
import math
import urllib.request
import matplotlib.pyplot as plt
import matplotlib


def get_pos(radius, name=''):
    sd = np.pi / 12
    ssd = np.pi / 24
    positions_anatomical = {
        'LFIa' + name: (radius * np.cos(np.pi - ssd), radius * np.sin(np.pi - ssd)),
        'RFIa' + name: (radius * np.cos(np.pi + ssd), radius * np.sin(np.pi + ssd)),

        'LFIp' + name: (radius * np.cos(np.pi - sd - ssd), radius * np.sin(np.pi - sd - ssd)),
        'RFIp' + name: (radius * np.cos(np.pi + sd + ssd), radius * np.sin(np.pi + sd + ssd)),

        'LTP' + name: (radius * np.cos(np.pi + 2 * sd + ssd), radius * np.sin(np.pi + 2 * sd + ssd)),
        'RTP' + name: (radius * np.cos(np.pi - 2 * sd - ssd), radius * np.sin(np.pi - 2 * sd - ssd)),

        'LTB' + name: (radius * np.cos(np.pi / 2), radius * 0.7 * np.sin(np.pi / 2)),
        'RTB' + name: (radius * np.cos(-np.pi / 2), radius * 0.7 * np.sin(-np.pi / 2)),

        'LTSa' + name: (radius * np.cos(np.pi / 2 + sd), radius * np.sin(np.pi / 2 + sd)),
        'RTSa' + name: (radius * np.cos(-np.pi / 2 - sd), radius * np.sin(-np.pi / 2 - sd)),

        'LTSp' + name: (radius * np.cos(np.pi / 2 - sd), radius * np.sin(np.pi / 2 - sd)),
        'RTSp' + name: (radius * np.cos(-np.pi / 2 + sd), radius * np.sin(-np.pi / 2 + sd)),

        'RPHa' + name: (radius * np.cos(np.pi / 2 + 2 * sd), radius * np.sin(np.pi / 2 + 2 * sd)),
        'LPHa' + name: (radius * np.cos(-np.pi / 2 - 2 * sd), radius * np.sin(-np.pi / 2 - 2 * sd)),

        'RPHp' + name: (radius * np.cos(2 * sd), radius * np.sin(2 * sd)),
        'LPHp' + name: (radius * np.cos(-2 * sd), radius * np.sin(-2 * sd)),

        'LAMY' + name: (radius * np.cos(sd), radius * 0.5 * np.sin(sd)),
        'RAMY' + name: (radius * np.cos(-sd), radius * 0.5 * np.sin(-sd)),

    }
    return positions_anatomical


"""channel_name = ['LFIa02 LFIa03', 'LFIa03 LFIa04', 'LFIa06 LFIa07', 'LFIa07 LFIa08', 'LFIp01 LFIp02', 'LFIp02 LFIp03',
                'LFIp03 LFIp04', 'LFIp04 LFIp05', 'LFIp05 LFIp06', 'LFIp06 LFIp07', 'LFIp07 LFIp08', 'LTP-01 LTP-02',
                'LTP-02 LTP-03', 'LTP-03 LTP-04', 'LTP-04 LTP-05', 'LTP-05 LTP-06', 'LTP-06 LTP-07', 'LTP-07 LTP-08',
                'LTP-08 LTP-09', 'LTP-09 LTP-10', 'LTSa01 LTSa02', 'LTSa02 LTSa03', 'LTSa03 LTSa04', 'LTSa04 LTSa05',
                'LTSa05 LTSa06', 'LTSa06 LTSa07', 'LTSa07 LTSa08', 'LTSp01 LTSp02', 'LTSp02 LTSp03', 'LTSp03 LTSp04',
                'LTSp04 LTSp05', 'LTSp05 LTSp06', 'LTSp06 LTSp07', 'LTSp07 LTSp08', 'LTSp08 LTSp09', 'LTSp09 LTSp10',
                'LAMY01 LAMY02', 'LAMY02 LAMY03', 'LAMY03 LAMY04', 'LAMY04 LAMY05', 'LAMY05 LAMY06', 'LAMY06 LAMY07',
                'LAMY07 LAMY08', 'LAMY08 LAMY09', 'LAMY09 LAMY10', 'LPHa01 LPHa02', 'LPHa02 LPHa03', 'LPHa03 LPHa04',
                'LPHa04 LPHa05', 'LPHa05 LPHa06', 'LPHa06 LPHa07', 'LPHa07 LPHa08', 'LPHa08 LPHa09', 'LPHa09 LPHa10',
                'LPHa10 LPHa11', 'LPHa11 LPHa12', 'LPHp01 LPHp02', 'LPHp02 LPHp03', 'LPHp03 LPHp04', 'LPHp04 LPHp05',
                'LPHp05 LPHp06', 'LPHp06 LPHp07', 'LPHp07 LPHp08', 'LPHp08 LPHp09', 'LPHp09 LPHp10', 'LPHp10 LPHp11',
                'LPHp11 LPHp12', 'LPHp12 LPHp13', 'LPHp13 LPHp14', 'LTB-01 LTB-02', 'LTB-02 LTB-03', 'LTB-03 LTB-04',
                'LTB-04 LTB-05', 'LTB-05 LTB-06', 'LTB-06 LTB-07', 'LTB-07 LTB-08', 'LTB-08 LTB-09', 'LTB-09 LTB-10',
                'LTB-10 LTB-11', 'LTB-11 LTB-12', 'RFIa01 RFIa02', 'RFIa02 RFIa03', 'RFIa03 RFIa04', 'RFIa04 RFIa05',
                'RFIa05 RFIa06', 'RFIa06 RFIa07', 'RFIa07 RFIa08', 'RFIp01 RFIp02', 'RFIp02 RFIp03', 'RFIp03 RFIp04',
                'RFIp04 RFIp05', 'RFIp05 RFIp06', 'RFIp06 RFIp07', 'RFIp07 RFIp08', 'RFIp08 RFIp09', 'RFIp09 RFIp10',
                'RTP-02 RTP-03', 'RTP-03 RTP-04', 'RTP-04 RTP-05', 'RTP-05 RTP-06', 'RTP-08 RTP-09', 'RTSa03 RTSa04',
                'RTSa04 RTSa05', 'RTSa05 RTSa06', 'RTSa06 RTSa07', 'RTSa07 RTSa08', 'RTSp06 RTSp07', 'RTSp07 RTSp08',
                'RAMY01 RAMY02', 'RAMY02 RAMY03', 'RAMY03 RAMY04', 'RAMY06 RAMY07', 'RAMY09 RAMY10', 'RAMY10 RAMY11',
                'RAMY11 RAMY12', 'RPHa01 RPHa02', 'RPHa02 RPHa03', 'RPHa03 RPHa04', 'RPHa04 RPHa05', 'RPHa05 RPHa06',
                'RPHa06 RPHa07', 'RPHa09 RPHa10', 'RPHa10 RPHa11', 'RPHa11 RPHa12', 'RPHa12 RPHa13', 'RPHa13 RPHa14',
                'RTB-05 RTB-06', 'RTB-06 RTB-07', 'RTB-07 RTB-08', 'RTB-08 RTB-09', 'RTB-09 RTB-10', 'RTB-10 RTB-11',
                'RPHp02 RPHp03', 'RPHp07 RPHp08', 'RPHp08 RPHp09', 'RPHp09 RPHp10', 'RPHp10 RPHp11', 'RPHp11 RPHp12',
                'RPHp12 RPHp13', 'RPHp13 RPHp14']
"""


def plot_brain_heatmap(activation_vector, channel_name, fig=None, ax=None, colorbar=None):
    n_electrodes = len(channel_name)
    normalized_activations = (activation_vector - np.min(activation_vector)) / (
            np.max(activation_vector) - np.min(activation_vector))
    normalized_activations = np.squeeze(normalized_activations)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(30, 20))
    for name, (x, y) in get_pos(4).items():
        channel_list = [ch.split(' ')[0] for ch in channel_name if name in ch]
        channel_fullname_list = [ch for ch in channel_name if name in ch]
        electrode_site_idx = [int(ch.split(' ')[0][-2:]) for ch in channel_name if name in ch]
        # print(f"{name} : {channel_list} number = {len(channel_list)}")
        for idx, ch_name in zip(electrode_site_idx, channel_fullname_list):
            channel_idx = channel_name.index(ch_name)
            x, y = get_pos(idx * 4 + 5)[name]
            ax.scatter(x, y, color='red', s=100)
            # ax.text(x, y, name, ha='center', va='center', fontsize=26)
            activation = normalized_activations[channel_idx]
            color = plt.cm.jet(activation)
            circle = plt.Circle((x, y), 2, color=color)
            ax.add_artist(circle)
            ax.text(x, y, name, ha='center', va='center', fontsize=16, color='white')
    # Colorbar to represent activation levels
    if colorbar is None:
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=ax, orientation='vertical')
        cbar.set_label('Activation Level', size=16)
    else:
        colorbar.update_normal(plt.cm.ScalarMappable(cmap='jet'))
        colorbar.set_label('Activation Level', size=16)

    # Hide axes and set limits
    ax.axis('off')
    return fig, ax


def mesh_properties(mesh_coords):
    """Calculate center and radius of sphere minimally containing a 3-D mesh

    Parameters
    ----------
    mesh_coords : tuple
        3-tuple with x-, y-, and z-coordinates (respectively) of 3-D mesh vertices
    """

    radii = []
    center = []

    for coords in mesh_coords:
        c_max = max(c for c in coords)
        c_min = min(c for c in coords)
        center.append((c_max + c_min) / 2)

        radius = (c_max - c_min) / 2
        radii.append(radius)

    return (center, max(radii))


"""path_fs = "D:\\Navid\\Projects2\\TSNE\\SurfTemplate\\"
file_name = 'BrainMesh_ICBM152.nv'
# Download and prepare dataset from BrainNet repo
# Read the file line by line and split each line
with open(path_fs + file_name, 'r') as f:
    lines = [line.strip().split() for line in f]

# Filter out lines with 3 columns for coords
lines = [line for line in lines if len(line) == 3]
coords_lines = []
triangles_lines = []

for line in lines:
    try:
        # Try converting the first element to an integer
        int(line[0])
        triangles_lines.append(line)
    except ValueError:
        coords_lines.append(line)

index_in_between = len(coords_lines)
print(index_in_between)

coords = np.array(coords_lines, dtype=float)
x, y, z = coords.T

# Filter out lines with 3 columns for triangles
triangles = np.array(triangles_lines, dtype=int)
triangles_zero_offset = triangles - 1
i, j, k = triangles_zero_offset.T

# Generate 3D mesh.  Simply replace with 'fig = go.Figure()' or turn opacity to zero if seeing brain mesh is not
# desired.
fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z,
                                i=i, j=j, k=k,
                                color='lightgray', opacity=0.5, name='', showscale=False, hoverinfo='none')])

# Generate networkx graph and initial 3-D positions using Kamada-Kawai path-length cost-function inside sphere
# containing brain mesh
G = nx.gnp_random_graph(200, 0.02, seed=42)  # Replace G with desired graph here

mesh_coords = (x, y, z)
mesh_center, mesh_radius = mesh_properties(mesh_coords)

scale_factor = 5  # Tune this value by hand to have more/fewer points between the brain hemispheres.
pos_3d = nx.kamada_kawai_layout(G, dim=3, center=mesh_center, scale=scale_factor * mesh_radius)

# Calculate final node positions on brain surface
pos_brain = {}

for node, position in pos_3d.items():
    squared_dist_matrix = np.sum((coords - position) ** 2, axis=1)
    pos_brain[node] = coords[np.argmin(squared_dist_matrix)]

# Prepare networkx graph positions for plotly node and edge traces
nodes_x = [position[0] for position in pos_brain.values()]
nodes_y = [position[1] for position in pos_brain.values()]
nodes_z = [position[2] for position in pos_brain.values()]

edge_x = []
edge_y = []
edge_z = []
for s, t in G.edges():
    edge_x += [nodes_x[s], nodes_x[t]]
    edge_y += [nodes_y[s], nodes_y[t]]
    edge_z += [nodes_z[s], nodes_z[t]]

# Decide some more meaningful logic for coloring certain nodes.  Currently the squared distance from the mesh point at index 42.
node_colors = []
for node in G.nodes():
    if np.sum((pos_brain[node] - coords[42]) ** 2) < 1000:
        node_colors.append('red')
    else:
        node_colors.append('gray')

# Add node plotly trace
fig.add_trace(go.Scatter3d(x=nodes_x, y=nodes_y, z=nodes_z,
                           # text=labels,
                           mode='markers',
                           # hoverinfo='text',
                           name='Nodes',
                           marker=dict(
                               size=5,
                               color=node_colors
                           )
                           ))

# Add edge plotly trace.  Comment out or turn opacity to zero if not desired.
fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z,
                           mode='lines',
                           hoverinfo='none',
                           name='Edges',
                           opacity=0.1,
                           line=dict(color='gray')
                           ))

# Make axes invisible
fig.update_scenes(xaxis_visible=False,
                  yaxis_visible=False,
                  zaxis_visible=False)

# Manually adjust size of figure
fig.update_layout(autosize=False,
                  width=800,
                  height=800)

fig.show()"""
