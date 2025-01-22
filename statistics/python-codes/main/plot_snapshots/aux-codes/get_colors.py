
import matplotlib.pyplot as plt
#==================================================


def get_color_from_int(inp, colormap, vmin=None, vmax=None): 
    norm = plt.Normalize(vmin, vmax)  
    return colormap(norm(inp))

#==========================================
