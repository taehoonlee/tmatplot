import matplotlib as mpl

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color',
                                             ['72a2dc', '0cadbb', '43bb97',
                                              '8fbf73', 'd0c16a', 'febf3d',
                                              'ffd200'])
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'

mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['lines.dashed_pattern'] = [6, 6]
mpl.rcParams['lines.dashdot_pattern'] = [3, 5, 1, 5]
mpl.rcParams['lines.dotted_pattern'] = [1, 3]
mpl.rcParams['lines.scale_dashes'] = False

mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.xmargin'] = 0.0

mpl.rcParams['grid.color'] = '#666666'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5

mpl.rcParams['image.cmap'] = 'rainbow'
mpl.rcParams['image.interpolation'] = 'nearest'
