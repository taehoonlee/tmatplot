import matplotlib as mpl


def size(x):
    mpl.rcParams['figure.dpi'] = x
    mpl.rcParams['savefig.dpi'] = x


def small():
    size(100)


def medium():
    size(200)


def large():
    size(300)


def get_colors(c, rgba=False):
    color_list = [
        # adapted from the parula(8) function in MATLAB
        ['72a2dc', '0cadbb', '43bb97', '8fbf73', 'd0c16a',
         'febf3d', 'ffd200'],
        # adapted from https://www.materialui.co/colors
        ['EF5350', 'EC407A', 'AB47BC', '7E57C2', '5C6BC0',
         '42A5F5', '29B6F6', '26C6DA', '26A69A', '66BB6A',
         '9CCC65', 'D4E157', 'FFEE58', 'FFCA28', 'FFA726',
         'FF7043'],
    ]
    if rgba is True:
        return ['#' + e for e in color_list[c]]
    else:
        return color_list[c]


def colors(c=0):
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', get_colors(c))


large()
colors()

mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['lines.dashed_pattern'] = [6, 6]
mpl.rcParams['lines.dashdot_pattern'] = [3, 5, 1, 5]
mpl.rcParams['lines.dotted_pattern'] = [1, 3]
mpl.rcParams['lines.scale_dashes'] = False

mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.xmargin'] = 0.0

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['image.cmap'] = 'rainbow'

mpl.rcParams['grid.color'] = '#666666'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5

mpl.rcParams['savefig.bbox'] = 'tight'
