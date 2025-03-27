import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from diffractio import degrees, mm, plt, sp, um, np
from diffractio.scalar_fields_XY import Scalar_field_XY
from diffractio.utils_drawing import draw_several_fields
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY

num_pixels = 512
z1 = 40
z2= 100

length = 100 * um
x0 = np.linspace(-length / 2, length / 2, num_pixels)
y0 = np.linspace(-length / 2, length / 2, num_pixels)
wavelength = 0.6238 * um

u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.35)

t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
t1.slit(x0=0, size=10 * um, angle=0 * degrees)

u2 = u1 * t1
u3 = u2.RS(z * um, new_field=True)
u4 = u2.RS(z * um, new_field=True)

fields = (u2, u3, u4)
titles = ('mask', 'Variable 1 um', 'Variable 2 um')

draw_several_fields(fields, titles=titles)

axcolor = 'lightgoldenrodyellow'
ax_z1 = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
ax_z2 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

z1_slider = Slider(ax_z1, 'Z Distance 1', 5 * um, 200 * um, valinit=40 * um)
z2_slider = Slider(ax_z2, 'Z Distance 2', 5 * um, 200 * um, valinit=100 * um)

def update(val):
    z_distance1 = z1_slider.val
    z_distance2 = z2_slider.val
    u3 = u2.RS(z1=z_distance1, new_field=True)
    u4 = u2.RS(z2=z_distance2, new_field=True)
    ax.clear()
    draw_several_fields((u2, u3, u4), titles=titles)
    fig.canvas.draw_idle()

z1_slider.on_changed(update)
z2_slider.on_changed(update)

plt.show()