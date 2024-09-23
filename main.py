from test_pattern_generator import TestPatternGenerator
import matplotlib.pyplot as plt
import numpy as np

dims = [501,501]
# dims=[201,201]

tgen = TestPatternGenerator(dims[0],dims[1])
tgen_r = TestPatternGenerator(dims[0],dims[1], channel="r")
tgen_g = TestPatternGenerator(dims[0],dims[1], channel="g")
tgen_b = TestPatternGenerator(dims[0],dims[1], channel="b")

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
fig.tight_layout()

print("\n= Starting test pattern generation demo =")

print("Generating test pattern 1...")
img = tgen.circular_aperture(rad=round(dims[0]/10))
plt.subplot(331)
plt.imshow(img, cmap="gray")
plt.title("Circular aperture")

print("Generating test pattern 2...")
img = tgen.crosshair()
img += tgen.crosshair(orientation=45)
plt.subplot(332)
plt.imshow(img, cmap="gray")
plt.title("Crosshair")

print("Generating test pattern 3...")
img = tgen.slanted_edge()
plt.subplot(333)
plt.imshow(img, cmap="gray")
plt.title("Slanted edge")

print("Generating test pattern 4...")
img = tgen.dot_grid()
plt.subplot(334)
plt.imshow(img, cmap="gray")
plt.title("Dot grid")

print("Generating test pattern 5...")
img = tgen.generate_testchart(test_chart_type="sharpness")
plt.subplot(335)
plt.imshow(img, cmap="gray")
plt.title("Preset testchart 1")

print("Generating test pattern 6...")
img = tgen.generate_testchart(test_chart_type="displacement")
plt.subplot(336)
plt.imshow(img, cmap="gray")
plt.title("Preset testchart 2")

print("Generating test pattern 7...")
img = tgen.sineplate(binary=True)
plt.subplot(337)
plt.imshow(img, cmap="gray")
plt.title("Binary sine plate")

print("Generating test pattern 8...")
img = tgen.zoneplate()
plt.subplot(338)
plt.imshow(img, cmap="gray")
plt.title("Zone plate")

print("Generating test pattern 9...")
h=dims[0]
w=dims[1]
img = tgen_r.circular_aperture(rad=round(h/50), offset=(-round(h/5),-round(w/2.5)))
img += tgen_b.line_grid2(orientation=45)
img += tgen_g.circular_aperture(rad=round(h/25), offset=(round(h/10), round(w/2.5)))
img += tgen_b.circular_aperture(rad=round(h/50), offset=(round(h/10), round(w/2.5)))
img += tgen_r.circular_aperture(rad=round(h/17), offset=(round(h/20), round(w/2)))
img += tgen_r.circular_aperture(rad=round(h/17), offset=(round(h/20), round(w/2)))
img += tgen_b.dot_grid(rad=round(h/167))
img += tgen_g.square_aperture(px=round(h/25), orientation=45, offset=(-round(h/3.3),-round(w/5)))
img += tgen_r.square_aperture(px=round(h/33), orientation=30, offset=(round(h/5),-round(w/3.3)))
img += tgen_g.square_aperture(px=round(h/33), orientation=45, offset=(round(h/5.1),-round(w/3.4)))
img += tgen_b.square_aperture(px=round(h/33), orientation=55, offset=(round(h/5.2),-round(w/3.5)))
for i, angle in enumerate(np.arange(1,90,13)):
    img += tgen_g.crosshair(orientation=angle+i)
    img += tgen_r.crosshair(orientation=angle+i*5)
# img += tgen.get_test_sprite(scale=round(h/62), angle=45, offset=(-round(h/10),round(h/5)))
# img += tgen.get_test_sprite(scale=round(h/125), offset=(round(h/5),-round(h/3.3)))
plt.subplot(339)
plt.imshow(img)
plt.title("Custom")

for ax in fig.axes:
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

print("All generated.\nShowing plot...")
plt.show()