import Trails
import Writer
import Curves
import Noise

for ex_method in [Curves.extrude_nn]:
    trail_map, curve_pts = Trails.gen_trail_map(512, extrude_method = ex_method)
    Writer.plot(trail_map)
    Writer.save_trail(trail_map, 65000, "ridges")
    Writer.save_trees(curve_pts, "ridges")
Writer.show()

"""
img = Noise.GradientNoise(512, 10)
cv2.imshow("hi", img)
cv2.waitKey(0)
"""
