import Trails
import Writer
import Curves

#for ex_method in [Curves.extrude_linear, Curves.extrude_diffusion, Curves.extrude_nn]:
#for ex_method in [Curves.extrude_nn, Curves.extrude_linear]:
#for ex_method in [Curves.extrude_linear, Curves.extrude_nn_v]:
for ex_method in [Curves.extrude_nn]:
    trail_map = Trails.gen_trail_map(512, extrude_method = ex_method)
    Writer.plot(trail_map)
Writer.show()

