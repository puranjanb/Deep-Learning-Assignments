import numpy as np
import matplotlib.pyplot as plt
import pattern
import src_to_implement.generator as gen
c = pattern.Checker(12, 3)
c.draw()
c.show()
ci = pattern.Circle(1024, 200, (512, 256))
ci.draw()
ci.show()
s = pattern.Spectrum(100)
s.draw()
s.show()

ig = gen.ImageGenerator('./exercise_data/', './Labels.json', 10, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
ig.show()
