import main as ravel
from IPython.display import SVG, display

#tick,svg = ravel.start(
#    50,  #size          - размеры карты
#    0.4, #density       - плотность препятствий
#    50   #num_agents    - количество агентов на карте
#)
all=0
for i in range(0, 10):
    tick,svg = ravel.start(50, 0.4, 50)
    all += tick

print(all/10)

#display(SVG(svg))