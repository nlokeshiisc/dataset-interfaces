import os
import constants as constants

for shift in [
    # constants.BASE,
    # constants.DUSK,
    constants.FOREST,
    # constants.FOG,
    constants.NIGHT,
    # constants.SNOW,
    constants.RAIN,
    constants.STUDIO,
    # constants.SUNLIGHT,
]:
    cmd = f"python main.py --shift {shift} --gpu 5 >> out.txt"
    print(cmd)
    os.system(cmd)
