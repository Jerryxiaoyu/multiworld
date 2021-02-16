import numpy as np
from multiworld.math import Pose
from ..simulation.body import Body
import glob
import os
import math

import random
import pybullet as p

import os, inspect

OBJECTS_DICT = {

#'mustard_bottle', 'pudding_box',  "potted_meat_can"   0.8 "banana"  0.5 "bowl"  "rubiks_cube  gelatin_box
'master_chef':["YCB/002_master_chef_can/master_chef.urdf",(0,0,0), [], 1],
'sugar_box':["YCB/004_sugar_box/sugar_box.urdf",(math.pi,-math.pi/2, 0), [], 1],
#'sugar_box':["YCB/004_sugar_box/sugar_box.urdf",(0,0,0), [], 1],
'tomato_soup_can':["YCB/005_tomato_soup_can/tomato_soup_can.urdf",(0,0,0), [], 1],
'mustard_bottle':["YCB/006_mustard_bottle/mustard_bottle.urdf",  (-math.pi/2, 0,  -math.pi/6), [], 1],
'pudding_box':["YCB/008_pudding_box/pudding_box.urdf",(0,0,0), [], 1],
'potted_meat_can':["YCB/010_potted_meat_can/potted_meat_can.urdf", (0,-math.pi/2,0), [], 1],
'banana':["YCB/011_banana/banana.urdf",(0,0,0), [], 0.6],
'apple':["YCB/013_apple/apple.urdf",(0,0,0), [], 1],
'cracker_box':["YCB/003_cracker_box/cracker_box.urdf", (0,0,0), [], 1],
'gelatin_box':["YCB/009_gelatin_box/gelatin_box.urdf", (0,0,0), [], 1],
'bowl':["YCB/024_bowl/bowl.urdf", (0,0,0), [], 0.5],
'mug':["YCB/025_mug/mug.urdf", (math.pi ,0,0), [], 1],
'plate':["YCB/029_plate/plate.urdf", (0,0,0), [], 1],
'rubiks_cube':["YCB/077_rubiks_cube/rubikes_cube.urdf", (0,0,0), [], 1],
'cups_a':["YCB/065-a_cups/cups_a.urdf", (0,0,0), [], 1],


'b_column': ["objects/blocks/block_column.urdf",(0,0,0), [], 1],


'b_semi_column': ["objects/blocks/block_semi_column.urdf",(0,0,0), [], 1],
'b_cube_m': ["objects/blocks/block_cube_m.urdf",(0,0,0), [], 1],
'b_cube_w': ["objects/blocks/block_cube_w.urdf",(0,0,0), [], 1],
'b_cuboid': ["objects/blocks/block_cuboid.urdf",(0,0,0), [], 1],
'b_cuboid2': ["objects/blocks/block_cuboid2.urdf",(math.pi/2,0,0), [], 1],#big
'b_L1':     ["objects/blocks/block_L1.urdf",(0,0,0), [], 1],
'b_L2':     ["objects/blocks/block_L2.urdf",(0,0,0), [], 1],



'ball_visual': ["objects/balls/ball_visual.urdf",(0,0,0), [], 1],

'box_b':  ['objects/box/box_blue.urdf',(0,0,0), [], 1],

'lshape_1':['Lshapes/auto_gen_objects_14420_5139.sdf',(0,0,0), [], 1],
'Lshape_train' :['Lshapes/train',(0,0,0), [], 1],


'shapenet':['shapenet/02876657',(0,0,0), [], 1],



'004_sugar_box': ["ycb/004_sugar_box/obj.urdf",(0,0,0), [], 0.8],
'007_tuna_fish_can': ["ycb/007_tuna_fish_can/obj.urdf",(0,0,0), [], 1],
'035_power_drill': ["ycb/035_power_drill/obj.urdf",(0,0,0), [], 0.6],
'051_large_clamp': ["ycb/051_large_clamp/obj.urdf",(0,0,0), [], 0.6],


'003_cracker_box': ["ycb/003_cracker_box/obj.urdf",(0,0,0), [], 0.5],
'006_mustard_bottle': ["ycb/006_mustard_bottle/obj.urdf",(0,0,0), [], 0.9],
'008_pudding_box': ["ycb/008_pudding_box/obj.urdf",(0,0,0), [], 1],
'009_gelatin_box': ["ycb/009_gelatin_box/obj.urdf",(0,0,0), [], 1],
'010_potted_meat_can': ["ycb/010_potted_meat_can/obj.urdf",(0,0,0), [], 1],
'011_banana': ["ycb/011_banana/obj.urdf",(0,0,0), [], 0.8],

'071_nine_hole_peg_test': ["ycb/071_nine_hole_peg_test/obj.urdf",(0,0,0), [], 0.6],
'filled_073-a_lego_duplo': ["ycb/filled-073-a_lego_duplo/obj.urdf",(0,0,0), [], 1],#big
'filled-073-b_lego_duplo':     ["ycb/filled-073-b_lego_duplo/obj.urdf",(0,0,0), [], 1],
'filled-073-c_lego_duplo':     ["ycb/filled-073-c_lego_duplo/obj.urdf",(0,0,0), [], 1],
'filled-073-d_lego_duplo': ["ycb/filled-073-d_lego_duplo/obj.urdf",(0,0,0), [], 1],#big
'filled-073-f_lego_duplo':     ["ycb/filled-073-f_lego_duplo/obj.urdf",(0,0,0), [], 1],
'flipped-065-a_cups':     ["ycb/flipped-065-a_cups/obj.urdf",(0,0,0), [], 1],
'flipped-065-b_cups': ["ycb/flipped-065-b_cups/obj.urdf",(0,0,0), [], 1],#big
'flipped-065-c_cups':     ["ycb/flipped-065-c_cups/obj.urdf",(0,0,0), [], 1],
'flipped-065-d_cups':     ["ycb/flipped-065-d_cups/obj.urdf",(0,0,0), [], 1],
'flipped-065-e_cups':     ["ycb/flipped-065-e_cups/obj.urdf",(0,0,0), [], 1],




#shapenet obj
'shapenet_bottle_1':     ["shapenet/02876657/6b8b2cb01c376064c8724d5673a063a6/obj.urdf",(0,0,0), [], 1],
'shapenet_bottle_2':     ["shapenet/02876657/547fa0085800c5c3846564a8a219239b/obj.urdf",(0,0,0), [], 1],

'shapenet_mug_1':     ["shapenet/03797390/599e604a8265cc0a98765d8aa3638e70/obj.urdf",(0,0,0), [], 1],
'shapenet_mug_2':     ["shapenet/03797390/b46e89995f4f9cc5161e440f04bd2a2/obj.urdf",(0,0,0), [], 1],

'shapenet_sofa_1':     ["shapenet/04256520/930873705bff9098e6e46d06d31ee634/obj.urdf",(0,0,0), [], 1],
'shapenet_sofa_2':     ["shapenet/04256520/f094521e8579917eea65c47b660136e7/obj.urdf",(0,0,0), [], 1],

'shapenet_phone_1':     ["shapenet/04401088/611afaaa1671ac8cc56f78d9daf213b/obj.urdf",(0,0,0), [], 1],
'shapenet_phone_2':     ["shapenet/04401088/b8555009f82af5da8c3645155d02fccc/obj.urdf",(0,0,0), [], 1],

}






