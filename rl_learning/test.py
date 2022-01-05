import ModelUtils
import Driver
import Odn
import Globals

links = ModelUtils.load_links("three_node")
print("************************")
print(links)
print("************************")
node_types = ModelUtils.load_node_types("three_node")
print("************************")
print(node_types)
print("************************")
param = ModelUtils.load_param("three_node")
print("************************")
print(param)
print("************************")
oltType = param['OLT']['type']
M = ModelUtils.make_model(links, node_types, param, "three_node")
M.oltType = oltType 
Driver.run_sim(M, t=Globals.SIM_TIME, output_dir="output")
