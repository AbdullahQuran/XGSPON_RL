LEARNING_PATH = ".\\rl_learning\\"
TEST_PATH = ".\\"
# RUN_COMMAND = "C:\ProgramData\Miniconda3\python ./test.py"
RUN_COMMAND = "C:\\Users\\Ahmed\\Miniconda3\\python ./test.py"

OUTPUT_PATH = TEST_PATH + "output"

topoFilePath = LEARNING_PATH + "three_node.topo"
nodeTypeFilePath = LEARNING_PATH + "three_node.type"
paramFilePath = LEARNING_PATH + "three_node.param"
resultsDir = TEST_PATH + "results\\"
globalsFilePath = '.\\scripts\\params.py'


def writeFile(path, content):
    f = open(path, "w")
    f.write(content)
    f.close()




