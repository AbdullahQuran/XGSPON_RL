import Globals
class MyClock():
    """This class represents optical distribution Network."""
    def __init__(self, step):
        self.now = 0
        self.step = step
    

    def inc(self, val=None):
        if val != None:
            self.now = self.now + val 
            return 
        self.now = self.now + self.step 

  

        