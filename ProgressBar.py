"""ProgressBar.py
"""
import sys
from rl_learning import Globals


def setup(env, sim_time):
    """Plots the simulation progress on the terminal
    """

    progress_bar = ProgressBar(env, sim_time)

    return progress_bar


class ProgressBar():
    """Displays simulation progress bar
    """

    def __init__(self, env, sim_time):

        self.env = env
        self.sim_time = sim_time
        self.bar_length = Globals.BAR_LENGTH
        self.delta_t = (sim_time/self.bar_length)-0.01*sim_time

    def display_bar(self):
        """Display progress bar
        """

        print(" [ ", end="")

        while True:

            yield self.env.timeout(self.delta_t)

            percent = self.env.now/self.sim_time
            hashes = '#' * int(round(percent * self.bar_length))
            spaces = ' ' * (self.bar_length - len(hashes))
            sys.stdout.write("\r [{0}] {1}%".format(hashes + spaces,
                                                    int(round(percent*100))))
            sys.stdout.flush()

    def run(self):
        """SimPy process to display progress bar
        """

        self.env.process(self.display_bar())
