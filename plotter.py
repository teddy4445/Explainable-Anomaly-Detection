# library imports
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# project imports
from consts import *


class Plotter:
    """
    A class responsible to generate plots for the project
    """

    # CONSTS #
    SHAPES = ["o", "x", "^", "D", "."]
    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def ds_des_plot():
        # todo: move to here the Data Set plot logic
        pass

    @staticmethod
    def solver_converge(exp_dict_list: list,
                        exp_names: list,
                        save_path: str):
        fig, ax = plt.subplots(figsize=(12, 6))
        for index, exp_dict in enumerate(exp_dict_list):
            ax.plot(np.array(exp_dict.convert_process["time"]),
                    np.array(exp_dict.convert_process["score"]),
                    '{}-'.format(Plotter.SHAPES[index]),
                    label=exp_names[index])

        fig.suptitle(f"Solvers Convergence Over Time",
                     fontsize=20)
        plt.xlabel("Time [sec]",
                   fontsize=16)
        plt.ylabel("AFEX Score",
                   fontsize=16)

        ax.legend()
        plt.savefig(save_path)
        plt.close()
