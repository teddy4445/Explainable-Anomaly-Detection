# library imports
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random

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
    def score_converge(exp_dict_list: list,
                       exp_names: list,
                       save_path: str):
        fig, ax = plt.subplots(figsize=(12, 6))
        for index, exp_dict in enumerate(exp_dict_list):
            ax.plot(np.array(exp_dict.convert_process["time"]),
                    np.array(exp_dict.convert_process["score"]),
                    '{}-'.format(Plotter.SHAPES[index]),
                    label=exp_names[index])

        fig.suptitle(f"Solvers Convergence Over Time - Score",
                     fontsize=20)
        plt.xlabel("Time [sec]",
                   fontsize=16)
        plt.ylabel("AFEX Score",
                   fontsize=16)

        ax.legend()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def shape_converge(exp_dict_list: list,
                       exp_names: list,
                       save_path: str):
        fig, ax = plt.subplots(figsize=(12, 6))
        for index, exp_dict in enumerate(exp_dict_list):
            rows_num = np.array(exp_dict.convert_process["shape"])[:, 0]
            cols_num = np.array(exp_dict.convert_process["shape"])[:, 1]
            conv_len = len(rows_num)
            if conv_len > 20:
                rows_num = rows_num[sorted(random.sample(range(conv_len), 20))]
                cols_num = cols_num[sorted(random.sample(range(conv_len), 20))]
            ax.plot(rows_num, cols_num,
                    '{}-'.format(Plotter.SHAPES[index]),
                    label=exp_names[index])

        fig.suptitle(f"Solvers Convergence Over Time - Shape",
                     fontsize=20)
        plt.xlabel("Num of Rows",
                   fontsize=16)
        plt.ylabel("Num of Columns",
                   fontsize=16)

        ax.legend()
        plt.savefig(save_path)
        plt.close()
