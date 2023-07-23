# library imports

# project imports


class FeatureDistribution:
    """
    A class responsible for to represent some given distribution with its properties
    """

    def __init__(self):
        pass


    def sample(self):
        pass


    def sample_col(self,
                   count: int):
        pass


# def create_corpus(iterations, features_num, row_count, f_diff, d_tag_size):
#     anom_algo_name = DBSCANwrapper.NAME
#     performance_metric = None  # InverseEuclideanSim, EuclideanSim()
#     performance_metric_name = None  # performance_metric.NAME
#     performance_metric_attempts = 0  # 2
#
#     meta_data = {"iterations": iterations,
#                  "features_num": features_num,
#                  "row_count": row_count,
#                  "f_diff": f_diff,
#                  "d_tag_size": d_tag_size,
#                  "anom_algo_name": anom_algo_name,
#                  "performance_metric_name": performance_metric_name,
#                  "performance_metric_attempts": performance_metric_attempts,
#                  "cols_dist": {}}
#     folder_path = os.path.join(DATA_FOLDER_PATH,
#                                f"{DBSCANwrapper.NAME}_rc{row_count}_pm{performance_metric_name}")
#     os.makedirs(folder_path, exist_ok=True)
#
#     for iteration in tqdm(range(iterations)):
#         # print(f"generate dataset {iteration}")
#
#         cols_dist_functions = {}
#         for feature in range(features_num):
#             mean = random.uniform(0, 1)
#             std = random.uniform(0, 1)
#             cols_dist_functions[feature] = FeatureDistributionNormal(mean=mean, std=std)
#         meta_data["cols_dist"][f"iter{iteration}"] = cols_dist_functions
#
#         SyntheticDatasetGeneration.generate_one(
#             anomaly_detection_algorithm=DBSCANwrapper(),
#             row_count=row_count,
#             cols_dist_functions=cols_dist_functions,
#             f_diff=f_diff,
#             d_tag_size=d_tag_size,
#             performance_metric=performance_metric,
#             performance_metric_attempts=performance_metric_attempts,
#             save_csv=os.path.join(folder_path, f"synt_iter{iteration}")
#         )
#
#     # save dictionary to person_data.pkl file
#     with open(os.path.join(folder_path, "meta_data.pkl"), 'wb') as fp:
#         pickle.dump(meta_data, fp)
#     return
#
#     # @profile
