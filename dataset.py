def data(num=0, low=True):
    """
    :param low:(default=True) choose low dimensional instance, False coose large scale dimensional istance
    :param num: number (low: 1 to 10, large: 1 to 21) of the chosen instance
    :return: dataset of items fro testing Knapsack PSO
    """
    path = "instances_01_KP"
    low_dim = ['f1_l-d_kp_10_269', 'f2_l-d_kp_20_878', 'f3_l-d_kp_4_20', 'f4_l-d_kp_4_11', 'f5_l-d_kp_15_375',
               'f6_l-d_kp_10_60', 'f7_l-d_kp_7_50', 'f8_l-d_kp_23_10000', 'f9_l-d_kp_5_80', 'f10_l-d_kp_20_879',
               "f11_l-d_kp_40_990"]
    large_scale = [
        "knapPI_1_100_1000_1",  # Uncorrelated data instances
        "knapPI_1_200_1000_1",
        "knapPI_1_500_1000_1",
        "knapPI_1_1000_1000_1",
        "knapPI_1_2000_1000_1",
        "knapPI_1_5000_1000_1",
        "knapPI_1_10000_1000_1",
        "knapPI_2_100_1000_1",  # Weakly correlated instances
        "knapPI_2_200_1000_1",
        "knapPI_2_500_1000_1",
        "knapPI_2_1000_1000_1",
        "knapPI_2_2000_1000_1",
        "knapPI_2_5000_1000_1",
        "knapPI_2_10000_1000_1",
        "knapPI_3_100_1000_1",  # Strongly correlated instances
        "knapPI_3_200_1000_1",
        "knapPI_3_500_1000_1",
        "knapPI_3_1000_1000_1",
        "knapPI_3_2000_1000_1",
        "knapPI_3_5000_1000_1",
        "knapPI_3_10000_1000_1"]
    opt_path = "-optimum/"

    try:
        dataset = {}
        i = 0
        if low:
            fp = open(path + '/low-dimensional/' + low_dim[num], "r")
            for row in fp:
                x = row.strip().split(" ")
                try:
                    dataset[i] = int(x[0]), int(x[1])
                except ValueError:
                    dataset[i] = float(x[0]), float(x[1])
                i += 1
            with open(path + '/low-dimensional' + opt_path + low_dim[num], "r") as fp_opt:
                first_line = fp_opt.readline()
                try:
                    opt = int(first_line.strip())
                except ValueError:
                    opt = float(first_line.strip())
                dataset[i] = "opt", opt
        else:
            fp = open(path + '/large_scale/' + large_scale[num], "r")
            for row in fp:
                x = row.strip().split(" ")
                if int(x[0]) != 0:
                    dataset[i] = int(x[0]), int(x[1])
                else:
                    with open(path + '/large_scale' + opt_path + large_scale[num], "r") as fp_opt:
                        first_line = fp_opt.readline()
                        try:
                            opt = int(first_line.strip())
                        except ValueError:
                            opt = float(first_line.strip())
                        dataset[i] = "opt", opt
                i += 1
    except IOError as err:
        print("Error:", err)
    finally:
        fp.close()
        fp_opt.close()
    return dataset
