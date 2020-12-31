def data(low=True, num=0):
    path = "instances_01_KP"
    low_dim = ['f1_l-d_kp_10_269', 'f2_l-d_kp_20_878', 'f3_l-d_kp_4_20', 'f4_l-d_kp_4_11', 'f5_l-d_kp_15_375', 'f6_l-d_kp_10_60', 'f7_l-d_kp_7_50', 'f8_l-d_kp_23_10000', 'f9_l-d_kp_5_80', 'f10_l-d_kp_20_879']
    large_scale = [
        "knapPI_1_100_1000_1",
        "knapPI_1_200_1000_1",
        "knapPI_1_500_1000_1",
        "knapPI_1_1000_1000_1",
        "knapPI_1_2000_1000_1",
        "knapPI_1_5000_1000_1",
        "knapPI_1_10000_1000_1"]

    try:
        dataset = {}
        i = 0
        if low:
            fp = open(path + '/low-dimensional/'+low_dim[num], "r")
            for row in fp:
                x = row.strip().split(" ")
                dataset[i] = int(x[0]), int(x[1])
                i += 1
        else:
            fp = open(path + '/large_scale/'+large_scale[num], "r")
            for row in fp:
                x = row.strip().split(" ")
                if int(x[0]) != 0:
                    dataset[i] = int(x[0]), int(x[1])
                i += 1
    except IOError as err:
        print("Error:", err)
    finally:
        fp.close()
    return dataset

