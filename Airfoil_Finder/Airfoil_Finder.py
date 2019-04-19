import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from collections import deque
import webbrowser
import zipfile


class AirfoilFinder(object):
    def __init__(self, sep_for_the_input_airfoil_data, plot_my_airfoil=False, n_the_most_similar_foils=3):
        """
        :param sep_for_the_input_airfoil_data: seperator for the input airfoil data file. For example, '\t' -> data is seperated by tab || ' ' -> data is seperated by one space, ',' -> seperated by comma(,)
        """
        self.plot_figsize = (16, 5)

        self.my_airfoil = self.get_my_airfoil(sep_for_the_input_airfoil_data, plot_my_airfoil)
        self.airfoil_data_names_from_database = []
        self.process_UIUC_airfoil_database_dir()

        self.dict_airfoil_database, self.dict_airfoil_full_names = self.build_airfoil_database()

        self.n_the_most_similar_foils = n_the_most_similar_foils
        self.dict_mse_scores = {}
        self.find_similar_airfoil_to_my_airfoil_from_database()

        self.find_the_n_most_similar_airfoils_and_save_and_plot()

    def process_UIUC_airfoil_database_dir(self):

        airfoil_data_names_from_database1 = os.listdir('../UIUC_airfoil_database')
        dat_only_list = []

        is_dat = False
        for i in airfoil_data_names_from_database1:
            if ('.dat' in i) or ('.DAT' in i):
                is_dat = True

        for file_name in airfoil_data_names_from_database1:

            if is_dat:
                # .dat 또는 .DAT 파일이 있다면,
                if ('.dat' in file_name) or ('.DAT' in file_name):
                    self.airfoil_data_names_from_database.append(file_name)

            elif os.path.isdir('../UIUC_airfoil_database/UIUC_airfoil_database'):
                # directory 안에 .dat 또는 .DAT 파일이 없는데, 중복된 UIUC_airfoil_database directory 가 있다면,
                airfoil_data_names_from_database2 = os.listdir('../UIUC_airfoil_database/UIUC_airfoil_database')

                is_dat2 = False
                for i in airfoil_data_names_from_database2:
                    if ('.dat' in i) or ('.DAT' in i):
                        is_dat2 = True

                for file_name2 in airfoil_data_names_from_database2:
                    if is_dat2:
                        if ('.dat' in file_name2) or ('.DAT' in file_name2):
                            self.airfoil_data_names_from_database.append(file_name2)
                    else:
                        print("[Error] There is no .dat file")
                        assert False

            elif (len(airfoil_data_names_from_database1) == 1) and ('.zip' in file_name):
                # directory 안에 .dat 또는 .DAT 파일이 없고, 중복된 UIUC_airfoil_database directory 도 없는데, zip 만 있다면,

                # unzip
                path_to_zip_file = '../UIUC_airfoil_database/UIUC_airfoil_database.zip'
                directory_to_extract_to = '../UIUC_airfoil_database/'

                print("* Unzipping the database  ...\n")
                zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
                zip_ref.extractall(directory_to_extract_to)
                zip_ref.close()

                airfoil_data_names_from_database2 = os.listdir('../UIUC_airfoil_database')
                for file_name3 in airfoil_data_names_from_database2:
                    if ('.dat' in file_name3) or ('.DAT' in file_name3):
                        self.airfoil_data_names_from_database.append(file_name3)

            else:
                print("[Error] Please check the 'UIUC_airfoil_database' directory.")
                assert False

        return dat_only_list

    def get_my_airfoil(self, sep, plot):
        """
        :param sep: seperator for the data file. For example, '\t' -> data is seperated by tab || ' ' -> data is seperated by one space
        :param plot:
        :return:
        """
        listdir_ = os.listdir('./my_airfoil')
        if len(listdir_) != 1:
            print("======================================")
            print("[Error] Please place one airfoil only.")
            print("======================================")
            assert False

        df_my_airfoil = pd.read_csv("./my_airfoil/{}".format(listdir_[0]), header=None, sep=sep)
        my_airfoil = df_my_airfoil.values

        if plot:
            plt.figure(figsize=self.plot_figsize)
            plt.title("Airfoil Shape Given - {}".format(listdir_[0]))
            plt.plot(my_airfoil[:, 0], my_airfoil[:, 1])
            plt.grid()
            plt.show()

        return my_airfoil

    def build_airfoil_database(self):
        print("* Building the database of the UIUC airfoils... *")
        number_of_airfoil_names_from_database = len(self.airfoil_data_names_from_database)

        dict_airfoil_database = {}
        dict_airfoil_full_names = {}

        for idx, airfoil_data_name in enumerate(self.airfoil_data_names_from_database):
            print("* Building the database of the UIUC airfoils... || Progress: {}/{}".format(idx + 1,
                                                                                              number_of_airfoil_names_from_database))

            coordinates = []
            airfoil_full_name = ''
            with open('../UIUC_airfoil_database/{}'.format(airfoil_data_name), 'r') as f:
                idx_to_find_airfoil_name = 0

                while True:
                    line = f.readline()
                    if idx_to_find_airfoil_name == 0:
                        airfoil_full_name = '{}'.format(line)
                        # print("airfoil_full_name: ", airfoil_full_name)

                    line_striped = line.strip()
                    line_striped_split = line_striped.split(' ')

                    line_striped_split_sorted = []
                    for i in line_striped_split:
                        if i != '':
                            line_striped_split_sorted.append(i)

                    is_tab = False
                    for i in line_striped_split_sorted:
                        if '\t' in i:
                            is_tab = True

                    if is_tab:
                        line_striped_split_sorted2 = []
                        for i in line_striped_split_sorted:
                            line_striped_split_sorted2.append(i.split('\t'))

                        line_striped_split_sorted = line_striped_split_sorted2[0]

                    if len(line_striped_split_sorted) == 2:
                        try:
                            X_coord, Y_coord = line_striped_split_sorted

                            try:
                                X_coord, Y_coord = eval(X_coord), eval(Y_coord)

                                if (X_coord <= 1.0) and (Y_coord <= 1.0):
                                    if (type(X_coord) != int) and (type(Y_coord) != int):
                                        coordinates.append([X_coord, Y_coord])
                            except (SyntaxError, NameError) as e:
                                pass

                        except ValueError:
                            pass

                    idx_to_find_airfoil_name += 1

                    if not line:
                        break

            coordinates_arr = np.array(coordinates)

            dict_airfoil_database['{}'.format(airfoil_data_name)] = coordinates_arr
            dict_airfoil_full_names['{}'.format(airfoil_data_name)] = airfoil_full_name

            # for the check
            # df_temp = pd.DataFrame(coordinates_arr)
            # df_temp.to_csv('./rearranged_UIUC_airfoil_database/{}'.format(airfoil_data_name), header=None, index=None)

        return dict_airfoil_database, dict_airfoil_full_names

    def sort_into_upper_lower_sides(self, df, tolerance_multiplier=1.2, back_part_ratio=0.90):
        df = df.drop_duplicates()

        df_sorted = df.sort_values(by=[0, 1])
        df_sorted = df_sorted.values

        # print("df_sorted: ", df_sorted)

        upper_side = []
        lower_side = []
        y_gap_hist = deque(maxlen=2)
        appending_status = 0

        for idx, x_y in enumerate(df_sorted):

            if idx == 0:
                upper_side.append(x_y)
                lower_side.append(x_y)
            else:
                gap_btn_previous_and_current_data = df_sorted[:, 1][idx] - df_sorted[:, 1][idx - 1]
                y_gap_hist.append(gap_btn_previous_and_current_data)

                if idx == 1:
                    if np.sign(gap_btn_previous_and_current_data) == 1:
                        upper_side.append(x_y)
                        appending_status = 1
                    elif np.sign(gap_btn_previous_and_current_data) == -1:
                        lower_side.append(x_y)
                        appending_status = -1
                    elif np.sign(gap_btn_previous_and_current_data) == 0:
                        appending_status = 0
                else:

                    if idx != (df_sorted.shape[0] - 1):

                        # print("df_sorted[:, 0][idx]: ", df_sorted[:, 0][idx])
                        # print("df_sorted[:, 1][idx]: ", df_sorted[:, 1][idx])
                        # print("np.abs(y_gap_hist[1]): ", np.abs(y_gap_hist[1]))
                        # print("np.abs(y_gap_hist[0]) / 5: ", np.abs(y_gap_hist[0]) / 5)

                        # print("np.sign(y_gap_hist[0]: ", np.sign(y_gap_hist[0]))
                        # print("np.sign(y_gap_hist[1]: ", np.sign(y_gap_hist[1]))

                        if np.abs(y_gap_hist[1]) >= (np.abs(y_gap_hist[0])) / 5 + 4e-3 * (
                                np.max(df_sorted[:, 1]) / 0.1):  # /5 originally
                            # sign(gap) 을 그대로 반영함.
                            if np.sign(y_gap_hist[1]) == 1:
                                # print("upper_side.append(x_y)")
                                # print('\n')
                                upper_side.append(x_y)
                                appending_status = 1

                            elif np.sign(y_gap_hist[1]) == -1:
                                # print("lower_side.append(x_y)")
                                # print('\n')
                                lower_side.append(x_y)
                                appending_status = -1

                            elif np.sign(y_gap_hist[1]) == 0:
                                upper_side.append(x_y)
                                lower_side.append(xy_y)
                                appending_status = 0
                        else:
                            # print("reverse.")
                            # print('\n')
                            # sign(gap) 을 반영하지않고, appending_status 를 반영함.
                            if appending_status == 1:
                                upper_side.append(x_y)
                                # print("idx:{} || appending_status == 1".format(idx))
                            elif appending_status == -1:
                                lower_side.append(x_y)
                                # print("idx:{} || appending_status == -1".format(idx))
                            elif appending_status == 0:
                                upper_side.append(x_y)
                                lower_side.append(x_y)
                                # print("idx:{} || appending_status == 0".format(idx))
                                # print("probably it's not gonna happen, I assume.")
                    else:
                        upper_side.append(x_y)
                        lower_side.append(x_y)

        upper_side, lower_side = np.array(upper_side), np.array(lower_side)

        SECOND_STAGE = True
        if SECOND_STAGE:
            upper_side_x = upper_side[:, 0]
            upper_side_y = upper_side[:, 1]

            lower_side_x = lower_side[:, 0]
            lower_side_y = lower_side[:, 1]

            upper_side2 = []
            lower_side2 = []

            upper_side2.append(upper_side[0])
            idx = 1
            should_be_dumped_to_lower_side_list = []

            # x축값 0.8 이후에 있는 데이터 개수가 작으면, 거의 직선일 확률이 높으므로, 강제로 back_part_ratio 를 0.8 로 set 한다.
            upper_side_x_at_0_8_idx, _ = self.find_closest_val_idx(0.8, upper_side_x)
            if len(upper_side_x[upper_side_x_at_0_8_idx:]) <= 10 - (0.8 * 10) + 3:
                back_part_ratio = 0.8

            while True:
                if idx == len(upper_side_x) - 1:
                    break

                if 1 <= idx <= 3:
                    # safe start
                    upper_side2.append(upper_side[idx])


                else:
                    if upper_side_x[idx] < back_part_ratio:
                        upper_side_y_distance_left_left = upper_side_y[idx - 1] - upper_side_y[idx - 2]
                        upper_side_y_distance_left = upper_side_y[idx] - upper_side_y[idx - 1]
                        upper_side_y_distance_right = upper_side_y[idx + 1] - upper_side_y[idx]

                        tolerance = upper_side_y_distance_left_left * tolerance_multiplier
                        tolerance = np.abs(tolerance) + 1.5e-2
                        if np.sign(upper_side_y_distance_left) == np.sign(upper_side_y_distance_right):
                            # print("(1)upper_side_x[idx]: ", upper_side_x[idx])
                            # print("(1)upper_side_y_distance_left: ", upper_side_y_distance_left)
                            # print("(1)upper_side_y_distance_right: ", upper_side_y_distance_right, '\n')
                            upper_side2.append(upper_side[idx])
                        elif (np.sign(upper_side_y_distance_left) != np.sign(upper_side_y_distance_right)) and (
                                np.abs(upper_side_y_distance_left) < tolerance):
                            # print("(2)upper_side_x[idx]: ", upper_side_x[idx])
                            # print("(2)upper_side_y_distance_left: ", upper_side_y_distance_left)
                            # print("(2)upper_side_y_distance_right: ", upper_side_y_distance_right)
                            # print("(2)np.abs(upper_side_y_distance_left): ", np.abs(upper_side_y_distance_left))
                            # print("(2)tolerance: ", tolerance, '\n')
                            # 왼쪽 오른쪽 부호는 다르지만, 그 차이가 그리 크지 않다면,
                            upper_side2.append(upper_side[idx])
                        else:
                            # print("(3)upper_side_x[idx]: ", upper_side_x[idx])
                            # print("(3)upper_side_y_distance_left: ", upper_side_y_distance_left)
                            # print("(3)upper_side_y_distance_right: ", upper_side_y_distance_right)
                            # print("(3)np.abs(upper_side_y_distance_left): ", np.abs(upper_side_y_distance_left))
                            # print("(3)tolerance: ", tolerance, '\n')
                            should_be_dumped_to_lower_side_list.append(upper_side[idx])
                    else:
                        # x축 90 퍼센트 이상에서는 imaginary line 을 그려서 sort 한다.
                        upper_side_x_at_backPart_idx, _ = self.find_closest_val_idx(back_part_ratio, upper_side_x)
                        upper_side_y_at_backPart = upper_side_y[upper_side_x_at_backPart_idx]
                        lower_side_y_at_backPart_idx, _ = self.find_closest_val_idx(back_part_ratio, lower_side_x)
                        lower_side_y_at_backPart = lower_side_y[lower_side_y_at_backPart_idx]

                        # upper_side_x_at_1_0_idx, _ = find_closest_val_idx(1.0, upper_side_x)
                        upper_side_y_at_1_0 = upper_side_y[-1]
                        # lower_side_x_at_1_0_idx, _ = find_closest_val_idx(1.0, lower_side_x)
                        lower_side_y_at_1_0 = lower_side_y[-1]

                        x_upper_val = upper_side_x[idx]

                        # if len(upper_side_x[upper_side_x_at_backPart_idx:]) >= (10 - back_part_ratio*10) + 1:

                        criteria_val = (1.0 - x_upper_val) / (1.0 - back_part_ratio) * (
                                upper_side_y_at_backPart + lower_side_y_at_backPart) / 2 + (
                                               x_upper_val - back_part_ratio) / (1.0 - back_part_ratio) * (
                                               upper_side_y_at_1_0 + lower_side_y_at_1_0) / 2

                        if upper_side_y[idx] >= criteria_val:
                            upper_side2.append(upper_side[idx])
                        else:
                            should_be_dumped_to_lower_side_list.append(upper_side[idx])

                idx += 1
            upper_side2.append(upper_side[-1])

            lower_side2.append(lower_side[0])
            idx = 1
            should_be_dumped_to_upper_side_list = []
            while True:
                if idx == len(lower_side_x) - 1:
                    break

                if 1 <= idx <= 3:
                    # safe start
                    lower_side2.append(lower_side[idx])
                else:
                    if lower_side_x[idx] < back_part_ratio:
                        lower_side_y_distance_left_left = lower_side_y[idx - 1] - lower_side_y[idx - 2]
                        lower_side_y_distance_left = lower_side_y[idx] - lower_side_y[idx - 1]
                        lower_side_y_distance_right = lower_side_y[idx + 1] - lower_side_y[idx]

                        tolerance = lower_side_y_distance_left_left * tolerance_multiplier
                        tolerance = np.abs(tolerance) + 1.5e-2
                        if np.sign(lower_side_y_distance_left) == np.sign(lower_side_y_distance_right):
                            # print("(1)lower_side_x[idx]: ", lower_side_x[idx])
                            # print("(1)lower_side_y_distance_left: ", lower_side_y_distance_left)
                            # print("(1)lower_side_y_distance_right: ", lower_side_y_distance_right, '\n')
                            lower_side2.append(lower_side[idx])
                        elif np.sign(lower_side_y_distance_left) != np.sign(lower_side_y_distance_right) and (
                                np.abs(lower_side_y_distance_left) < tolerance):
                            # print("(2)lower_side_x[idx]: ", lower_side_x[idx])
                            # print("(2)lower_side_y_distance_left: ", lower_side_y_distance_left)
                            # print("(2)lower_side_y_distance_right: ", lower_side_y_distance_right)
                            # print("(2)np.abs(lower_side_y_distance_left): ", np.abs(lower_side_y_distance_left))
                            # print("(2)tolerance: ", tolerance, '\n')
                            # 왼쪽 오른쪽 부호는 다르지만, 그 차이가 그리 크지 않다면,
                            lower_side2.append(lower_side[idx])
                        else:
                            # print("(3)lower_side_x[idx]: ", lower_side_x[idx])
                            # print("(3)lower_side_y_distance_left: ", lower_side_y_distance_left)
                            # print("(3)lower_side_y_distance_right: ", lower_side_y_distance_right)
                            # print("(3)np.abs(lower_side_y_distance_left): ", np.abs(lower_side_y_distance_left))
                            # print("(3)tolerance: ", tolerance, '\n')
                            should_be_dumped_to_upper_side_list.append(lower_side[idx])
                    else:
                        # x축 90 퍼센트 이상에서는 imaginary line 을 그려서 sort 한다.
                        upper_side_x_at_backPart_idx, _ = self.find_closest_val_idx(back_part_ratio, upper_side_x)
                        upper_side_y_at_backPart = upper_side_y[upper_side_x_at_backPart_idx]
                        lower_side_y_at_backPart_idx, _ = self.find_closest_val_idx(back_part_ratio, lower_side_x)
                        lower_side_y_at_backPart = lower_side_y[lower_side_y_at_backPart_idx]

                        # upper_side_x_at_1_0_idx, _ = find_closest_val_idx(1.0, upper_side_x)
                        upper_side_y_at_1_0 = upper_side_y[-1]
                        # lower_side_x_at_1_0_idx, _ = find_closest_val_idx(1.0, lower_side_x)
                        lower_side_y_at_1_0 = lower_side_y[-1]

                        x_lower_val = lower_side_x[idx]

                        criteria_val = (1.0 - x_lower_val) / (1.0 - back_part_ratio) * (
                                upper_side_y_at_backPart + lower_side_y_at_backPart) / 2 + (
                                               x_lower_val - back_part_ratio) / (1.0 - back_part_ratio) * (
                                               upper_side_y_at_1_0 + lower_side_y_at_1_0) / 2

                        if lower_side_y[idx] <= criteria_val:
                            lower_side2.append(lower_side[idx])
                        elif (lower_side_y[idx] <= criteria_val) and ():
                            should_be_dumped_to_upper_side_list.append(lower_side[idx])

                idx += 1
            lower_side2.append(lower_side[-1])

            # to array
            upper_side2, lower_side2 = np.array(upper_side2), np.array(lower_side2)

            should_be_dumped_to_lower_side_list = np.array(should_be_dumped_to_lower_side_list)
            should_be_dumped_to_upper_side_list = np.array(should_be_dumped_to_upper_side_list)

            # fill out the should_be_dumped_to_.. list
            try:
                X_should_be_dumped_to_lower_side_list = should_be_dumped_to_lower_side_list[:, 0]
                Y_should_be_dumped_to_lower_side_list = should_be_dumped_to_lower_side_list[:, 1]
            except IndexError:
                X_should_be_dumped_to_lower_side_list = []
                Y_should_be_dumped_to_lower_side_list = []

            try:
                X_should_be_dumped_to_upper_side_list = should_be_dumped_to_upper_side_list[:, 0]
                Y_should_be_dumped_to_upper_side_list = should_be_dumped_to_upper_side_list[:, 1]
            except IndexError:
                X_should_be_dumped_to_upper_side_list = []
                Y_should_be_dumped_to_upper_side_list = []

            for X_should_be_dumped_to_lower_side, Y_should_be_dumped_to_lower_side in zip(
                    X_should_be_dumped_to_lower_side_list, Y_should_be_dumped_to_lower_side_list):
                for i, low_val in enumerate(lower_side2):
                    x_low_val = low_val[0]
                    y_low_val = low_val[1]

                    if (X_should_be_dumped_to_lower_side - x_low_val) < 0:
                        lower_side2 = np.insert(lower_side2, i,
                                                [X_should_be_dumped_to_lower_side, Y_should_be_dumped_to_lower_side],
                                                axis=0)
                        break

            for X_should_be_dumped_to_upper_side, Y_should_be_dumped_to_upper_side in zip(
                    X_should_be_dumped_to_upper_side_list, Y_should_be_dumped_to_upper_side_list):
                for i, upper_val in enumerate(upper_side2):
                    x_upper_val = upper_val[0]
                    y_upper_val = upper_val[1]

                    if (X_should_be_dumped_to_upper_side - x_upper_val) < 0:
                        upper_side2 = np.insert(upper_side2, i,
                                                [X_should_be_dumped_to_upper_side, Y_should_be_dumped_to_upper_side],
                                                axis=0)
                        break

            # update
            upper_side = upper_side2
            lower_side = lower_side2

        return upper_side, lower_side

    def find_similar_airfoil_to_my_airfoil_from_database(self):
        num_airfoils_in_database = len(self.dict_airfoil_database.keys())
        dict_airfoil_database_items_list = list(self.dict_airfoil_database.items())

        upper_side_my_airfoil, lower_side_my_airfoil = self.sort_into_upper_lower_sides(
            df=pd.DataFrame(self.my_airfoil))

        # save the plot - test
        # plt.plot(upper_side_my_airfoil[:, 0], upper_side_my_airfoil[:, 1])
        # plt.plot(lower_side_my_airfoil[:, 0], lower_side_my_airfoil[:, 1])
        # plt.savefig('./upper_lower_sorted_imgs-test/my_airfoil.png')
        # plt.cla()

        for idx, foil_data in enumerate(dict_airfoil_database_items_list):
            airfoil_data_name = foil_data[0]
            airfoil_data_arr = foil_data[1]

            print("* Finding the {} most similar airfoils from the UIUC airfoil database... || Progress:{}/{}".format(
                self.n_the_most_similar_foils, idx + 1, num_airfoils_in_database))

            # size comparison to decide what are the smaller and bigger data
            my_airfoil_size = self.my_airfoil.shape[0]
            airfoil_data_arr_size = airfoil_data_arr.shape[0]

            # sort into the upper and lower sides and stack them - reformatting
            if 'ah7476' in airfoil_data_name:
                upper_side_airfoil_data_arr, lower_side_airfoil_data_arr = self.sort_into_upper_lower_sides(
                    df=pd.DataFrame(airfoil_data_arr), back_part_ratio=0.79)
            else:
                try:
                    upper_side_airfoil_data_arr, lower_side_airfoil_data_arr = self.sort_into_upper_lower_sides(
                        df=pd.DataFrame(airfoil_data_arr))
                except IndexError:
                    continue
            upper_side_my_airfoil, lower_side_my_airfoil = upper_side_my_airfoil, lower_side_my_airfoil

            reformatted_airfoil_data_arr = np.vstack((upper_side_airfoil_data_arr, lower_side_airfoil_data_arr))
            reformatted_my_airfoil = np.vstack((upper_side_my_airfoil, lower_side_my_airfoil))

            # plt.plot(reformatted_my_airfoil[:, 0], reformatted_my_airfoil[:, 1], '-o')  # my_airfoil
            # plt.grid()
            # plt.show()

            # save the plot - test
            # print("airfoil_data_name: ", airfoil_data_name)
            # plt.plot(upper_side_airfoil_data_arr[:, 0], upper_side_airfoil_data_arr[:, 1])
            # plt.plot(lower_side_airfoil_data_arr[:, 0], lower_side_airfoil_data_arr[:, 1])
            # plt.savefig('./upper_lower_sorted_imgs-test/{}.png'.format(airfoil_data_name.split('.')[0]))
            # plt.cla()
            # print('\n')

            upper_side_smaller_data, lower_side_smaller_data = 0, 0  # initialized
            upper_side_bigger_data, lower_side_bigger_data = 0, 0  # initialized

            if my_airfoil_size >= airfoil_data_arr_size:
                upper_side_bigger_data, lower_side_bigger_data = upper_side_my_airfoil, lower_side_my_airfoil
                upper_side_smaller_data, lower_side_smaller_data = upper_side_airfoil_data_arr, lower_side_airfoil_data_arr
            elif my_airfoil_size < airfoil_data_arr_size:
                upper_side_bigger_data, lower_side_bigger_data = upper_side_airfoil_data_arr, lower_side_airfoil_data_arr
                upper_side_smaller_data, lower_side_smaller_data = upper_side_my_airfoil, lower_side_my_airfoil

            # fit the bigger data to the format of the smaller data
            # bigger_data_newX, bigger_data_newY = self.sort_by_linear_interpolation(smaller_data=smaller_data, bigger_data=bigger_data)
            upper_side_bigger_data_newX, upper_side_bigger_data_newY = self.sort_by_linear_interpolation(
                smaller_data=upper_side_smaller_data, bigger_data=upper_side_bigger_data)
            lower_side_bigger_data_newX, lower_side_bigger_data_newY = self.sort_by_linear_interpolation(
                smaller_data=lower_side_smaller_data, bigger_data=lower_side_bigger_data)

            upper_new_bigger_data = np.vstack((upper_side_bigger_data_newX, upper_side_bigger_data_newY)).T
            lower_new_bigger_data = np.vstack((lower_side_bigger_data_newX, lower_side_bigger_data_newY)).T

            new_bigger_data = np.vstack((upper_new_bigger_data, lower_new_bigger_data))
            smaller_data = np.vstack((upper_side_smaller_data, lower_side_smaller_data))

            #print("new_bigger_data: ", new_bigger_data)
            #print("smaller_data: ", smaller_data)

            #plt.plot(new_bigger_data[:, 0], new_bigger_data[:, 1], '-o')  # my_airfoil
            #plt.plot(smaller_data[:, 0], smaller_data[:, 1], '-o')
            #plt.grid()
            #plt.ylim(-0.04, 0.1)
            #plt.show()

            # get the mse(mean squared error) score
            self.dict_mse_scores[airfoil_data_name] = np.mean((new_bigger_data - smaller_data) ** 2)

    def find_closest_val_idx(self, val, look_up_list):
        gaps = []
        for i in look_up_list:
            gaps.append(val - i)

        closest_val_idx = np.argmin(np.abs(gaps))
        closest_val_gap = gaps[closest_val_idx]

        return closest_val_idx, closest_val_gap

    def sort_by_linear_interpolation(self, smaller_data, bigger_data):
        new_X_coords = []
        new_Y_coords = []

        smaller_data_X = smaller_data[:, 0]
        smaller_data_Y = smaller_data[:, 1]

        bigger_data_X = bigger_data[:, 0]
        bigger_data_Y = bigger_data[:, 1]

        # bigger_data 가 smaller_data 에 fit 하기위해 모든 closest_val_idx & gap 을 찾는다.
        # smaller_data 의 데이터에서 매칭되는 데이터를 bigger_data 에서 lookup 한다.
        for s_data_X in smaller_data_X:
            closest_valX_idx, closest_valX_gap = self.find_closest_val_idx(s_data_X, bigger_data_X)

            if closest_valX_gap == 0:
                new_X_coords.append(bigger_data_X[closest_valX_idx])
                new_Y_coords.append(bigger_data_Y[closest_valX_idx])
            elif closest_valX_gap > 0:
                new_X_coords.append(bigger_data_X[closest_valX_idx] + closest_valX_gap)

                Xgap2 = np.abs(bigger_data_X[closest_valX_idx + 1] - bigger_data_X[closest_valX_idx])
                interpolated_Y_val = (Xgap2 - closest_valX_gap) / Xgap2 * bigger_data_Y[
                    closest_valX_idx] + closest_valX_gap / Xgap2 * bigger_data_Y[closest_valX_idx + 1]
                new_Y_coords.append(interpolated_Y_val)
            elif closest_valX_gap < 0:
                new_X_coords.append(bigger_data_X[closest_valX_idx] + closest_valX_gap)

                Xgap2 = np.abs(bigger_data_X[closest_valX_idx] - bigger_data_X[closest_valX_idx - 1])
                interpolated_Y_val = (Xgap2 - (Xgap2 + closest_valX_gap)) / Xgap2 * bigger_data_Y[
                    closest_valX_idx - 1] + (Xgap2 + closest_valX_gap) / Xgap2 * bigger_data_Y[closest_valX_idx]
                new_Y_coords.append(interpolated_Y_val)

        return new_X_coords, new_Y_coords

    def find_the_n_most_similar_airfoils_and_save_and_plot(self):
        keys = list(self.dict_mse_scores.keys())
        values = list(self.dict_mse_scores.values())

        df_keys_values = pd.DataFrame([keys, values]).T
        df_keys_values.columns = ['foil_names', 'mse_scores']

        df_sorted = df_keys_values.sort_values(by='mse_scores')
        df_the_n_most_similar_foils = df_sorted.iloc[:self.n_the_most_similar_foils, :self.n_the_most_similar_foils]

        the_n_most_similar_foil_data_names = df_the_n_most_similar_foils['foil_names'].tolist()

        the_n_most_similar_foil_full_names = []
        for data_name in the_n_most_similar_foil_data_names:
            the_n_most_similar_foil_full_names.append(self.dict_airfoil_full_names[data_name])

        # make the directory of results if that doesn't exist yet.
        isdir_ = os.path.isdir('./results')
        if not isdir_:
            os.mkdir('./results')

        # save the names in the notepad.
        print("\nSave the result in notepad...")
        f = open('./results/the_n_most_similar_foil_names.txt', 'w')
        f.write('* The {} most similar airfoils found from the UIUC airfoil database\n'.format(
            self.n_the_most_similar_foils))
        f.write('- UIUC airfoil database link: https://m-selig.ae.illinois.edu/ads/coord_database.html#M\n')
        f.write('\n')
        f.write('* Ranking, DataName, FullName *\n')
        f.write('======================================================\n')
        idx = 1
        for data_name, full_name in zip(the_n_most_similar_foil_data_names, the_n_most_similar_foil_full_names):
            f.write('Ranking-{}. {}, {}\n'.format(idx, data_name, full_name))
            idx += 1
        f.write('======================================================\n')

        f.write('\n\n\n')
        f.write('Made by Daesoo Lee(이대수), Masters, Korea Maritime and Ocean University\n')
        f.write('e-mail : daesoolee@kmou.ac.kr\n')
        f.write('first made on 04/14/2019\n')
        f.close()

        dirname = os.path.dirname(__file__)
        #os.system(os.path.join(dirname, 'results/the_n_most_similar_foil_names.txt'))
        webbrowser.open(os.path.join(dirname, 'results/the_n_most_similar_foil_names.txt'))

        # plot
        print("\nPlot the result...")
        print("\n* Results are saved in the folder named 'results'. *")
        legends_list = []
        plt.figure(figsize=self.plot_figsize)
        plt.title("Airfoils Found")
        plt.plot(self.my_airfoil[:, 0], self.my_airfoil[:, 1])
        legends_list.append('my_air_foil')

        idx = 1
        for data_name, full_name in zip(the_n_most_similar_foil_data_names, the_n_most_similar_foil_full_names):
            airfoil_data_arr = self.dict_airfoil_database[data_name]
            plt.plot(airfoil_data_arr[:, 0], airfoil_data_arr[:, 1])
            legends_list.append("Ranking-{}, {}".format(idx, full_name))
            idx += 1

        plt.legend(legends_list)
        plt.grid()
        plt.savefig('./results/the_n_most_similar_foils.png')
        plt.show()


if __name__ == "__main__":
    finder = AirfoilFinder(sep_for_the_input_airfoil_data='\t')

