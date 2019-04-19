'''
* 웹크롤링으로 UIUC 에어포일 데이터베이스 웹페이지 에서, 에어포일 데이터 긁어오는 실행파일 *
- UIUC Airfoil Database: https://m-selig.ae.illinois.edu/ads/coord_database.html#M

# Made by Daesoo Lee (이대수), Masters, Korea Maritime and Ocean University.
# e-mail : daesoolee@kmou.ac.kr
# first made on 04/13/2019

# 사용법:
1. 위의 UIUC airfoil Database 에 접속한다.
2. 1500여개의 .dat 또는 .DAT 파일들이 있는데, 웹크롤링을 "처음 시작할 데이터파일"과 "마지막 데이터파일"을 선정한다.
3. 아래의 INPUT 에 선정한 첫_데이터파일(first_airfoil_dat_name), 마지막_데이터파일(last_airfoil_dat_name) 의 파일명들을 입력한다.
4. 실행
'''

# ===== INPUT ======================
first_airfoil_dat_name = "2032c.dat"
last_airfoil_dat_name = "ys930.dat"
# ==================================


import os
import urllib.request as req
from bs4 import BeautifulSoup

url = "https://m-selig.ae.illinois.edu/ads/coord_database.html"
res = req.urlopen(url)
bs = BeautifulSoup(res, "html.parser")

coord_databaseContent = bs.find(id="coord_databaseContent")
all_the_tag_a = coord_databaseContent.find_all("a")  # find everything with the tag "a"


first_airfoil_dat_name_idx = 0
last_airfoil_dat_name_idx = 0

airfoil_name_list = []
idx = 0
for airfoil in all_the_tag_a:
    airfoil_name = airfoil.get_text()

    if ".dat" in airfoil_name:
        airfoil_name_list.append(airfoil_name)

        if airfoil_name == first_airfoil_dat_name:
            first_airfoil_dat_name_idx = idx
        if airfoil_name == last_airfoil_dat_name:
            last_airfoil_dat_name_idx = idx

        idx += 1

print("first_airfoil_dat_name_idx: ", first_airfoil_dat_name_idx)
print("last_airfoil_dat_name_idx: ", last_airfoil_dat_name_idx)

sliced_airfoil_name_list = airfoil_name_list[first_airfoil_dat_name_idx:last_airfoil_dat_name_idx]
len_sliced_airfoil_name_list = len(sliced_airfoil_name_list)
print("number of sliced_airfoil_name_list: ", len(sliced_airfoil_name_list))
print("sliced_airfoil_name_list: ", sliced_airfoil_name_list)

is_dir = os.path.isdir('./UIUC_airfoil_database')
if not is_dir:
    os.mkdir('./UIUC_airfoil_database')

for idx, airfoil_name in enumerate(sliced_airfoil_name_list):
    airfoil_name_wo_extension = airfoil_name.split('.')[0]

    # taking out the coordinate data
    print("({}/{})airfoil_name: {}".format(idx+1, len_sliced_airfoil_name_list, airfoil_name))
    url_foil = "https://m-selig.ae.illinois.edu/ads/coord/{}".format(airfoil_name)
    print("url_foil: ", url_foil)
    res_foil = req.urlopen(url_foil)

    with open("./UIUC_airfoil_database/{}.dat".format(airfoil_name_wo_extension), 'wb') as f:
        f.write(res_foil.read())
