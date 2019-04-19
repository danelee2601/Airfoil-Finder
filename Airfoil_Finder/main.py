from Airfoil_Finder import AirfoilFinder
import os

"""
======================================================================================================================================
* 함수 arguments 에 대한 설명 *

1. sep_for_the_input_airfoil_data: 유저(user)가 input 으로 넣는 airfoil 좌표 데이터파일이, 어떤 "구분자" 로 나누어져있는지 명시.
   - '\t':tab 으로 나뉨  //  ',':comma로 나뉨  //  ' ':space로 나뉨

2. plot_my_airfoil: 유저가 input으로 넣는 airfoil 데이터를 plot 한다. (True or False)

3. n_the_most_similar_foils: 최대 몇개까지 input으로 넣은 airfoil 과 가장 유사한 foil을 output 할지 결정한다.
======================================================================================================================================

* 참고사항 *
- my_airfoil 폴더안에, 유저가 가지고있는 airfoil 좌표 데이터를 넣는다. (확장자는 txt, csv, dat 중 하나)
- my_airfoil 폴더안에는 하나의 airfoil 데이터만 넣도록 한다. (프로그램 한번실행할때, 하나의 given airfoil 에 대해서, 데이터베이스안의 airfoil 들과 비교하기때문에.)
"""

# Execute
AirfoilFinder(sep_for_the_input_airfoil_data='\t', plot_my_airfoil=True, n_the_most_similar_foils=3)


