# Airfoil-Finder

<h2>Description</h2>

<ul>

<li> 유저가 input으로 넣은 airfoil 형상좌표 데이터 를 UIUC airfoil database(2019년 기준, 약 1550개) 와 비교하여, 가장 유사한 airfoil 을 top n 개를 찾아준다. (가장 유사한 airfoil 을 몇개까지 찾을지는 유저가 선택할수있음)

<li> <i>목적</i> : 유저가 input으로 넣은 airfoil 과 유사한 airfoil 을 데이터베이스에서 찾고, 자신이 가진 airfoil 의 항,양력값의 정도높은 추정이 가능하다.
</ul>

  참고:
- UIUC airfoil database link: https://m-selig.ae.illinois.edu/ads/coord_database.html#M

<h2>구성 프로그램</h2>
<ol>
<li>(Main) Input airfoil 과 가장 유사한 에어포일을 데이터베이스에서 찾아주는 프로그램 ("Airfoil_Finder" 폴더내에 included)<br>
<li>(Optional) UIUC airfoil database website 에서 webcrawling 으로 데이터를 긁어오는 webcrawler (2019/04/15 기준으로, 모든 UIUC airfoil 데이터가 UIUC_airfoil_database 폴더안에 이미 들어있어있음. 유저가 바로 Main프로그램(Airfoil_Finder)을 사용할수있도록, 사전에 이 웹크롤러로 모든 데이터를 수집완료해둠.)
</ol>

<h2>Dependency (names of libraries)</h2> 
<i><b>Necessary</b></i>: numpy, matplotlib, pandas <br>
<i>Optional</i>: BeautifulSoup(for the webcrawler)

<h2>Quick Start</h2>

1. Airfoil_Finder폴더 내 "my_airfoil" 폴더안에, 자신이 가지고있는 airfoil좌표데이터를 넣는다.<br> 
&nbsp;&nbsp;* 유의사항: <br>
&nbsp;&nbsp;(a) 확장자는 txt, csv, dat 이여야 할것<br>
&nbsp;&nbsp;(b) 구분자(seperator)를 기억할것, 보통 tab or comma or space 들이 사용됨<br>
&nbsp;&nbsp;(c) my_airfoil 폴더안에는 하나의 airfoil좌표데이터만 넣을것 (프로그램이 하나의 given airfoil 에 대해서, 데이터베이스안의 airfoil 들과 비교하기때문에)<br>
&nbsp;&nbsp;(d) my_airfoil폴더안에 airfoil좌표데이터 예시로 "Tmotor.txt"가 들어있음. <br>


2. Airfoil_Finder 폴더안에 main.py 를 편집기(e.g. Pycharm)로 연다.<br>

3. main.py안에 AirfoilFinder() 함수안에 세개의 argument가 있다. 각각에 대한 자세한설명은 프로그램 내에 명시되어져있다.<br>

4. 위에서 명시된 세개의 argument 를 채우고, 프로그램을 실행한다.<br>
<hr> 

<p align="center">
<i>
Made by Daesoo Lee (이대수), Masters, Korea Maritime and Ocean University (한국해양대학교)<br>
e-mail : daesoolee@kmou.ac.kr<br>
First made on 04/14/2019<br><br></p>

<p align="center">
Made for 울산대학교 천이난류유동연구실
</i>
</p>

