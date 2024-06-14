from datetime import datetime
from lunardate import LunarDate

def solar_to_lunar(year, month, day, time):
    # 양력 날짜 객체 생성
    solar_date = datetime(year, month, day)
    
    # 음력 날짜 변환
    lunar_date = LunarDate.fromSolarDate(year, month, day)
    
    # 결과 출력
    return [lunar_date.year,lunar_date.month,lunar_date.day,time]




# 변환 함수 호출

a=input('양력 생일을 입력하시오. 예) 1998 1 24 15:24      : ').split() #음력을 입력했을때의 기준으로 음력간지를 찾는 알고리즘이다.
b=solar_to_lunar(int(a[0]), int(a[1]), int(a[2]), a[3])
#연간지
ygan=(int(b[0])+7)%10
yji=(int(b[0])+9)%12
#월간지
mgan=(2*(ygan%5)+int(b[1]))%10
mji=(int(b[1])+2)%12
#일간지
if int(a[1]) in [1,2]:
    y=str(int(a[0])-1)
    m=int(a[1])+12
    fy=int(y[:2])
    by=int(y[2:])
    d=int(a[2])
    dgan=4*fy+int(fy/4)+5*by+int(by/4)+int((3*m+3)/5)+d+7
    dgan=dgan%10
    dji=8*fy+int(fy/4)+5*by+int(by/4)+6*m+int((3*m+3)/5)+d+1
    dji=dji%12
else:
    fy=int(a[0][:2])
    by=int(a[0][2:])
    m=int(a[1])
    d=int(a[2])
    dgan=4*fy+int(fy/4)+5*by+int(by/4)+int((3*m+3)/5)+d+7
    dgan=dgan%10
    dji=8*fy+int(fy/4)+5*by+int(by/4)+6*m+int((3*m+3)/5)+d+1
    dji=dji%12
#시 간지 
time=int(a[3].split(':')[0]+a[3].split(':')[1])
if 130<=time<330:
    time=2
elif 330<=time<530:
    time=3
elif 530<=time<730:
    time=4
elif 730<=time<930:
    time=5
elif 930<=time<1130:
    time=6
elif 1130<=time<1330:
    time=7
elif 1330<=time<1530:
    time=8
elif 1530<=time<1730:
    time=9
elif 1730<=time<1930:
    time=10
elif 1930<=time<2130:
    time=11
elif 2130<=time<2330:
    time=12
else:
    time=1
if dgan in [5,10]:
    hgan=(8+time)%10
else:
    hgan=(2*(dgan%5)-2+time)%10
hji=time%12
gan={1:'갑',2:'을',3:'병',4:'정',5:'무',6:'기',7:'경',8:'신',9:'임',0:'계'}
ji={1:'자',2:'축',3:'인',4:'묘',5:'진',6:'사',7:'오',8:'미',9:'신',10:'유',11:'술',0:'해'}
chgan={'갑':'甲','을':'乙','병':'丙','정':'丁','무':'戊','기':'己','경':'庚','신':'辛','임':'壬','계':'癸'}
chji={'자':'子','축':'丑','인':'寅','묘':'卯','진':'辰','사':'巳','오':'午','미':'未','신':'申','유':'酉','술':'戌','해':'亥'}
print('')
print(f'{gan[ygan]}{ji[yji]}년 {gan[mgan]}{ji[mji]}월 {gan[dgan]}{ji[dji]}일 {gan[hgan]}{ji[hji]}시')
print('')
print(f'{chgan[gan[ygan]]}{chji[ji[yji]]}년 {chgan[gan[mgan]]}{chji[ji[mji]]}월 {chgan[gan[dgan]]}{chji[ji[dji]]}일 {chgan[gan[hgan]]}{chji[ji[hji]]}시')
