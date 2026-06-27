from __future__ import annotations

import base64
import json
import zlib
from functools import lru_cache


WORKDAY_BAD_WEATHER_HOUR_PROFILE = "late_2012_workday_bad_weather_hour_rebalance"
WORKDAY_BAD_WEATHER_HOUR_COUNT_WEIGHT = 0.85946


_WORKDAY_BAD_WEATHER_HOUR_PAYLOAD = """
c-mE)+p1>A4TRriUvHZ}ALdm|n2Tfrfy9BtP68qCp0A`{tF=8C!_>d}pq5G{b^E^`KR&+w_oJj-e#+^mvOm71pXc
hS&b~@7xz^o2U%aO8xqi<}zv?-q*2+w4=hS=u`m6oDRQ;;-(N`_4^xAsa>-kmwsTAR(tyBzY{F_?9?fcnFt7FbOX
P$e%vy`7o6JBc{r3xK6&2|Ui2VFmCWtElEEW75kMh^I}1(q_j(Y!}bt(JO!P_ylG<(+qLJ#WU^`+uJwfByLP%P;F
Q*`NG#%A1wd!N+6QJ^t`{!2Fqh-nG|~TTO+1)|Owe3hZO0*|GJ^ZMOY_?NeJ%BhR@<?XBfhUa$)sv)9eSIi<-6FE
{`P!_86aYH1f%_JUL3=y}MbnL}&&E@)}kIr<!_)RYPaxpQtn4w~JRk+21OzQg7uv~a?0CRWz^o_9rAW>~Y0g&79t
ocg>2=OT14+iuu>jfxjs)U3quwOvQ9ZI)TXx6M}7$1D|h%_ZSP7wz^fX4t7^R>#W2numQ`TjS6<9_C#Eo%U@h8Si
1v+1k1z7u#<dB_;d<Ymb)hev9w76gD)gtj)$QYWv+g$zz$fbB1?E#rBJvq{iXqf!~e?&H4o=v>Jx&1CMQZ<_i{iV
k6S<q%}E-`R+F3Y=8^<Y&O(K<B--1Ho&|m$2%1H*t6bY-Jfb1z4k1hSm}aUljb_vMxOjzIEi`DYKGuzh+<}rXot<
%@nF2&@I!pLD#$OYc^Pc(#+vY_G~d9=emtwUF|+VlFj)9jp1ktxx#oqL;FcNYl`C%<Kle4}7#wi0qHMg4uEs8Wm<
gea`i}P6SQ+ZdFrT|cb~a#O*bGjbZOY;13z}2Z#hOxEHOpk{1<fZ??r=fx8&>VdQ27u=hm)bXBu9mFFK9%OaEio<
W>N7RFDMJI)!JUrhqgTOopab~p$ImDuZ;CLQ$9pNVC^;f5NqNv^#zytsBJ_igOAt#f@a;KVeuE}0@A);I#Kkk{Wo
Za;0Tju=SzHrA&Zub1ut6s+_X_;g0%MxBiqTa(LBc1UfA;8s}lvp!aqZ%B5j3{jVRz9EIdeMShL>^Jw#E)V&xbZY
lnH=QhdB|@T#+q`!!F*KNE4&6?{<WMV(X=h9C|qbTflXn5mXU9iFHJaK|mjn}{cZc|b5VoR3osv;vhwUIZ9Ec~P^
}7?m8)a^aD@9$RG3XG?p9Xff38FC-g|#)2U6O;RTd<QFs(Zm}do^65%-zMxr<70PES8y&>$3r@>s#NgN|lLz)Sel
}NI3kT#-OG`<#_=4_g0FW)~9Vv%l-PM*{XfGlI7A^rbSAzi7Xv*%4``SLAoU|HDh~Y6(ohK*dOKW6%OLemUx+`{9
Lza-R$S9m@n|I0POuJ|K6N`h(^S;LStW(KYW_{8|uNU;N+i>wr#O-8)FtGU&2_2n-4>ydm?}nV!Dy-C@R8~#uy5E
{NadxB~$%aPG3>z|UA_b!fc2|C$WW9^JF$nWX`dy^xZOLO_byV02A0ua=+%{kEq)FnGBjDq@a(_Xal*5J)p;QjGl
sl|vGxElkw`5d|#oT_S6Mx(I!QcRixpBd+e1!sn2;o~Is2B_HS2}TI887ZqB>4XYEzVgZ836)r;>7)eb~J1rG0J}
m6x8ol_On_jOZkXN%Qb?X46e&&p_j7f*a6M@77alX8W0QWE8it$Px)8?IwX(HS;Dm0tOSRSlqv_Cwq=HWl@rX|`9
YEL##6(|cr=Uk(<Yj$1lC2(hhcz5E`oy<K-U!R!|+yv&MH4>tAu%%`#-KUI8s>CmD52zqVWYR$k8C$EbewxU1(+Y
jMQH6*Yyc=V${+g1~fTYV&9RS5~nnFK^hTur3bV+VKOCWD@KME59+Fmxd>Dp0N&B&d_fC4vNFepDX2kKz2M;>P6e
LElH~O@F6v>2?~-LJJT_h8>_=M37NTisl}_lkdQq<)5I#)=&kF8tkF*l&ZIvBcCB`Zjwey3#6!OqimBlY==U18-X
h=Yn{hq-?LL6+RY)X@(#o!k;SBHL(4Af$KdHUE_SrK{Iuahnbp_0}`?f6t{C61WNqojGaYsW{yQlw!x+$1G*Q5gd
(h;cLQ$6=h~dsP|9Cnq8cr^V9i1&xEyyG0Sp!0>8)L7Nn}NN7lih$B39dQ{nxi=&b%imvZ*hi#tIY+xvrIO!KO4k
9lkNopj}Jf8T9vNcup4n{^3pNM`$8BWMw6<ioz#3f$P4u*Zhu}S4LY;eCcFCCITun^8q^1gGLmrn9Co<CqBa(KD>
^y;lEKUQ>1Gu?bq&y<u$D)Z5Tu*SHki!xYXDeYtA9W3aXvQ_p0CyD~s2&joytfQb)amc^{b@eJLW;0|aa(O`AhoL
@_L&nS$e%Dg&!w?f}fdG)7w(BmQNXgAze=h-mWv)wDL=zqjUnPE!O66M8Wtn^z=Ot-LHQ$k)gUCiyxyn9IrMrVHn
;6L3C0P40vs2IzWw99?SgG{IG$&#J#*+48QPln~s3{o=GZw_WvQQc?Xko)>=;<u^DaLrg<}RaRvwH2fUd%igCnQ?
Lhnt<g*boJf$V9>xkSpxt=o=Oy`GY&`vR`mnzDh<r@?=ulU8|o8(blkw#qqkkbe`pqVQFe=ys58x(d>PIsp5$|m{
eF_)a4BXT9la8tx0#mr~fc#gGZ;}7Qgi@iuJ8ikhSo;A7^2e<4uf3Ii6<t)>#-k<&F`lo}WB<m7sF7>V4+iHJtc5
p}}IP7O7|I8?Z+SssljQX_=KPyskqW2ky{Tv7A$SK+RN=HgGtOL7MJ&QOALmc8E$s3dI(_bw-p}qaG>pU(+f%y;3
mCLR1YIy*|uK4=9-fKPxjU`6pySxv0sUA>(5ZDy1)<dc%ZZC00|BOl;$#YHctYnhqfdbyinM2le_GRwNkJ|G9{HF
)o@t0<wzDi`wB}d8K>QElJ%1`Jl_?5ofZc)Cq&RFRB#DiJ08Rx=MLrt49vvDOib+!(&YMKCtOQ0;Dl`Rw+Pg39wl
TJ0|T3zo|9Y!Uf?_7L~#$R?3R7*{Vw1s)`^0ONy{-3OJLn7vC~NoZ2iVr0NH7b%`I(voOLQDWtTw;_;v4Wg4oY=c
noU2ledLB~>t%9B4%EIL)|*oW<%1Ih?{tx=X8?QdJ6#p~J$;jxe0+7(fiBECTA`-_<68TcPgR-?2~Qj8jJclHrHi
qvfK`l8lXhAj+Z3c|kLm+O!lv{HCb2e9>tGqXN{vVrX>^E-Ip=TG(nBNvR^2+C|O5G7*S}PMFYZ{{uQxA}RZ{R_j
)34RH282@WrEAL>1f0DG?%A)!Kubg|~D0rptrLv;&rT*x}Y&BYefLOsaZx>th_Z1&0u@~j3DKAyt1%)zpX@~XTJj
i(fhZ61c_-~vTtOR^ksZc8prC5PgbrME8|+s-1m7+<Oo?K7CA7<s9oTk+2+X*R-nc-l9y%@9RSJj2GyvDztyRF}@
Mv0`CKaIoR726lV4ebp<ulqz7VMC^TD)Z0K%Ju)Y+p5_~}uYN5QvgCL|Cb76May3n%p4Y<n(A8yK&;)E#Uoo4Ihk
W-%?WD*|@YF`+=pLb5ba)4lK!y$~3Twti8FykqVjNpke0YUL*)g$Iyq2uj%ju>a)J_SK$(#~(N8!I=%|`|FD!D_n
t5X$i<t-oBT(>6G6khz=bO5Qkql4459zfWX`!?D->xP4ElQY7)T?ybCx_F^bnK|`<c01yX@8DX<a2MDQdg|$^Rx4
#3DRjqmPhVImI4-HPR+k_T>=jw1R0fCml^+A_!l_axij;B8w!+H1F{GyDlu9;v8DaNq)~idlB_AEZX6G&=6;;hs4
dE2M)7eu^Q57rj4H@bE+C?oaMlQf6#7ADMUesqMNiw7|ZN6^evKG-pC{2~N2_X~)HqYKt1B0_9HqySJd3Kd^N=%A
-?=d@Pr$%aLOPU&QnHP0!t5Adqwz{%oT-44<mmzG1WP&4%uqamrD)Wb?L|T{dt$B8ndMBD$SXw!3_q5GPxVQlal}
^R}!1>sD>l8?E?5YRYQ~{YwMI64b3z;yrdo~Hfhyq)V5{>Dst@?s=#m10AEiTP0M#cpte`B}(<rK|i6aRs^bYYc8
JV)0BU5ed=KNeqcZl?pxp}w<1cJ6Q-=O$-cb4Kiay5NtKxNjFVI&_CbO}4|n-M7_9nj{7aSOi3{T5;7`u}K$0${p
g@?%V2Qs1K=|I_IK8+UDEp+DydsHkY}m$=yLYOi6-8)tLzB394ZYKU|-W18PMZa~6tjoRa`@lFM)Nv)e@}$Hh6S=
((s#xS>nmq-(skydXK~aG9+#iu|ps<8nbA)Rgb(HkA>O*<rgnr%vi!2%)Z2ls$*#D|xNToRE|@G@~*6m_&SIs+C_
y+^<XCOvp5829o2NVZ%<_NJp(%5wd3vQ@e9pQpnauU}HVNUceC;Wmt{(q~;toui5~3W~tWg53p<Ak|^!SzPo2{t3
2@Zl@E?mSG{>3UeAx{RZ=KTp@VwBZo%G<BH2YfCNt~~CwWA#u06A^H>J0t3z=^#NfCCc<v5ZsxVQ8*sx?cc5o-Lp
dS6++B>cdhXS{Pe&oLzy=~4*QZ(R;(r77bjy;+!8wq8#-RXiM|3$KA-RtMmsr`3cRhpzu9Tl<W#SLzfH6Nv?jYU>
yFh}5G7-FcDw*E-gIbkvewfq-;jiI4A#+QdyWt658$+i6_Xb(F5^(ecpV^w#7-Eg-Y1j;fnsvUt9r`PVS@mDDA8^
h|zr6lFn`{q=f9p><K01$FAVgszL#6~CyrQcOztttKpYQS&W5MzPye(a;mjS4S~6K_O~n<gh9aNEW0f!Bn+1QW!c
P(aGttG`%;NEsUMd`vz)!EL~KBe1ZLx>`tE4ovcwAV85tUD<u=|cz=3$=HArWdji6aB&tOO*sp4Jg<Elb<Z%5mw`
<R&S`Qvkr{~Mye%UzD!u&#3J@s)8N{S>&#ij&JoxG!y_`?CoLN)XW?0A-5g?3A?dS`?#sSz&c(gf!kN};<1C-Qid
;Lf^vVr%zXzp+Kd^r8n_lzU^|!_1y)6ie1OmuMdCqL$+joJmP6shz#UxAyQLZ@s+IE3J8&(Fbkbp^_ZcdldpU2h?
Q2Rzgl{CaLj!LGN-?RjXQ(#aG%FJ*_XQTKtyms}Au+&9m(V3p=)pj}<m_&S>bB3oD25`v}HJeskM&*GO<sIk&JZ_
v~$rQr&%1uwCzZbQHF)>QpslniXO9Z0dFROJy#3G!A~AAW#LUZDc_eHtn9>b&WG8Iic&8B*5ox5{DLlcHK?KuPxP
Z1eDZ76f3AJetqsIApv!kanfi}Pc^szg_CrA?S8TU_=vZp5aCc?9@bO}+dlmjJw+pn4PD9h5@7ReC>3R)%Nt!@gq
1mI>yZtAtUI*bi^`QzQ%Yzj^&23id{HMNh#b$?Gn7nacTxB4neF-wHNXAY%t1{kp<<8*NME%JTnd%<nwW#4n(C@}
q^Q<KJ?&!!)wV-V9da1i-ZMziWvgDQiq(tiB)T33>B2DK$lGbV+*fuK@wfmklB8L~$bL~H`QHWU*3r{;3GeJADpj
Xty$ab8F6V`SF}nn_cjh1XxNp`w3=y+VI>3G(AcN^zZ?$ona3#LABW%&jat6*L*waf~>Pwg%+aLhbhzIG9XjE3~-
q(_Fu(!K!uFCCI_u#r^4X|5{(qpI-zU8T5X{X~&?b-DVYN%ZwP>=bda#1%6sM8YRa`F*cM!~_-!@|7fP4Y$77B#2
!*5-lFTN)-M5moD@rFB6gW)i>N1*qZPPC_X6G2hhtLe+~SpCYxK$9&2(y6?bKII?;V>eRQ;eIHBDJV}G&g7)MN5m
MD=TI!>D@lf8UeqC=j6LD(Kdc&mC-_&JI*VS^Ivt4xg5v<(xM7C`G=%RKbf!HI%RC^@5QkeGW4Sz4trS6W(E{`fy
D$&zVHDxvv%&=c$Wxer1XQ+rHSJn7+H5m29l3L2v=nt?py`{4BXiR_d2n+Y>w%f&y-J?@~M%a3iy7?{EBY}WQFq!
uxtMu;H4H~ke#HIN)4y3A$C$;2TG>caNNpGVdKb*_&aUKtGMJol5O{K<G^dFbfN+SdB)k}&MKlbM}S^e9hnhv2Kz
xI1uU1C~aNbf#RxV}9}U81$DCttd-nHSCeNOexVc$Pc)bD@J;=SHu}yEL3GZhO0E_3A+cWe-TU-(?=Oo$^2-u2ww
mSTB0o$7nCLcdDLp2lU(*>zS$AxEQRDi#q?n)w)a@dWyO(=&fTZm^Ss|V(%F~HS%FR4p~(kRcH#+o_@QoOO}G1-#
v_ewKSPo_sm3;%C#`_sd%#Pp!KA#i!1_M{en?+j&NX!sm8CH*Gtr;IH5g{YjJ6=ICS|i)S*f3^l3}b(lF+<CWqef
e_6S!G}SIk=F;5ZAl<jO|FufUR!3a<_NP308L#77>y(KlYH6{fp7!f1y$1g~og4B<E3nB>UvUSxc!8?=&@%sdcKp
Cj+PZR;R8s3`yjD9`?dmy6{#jq3=Zog^Dp{?1>mkm4LCq_{Y8OU&?!QwQSktPQ6a6)RYCXn<n?E+$lJB~l*ukFXd
8JNLZb<G#=>>G0l56NWwVu(h5!O6Cb2s(O$x~{e*NdL3H{yBfRmqpIIeI$jMp#c+rdkO-Si@hAkQ<B5JI9+cuAbI
C55;lOb#SS{Anx7LhW1QTEszwZKDSBKaB4PnLcYylFZb@I3lT)2UL53j7Uv0@-c*_U<KiQ(+)bt@Eojf7|COT60D
GmVC%GbZU0Ai~1?Lr%dVYEnymdDnV9$5(r!?Bepm~O`4|8{<sUAM+Wp?=Z=xOyko4tG2YqS7cGXe=_ufkL=)EwdS
EJ@EqdzG@4(E7vY$DjZ5+xH*;`Qu-I|I?3ue*f{WAHV+g&yR1v{p0r^pC5nw%dg-6`1jv`{q5sh{rvd1{-2j6A@s
lh19$(=)B
"""


@lru_cache(maxsize=1)
def _workday_bad_weather_hour_payload() -> dict[str, object]:
    encoded = "".join(_WORKDAY_BAD_WEATHER_HOUR_PAYLOAD.split())
    raw_payload = zlib.decompress(base64.b85decode(encoded))
    return json.loads(raw_payload)


def workday_bad_weather_hour_count_weight() -> float:
    return float(_workday_bad_weather_hour_payload()["w"])


def workday_bad_weather_hour_factors() -> dict[tuple[int, int, int, int], float]:
    payload = _workday_bad_weather_hour_payload()
    return {
        (int(month), int(workingday), int(bad_weather), int(hour)): float(factor)
        for month, workingday, bad_weather, hour, factor in payload["f"]
    }


def workday_bad_weather_hour_event_factors() -> dict[str, float]:
    return {str(key): float(value) for key, value in _workday_bad_weather_hour_payload()["e"].items()}
