# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

# For the diagnostics output from Task 3.1, please refer to the very last section in this README.md
- refer to the last section

# Task 3.4: Performance comparison graph
![image](https://github.com/user-attachments/assets/df8daf45-549c-44c5-947d-a821fe81d4d0)


# training result: python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
Epoch  0  loss  7.445822393189848 correct 30
epoch_time: 7.2977190017700195
Epoch  10  loss  4.521938743726753 correct 43
epoch_time: 0.11993694305419922
Epoch  20  loss  5.912583759639359 correct 42
epoch_time: 0.11476683616638184
Epoch  30  loss  3.5800494981351556 correct 46
epoch_time: 0.1116189956665039
Epoch  40  loss  3.2767126055851876 correct 44
epoch_time: 0.10181903839111328
Epoch  50  loss  2.6024761428175287 correct 48
epoch_time: 0.09859800338745117
Epoch  60  loss  3.1597644828956803 correct 49
epoch_time: 0.1112067699432373
Epoch  70  loss  1.9079527705290502 correct 49
epoch_time: 0.10796785354614258
Epoch  80  loss  2.453387652272158 correct 48
epoch_time: 0.11259698867797852
Epoch  90  loss  1.5339929437738407 correct 49
epoch_time: 0.0993201732635498
Epoch  100  loss  0.5323745808912271 correct 47
epoch_time: 0.10586714744567871
Epoch  110  loss  2.9447016577921086 correct 46
epoch_time: 0.11317110061645508
Epoch  120  loss  2.867792115123219 correct 45
epoch_time: 0.1107938289642334
Epoch  130  loss  1.1059230694884628 correct 50
epoch_time: 0.11066198348999023
Epoch  140  loss  1.1956944992872192 correct 48
epoch_time: 0.12964177131652832
Epoch  150  loss  1.6603651686084977 correct 49
epoch_time: 0.1007530689239502
Epoch  160  loss  0.8765370374196019 correct 48
epoch_time: 0.11138725280761719
Epoch  170  loss  0.8159875225452595 correct 49
epoch_time: 0.10635614395141602
Epoch  180  loss  2.4344370473751296 correct 47
epoch_time: 0.09469008445739746
Epoch  190  loss  2.0958200089693833 correct 50
epoch_time: 0.10568594932556152
Epoch  200  loss  0.6914263535919101 correct 49
epoch_time: 0.1157231330871582
Epoch  210  loss  1.4352592726937574 correct 49
epoch_time: 0.10869717597961426
Epoch  220  loss  1.249815027804072 correct 49
epoch_time: 0.0956120491027832
Epoch  230  loss  1.1453899102493963 correct 49
epoch_time: 0.10956811904907227
Epoch  240  loss  0.7757416557826379 correct 49
epoch_time: 0.09362316131591797
Epoch  250  loss  1.1599120942176464 correct 49
epoch_time: 0.1254730224609375
Epoch  260  loss  0.09505005302555718 correct 49
epoch_time: 0.09549593925476074
Epoch  270  loss  0.6803133764646427 correct 49
epoch_time: 0.10737299919128418
Epoch  280  loss  1.6798644193805907 correct 49
epoch_time: 0.11071491241455078
Epoch  290  loss  0.9751171440948471 correct 49
epoch_time: 0.08868718147277832
Epoch  300  loss  0.44717199661665863 correct 49
epoch_time: 0.10836505889892578
Epoch  310  loss  0.33007361760941856 correct 50
epoch_time: 0.11147499084472656
Epoch  320  loss  0.6413110610359011 correct 49
epoch_time: 0.10384702682495117
Epoch  330  loss  0.6969251688636757 correct 49
epoch_time: 0.10065603256225586
Epoch  340  loss  0.9756319252359629 correct 50
epoch_time: 0.09060001373291016
Epoch  350  loss  0.2313731555591904 correct 50
epoch_time: 0.09162402153015137
Epoch  360  loss  0.8391960959146894 correct 50
epoch_time: 0.1084442138671875
Epoch  370  loss  0.716119580809441 correct 50
epoch_time: 0.11356592178344727
Epoch  380  loss  1.0692171020845782 correct 50
epoch_time: 0.11256790161132812
Epoch  390  loss  1.34141424425182 correct 48
epoch_time: 0.11257600784301758
Epoch  400  loss  0.9141063396329546 correct 49
epoch_time: 0.11143088340759277
Epoch  410  loss  1.6701766320801208 correct 49
epoch_time: 0.09439992904663086
Epoch  420  loss  1.0535018656411124 correct 50
epoch_time: 0.1037299633026123
Epoch  430  loss  1.218354574320491 correct 49
epoch_time: 0.10583615303039551
Epoch  440  loss  0.22876178094150776 correct 50
epoch_time: 0.11164474487304688
Epoch  450  loss  0.2503831384764659 correct 50
epoch_time: 0.1143348217010498
Epoch  460  loss  0.31560853152157853 correct 50
epoch_time: 0.10956573486328125
Epoch  470  loss  0.303491106224306 correct 49
epoch_time: 0.1053781509399414
Epoch  480  loss  0.15131628847663733 correct 50
epoch_time: 0.11961984634399414
Epoch  490  loss  0.17998043220229099 correct 50
epoch_time: 0.09608578681945801


# training result: python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
Epoch  0  loss  5.3947809963010265 correct 29
Epoch  10  loss  6.131547143167726 correct 41
Epoch  20  loss  5.054444264330629 correct 43
Epoch  30  loss  4.237249854852863 correct 43
Epoch  40  loss  4.038536655154629 correct 45
Epoch  50  loss  2.122629213553318 correct 46
Epoch  60  loss  2.854816675984871 correct 47
Epoch  70  loss  2.7919511976774958 correct 46
Epoch  80  loss  1.3360212690077933 correct 47
Epoch  90  loss  2.907089985998831 correct 49
Epoch  100  loss  1.774784307381059 correct 48
Epoch  110  loss  1.6103057222726616 correct 49
Epoch  120  loss  1.8786960318060029 correct 49
Epoch  130  loss  1.713317434770014 correct 49
Epoch  140  loss  1.7071688674426766 correct 49
Epoch  150  loss  1.577448320853768 correct 50
Epoch  160  loss  1.9323968967313119 correct 49
Epoch  170  loss  0.34838275865869806 correct 50
Epoch  180  loss  1.8304009835008062 correct 50
Epoch  190  loss  1.4134949756917983 correct 50
Epoch  200  loss  0.30856164310616313 correct 49
Epoch  210  loss  1.9286405731462137 correct 46
Epoch  220  loss  0.9519255207187081 correct 47
Epoch  230  loss  0.6378267738821762 correct 49
Epoch  240  loss  0.5352748146284406 correct 50
Epoch  250  loss  1.2706817296967845 correct 50
Epoch  260  loss  0.3956953574945397 correct 50
Epoch  270  loss  1.0229260529458282 correct 50
Epoch  280  loss  1.0545207753490797 correct 50
Epoch  290  loss  0.16415699877610448 correct 50
Epoch  300  loss  0.6243242633692028 correct 50
Epoch  310  loss  0.3785954004131106 correct 50
Epoch  320  loss  0.693022735704229 correct 50
Epoch  330  loss  0.7420111453300758 correct 50
Epoch  340  loss  0.281601102417783 correct 49
Epoch  350  loss  1.1028274366185329 correct 50
Epoch  360  loss  0.8390095934269279 correct 50
Epoch  370  loss  0.38855129153812706 correct 50
Epoch  380  loss  0.2200140191093598 correct 50
Epoch  390  loss  1.091925377603633 correct 50
Epoch  400  loss  0.3668255126640879 correct 50
Epoch  410  loss  0.3241201826513259 correct 50
Epoch  420  loss  0.6811873570996263 correct 50
Epoch  430  loss  0.6124997576291737 correct 50
Epoch  440  loss  0.7242899035887057 correct 50
Epoch  450  loss  0.6266417948912658 correct 50
Epoch  460  loss  0.385925649684277 correct 50
Epoch  470  loss  0.9827391226508098 correct 50
Epoch  480  loss  0.11844046090242219 correct 50
Epoch  490  loss  0.340771990449523 correct 50

# training result: python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
Epoch  0  loss  6.163998435435027 correct 48
epoch_time: 7.865776777267456
Epoch  10  loss  1.1579763546871347 correct 48
epoch_time: 0.10357499122619629
Epoch  20  loss  0.7118525845750189 correct 50
epoch_time: 0.10042405128479004
Epoch  30  loss  0.40373100236810155 correct 50
epoch_time: 0.09321022033691406
Epoch  40  loss  1.1095464652755296 correct 50
epoch_time: 0.10924410820007324
Epoch  50  loss  0.5049756157433383 correct 50
epoch_time: 0.11392903327941895
Epoch  60  loss  0.3729384155077025 correct 50
epoch_time: 0.09982991218566895
Epoch  70  loss  0.455246840175283 correct 50
epoch_time: 0.2530250549316406
Epoch  80  loss  0.2112975445573127 correct 50
epoch_time: 0.10764694213867188
Epoch  90  loss  0.2114209054624139 correct 50
epoch_time: 0.10713410377502441
Epoch  100  loss  0.7727265816145937 correct 50
epoch_time: 0.11004209518432617
Epoch  110  loss  0.8729909389256145 correct 50
epoch_time: 0.09215283393859863
Epoch  120  loss  0.6120108924530303 correct 50
epoch_time: 0.10887002944946289
Epoch  130  loss  0.5433328682035204 correct 50
epoch_time: 0.11092090606689453
Epoch  140  loss  0.303968040411877 correct 50
epoch_time: 0.09272384643554688
Epoch  150  loss  0.7169409064547434 correct 50
epoch_time: 0.10804891586303711
Epoch  160  loss  0.1889193779167473 correct 50
epoch_time: 0.11745905876159668
Epoch  170  loss  0.0841627988202809 correct 50
epoch_time: 0.11202812194824219
Epoch  180  loss  0.030318082497728112 correct 50
epoch_time: 0.11410689353942871
Epoch  190  loss  0.34764885243543964 correct 50
epoch_time: 0.11742377281188965
Epoch  200  loss  0.5034694342344429 correct 50
epoch_time: 0.2034599781036377
Epoch  210  loss  0.05506963512058817 correct 50
epoch_time: 0.1222679615020752
Epoch  220  loss  0.7288658292186783 correct 50
epoch_time: 0.09356498718261719
Epoch  230  loss  0.2724406654193557 correct 50
epoch_time: 0.09386801719665527
Epoch  240  loss  0.4333465022731868 correct 50
epoch_time: 0.10558485984802246
Epoch  250  loss  0.0011459662141643744 correct 50
epoch_time: 0.09274482727050781
Epoch  260  loss  0.49900996253025065 correct 50
epoch_time: 0.11388492584228516
Epoch  270  loss  0.026995718263733058 correct 50
epoch_time: 0.10363292694091797
Epoch  280  loss  0.2635435331590035 correct 50
epoch_time: 0.11456799507141113
Epoch  290  loss  0.1943782739827185 correct 50
epoch_time: 0.09852886199951172
Epoch  300  loss  0.37806157553788083 correct 50
epoch_time: 0.10616230964660645
Epoch  310  loss  0.05166152712742392 correct 50
epoch_time: 0.10873889923095703
Epoch  320  loss  0.15439924529006535 correct 50
epoch_time: 0.11493396759033203
Epoch  330  loss  0.26776419717186645 correct 50
epoch_time: 0.11307597160339355
Epoch  340  loss  0.3483582831663735 correct 50
epoch_time: 0.1066443920135498
Epoch  350  loss  0.2824426364542978 correct 50
epoch_time: 0.1213538646697998
Epoch  360  loss  0.3731540149361168 correct 50
epoch_time: 0.11135005950927734
Epoch  370  loss  0.00011393912409554713 correct 50
epoch_time: 0.10602807998657227
Epoch  380  loss  0.00937767653952811 correct 50
epoch_time: 0.09551119804382324
Epoch  390  loss  0.1880885712589458 correct 50
epoch_time: 0.1041109561920166
Epoch  400  loss  0.09169786441523699 correct 50
epoch_time: 0.11366391181945801
Epoch  410  loss  0.00129752214749818 correct 50
epoch_time: 0.1163487434387207
Epoch  420  loss  0.04985716451903268 correct 50
epoch_time: 0.11545991897583008
Epoch  430  loss  0.2540338245191435 correct 50
epoch_time: 0.12076592445373535
Epoch  440  loss  0.13094141957798403 correct 50
epoch_time: 0.09376120567321777
Epoch  450  loss  0.22559603940926595 correct 50
epoch_time: 0.09879326820373535
Epoch  460  loss  0.040550392150974474 correct 50
epoch_time: 0.11871576309204102
Epoch  470  loss  0.32404810189209965 correct 50
epoch_time: 0.11341428756713867
Epoch  480  loss  0.04339816925004113 correct 50
epoch_time: 0.10864901542663574
Epoch  490  loss  0.1335398363422713 correct 50
epoch_time: 0.09042787551879883

# training result: python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
Epoch  0  loss  5.985133223944341 correct 47
Epoch  10  loss  1.8128165886733794 correct 47
Epoch  20  loss  1.3333058003523397 correct 49
Epoch  30  loss  0.9469406721991892 correct 49
Epoch  40  loss  1.671246536464727 correct 50
Epoch  50  loss  0.5610160347193558 correct 49
Epoch  60  loss  0.8048781313728283 correct 50
Epoch  70  loss  0.15877539646367161 correct 50
Epoch  80  loss  0.5060754467341126 correct 50
Epoch  90  loss  0.29473578400775946 correct 50
Epoch  100  loss  0.4731815501478673 correct 50
Epoch  110  loss  0.7937629511119202 correct 50
Epoch  120  loss  0.11439163474193925 correct 50
Epoch  130  loss  0.3474182617853241 correct 50
Epoch  140  loss  0.04258161732731975 correct 50
Epoch  150  loss  0.32390700149147233 correct 50
Epoch  160  loss  0.38897150684440757 correct 50
Epoch  170  loss  0.22747499522476305 correct 50
Epoch  180  loss  0.09960428944913091 correct 50
Epoch  190  loss  0.17322403139097614 correct 50
Epoch  200  loss  0.06852665020351863 correct 50
Epoch  210  loss  0.1767169955111901 correct 50
Epoch  220  loss  0.06036456320540576 correct 50
Epoch  230  loss  0.5764498020265989 correct 50
Epoch  240  loss  0.10617205021531152 correct 50
Epoch  250  loss  0.12197770708465192 correct 50
Epoch  260  loss  0.04523165852004131 correct 50
Epoch  270  loss  0.007526359706512914 correct 50
Epoch  280  loss  0.41418589298809916 correct 50
Epoch  290  loss  0.037792273877176576 correct 50
Epoch  300  loss  0.017185835576316883 correct 50
Epoch  310  loss  0.23309138102627427 correct 50
Epoch  320  loss  0.04919910784799366 correct 50
Epoch  330  loss  0.16195093809930763 correct 50
Epoch  340  loss  0.04322497705144954 correct 50
Epoch  350  loss  0.2031502699305302 correct 50
Epoch  360  loss  0.327603057751918 correct 50
Epoch  370  loss  0.03533443668503492 correct 50
Epoch  380  loss  0.11853146348883938 correct 50
Epoch  390  loss  0.06895389238212675 correct 50
Epoch  400  loss  0.17883990739871725 correct 50
Epoch  410  loss  0.022440943200448937 correct 50
Epoch  420  loss  0.13883057696069742 correct 50
Epoch  430  loss  0.12441363546147184 correct 50
Epoch  440  loss  0.03500147422522322 correct 50
Epoch  450  loss  0.00237223946009531 correct 50
Epoch  460  loss  0.009668915354175843 correct 50
Epoch  470  loss  0.12734812073299115 correct 50
Epoch  480  loss  0.11399499260035534 correct 50
Epoch  490  loss  0.18440151112197123 correct 50

# training result: python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05
Epoch  0  loss  6.206980370350524 correct 33
Epoch  10  loss  6.193138606378534 correct 41
Epoch  20  loss  5.126704174074206 correct 41
Epoch  30  loss  3.7305234268305267 correct 42
Epoch  40  loss  5.100149475480432 correct 43
Epoch  50  loss  2.599017456646112 correct 42
Epoch  60  loss  2.5983459089200016 correct 43
Epoch  70  loss  3.656223894210413 correct 42
Epoch  80  loss  2.5844558706010465 correct 45
Epoch  90  loss  2.016396046925196 correct 46
Epoch  100  loss  0.8709626019323893 correct 43
Epoch  110  loss  2.8364134201490123 correct 42
Epoch  120  loss  2.7739760820074495 correct 42
Epoch  130  loss  1.5509605205865669 correct 49
Epoch  140  loss  1.413207802076577 correct 49
Epoch  150  loss  2.636658758464604 correct 46
Epoch  160  loss  1.2223385838163783 correct 48
Epoch  170  loss  1.5572852069278498 correct 48
Epoch  180  loss  2.1443964829018634 correct 46
Epoch  190  loss  1.518254096208731 correct 49
Epoch  200  loss  1.0340351638207064 correct 48
Epoch  210  loss  0.8327561983377963 correct 48
Epoch  220  loss  1.9178314377964814 correct 48
Epoch  230  loss  0.5889141973430247 correct 47
Epoch  240  loss  0.25113807699976526 correct 47
Epoch  250  loss  2.1452215335196616 correct 48
Epoch  260  loss  1.1425952624214277 correct 49
Epoch  270  loss  1.938852026890951 correct 48
Epoch  280  loss  0.4333608186171193 correct 49
Epoch  290  loss  1.9681000975755036 correct 49
Epoch  300  loss  2.2861509331782957 correct 48
Epoch  310  loss  1.1859519491778603 correct 49
Epoch  320  loss  1.8717172750536912 correct 47
Epoch  330  loss  0.9725522682650783 correct 49
Epoch  340  loss  2.877778786068698 correct 48
Epoch  350  loss  0.946545135697042 correct 48
Epoch  360  loss  2.3109287060807246 correct 50
Epoch  370  loss  0.6071088219030585 correct 49
Epoch  380  loss  0.878153890277749 correct 48
Epoch  390  loss  1.0599279997035747 correct 49
Epoch  400  loss  0.41213026553309423 correct 50
Epoch  410  loss  0.5680268133065663 correct 48
Epoch  420  loss  0.5728263950034523 correct 49
Epoch  430  loss  0.13394779071552615 correct 49
Epoch  440  loss  2.0471769831219557 correct 47
Epoch  450  loss  1.0022091859720452 correct 49
Epoch  460  loss  1.0664660317635684 correct 49
Epoch  470  loss  1.5622413868661307 correct 50
Epoch  480  loss  1.4424825441185352 correct 50
Epoch  490  loss  1.6819722468427918 correct 49


# training result: python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05
Epoch  0  loss  8.36315031282611 correct 28
epoch_time: 7.279940128326416
Epoch  10  loss  4.072067396841534 correct 46
epoch_time: 0.11493802070617676
Epoch  20  loss  3.1478568003970206 correct 45
epoch_time: 0.11676788330078125
Epoch  30  loss  2.6401245280205217 correct 48
epoch_time: 0.11493086814880371
Epoch  40  loss  2.0280877067931096 correct 48
epoch_time: 0.10429906845092773
Epoch  50  loss  1.433252310827919 correct 48
epoch_time: 0.11419010162353516
Epoch  60  loss  2.19482604125391 correct 46
epoch_time: 0.0943148136138916
Epoch  70  loss  1.2899295860392677 correct 48
epoch_time: 0.09535574913024902
Epoch  80  loss  2.189720959545852 correct 49
epoch_time: 0.08821320533752441
Epoch  90  loss  1.6994855387609467 correct 49
epoch_time: 0.10616827011108398
Epoch  100  loss  2.4650557428469555 correct 48
epoch_time: 0.1116180419921875
Epoch  110  loss  1.8794672259139589 correct 48
epoch_time: 0.10899209976196289
Epoch  120  loss  0.5431485241957159 correct 50
epoch_time: 0.10568499565124512
Epoch  130  loss  2.410874649142487 correct 48
epoch_time: 0.10943889617919922
Epoch  140  loss  1.5986955282794135 correct 50
epoch_time: 0.10559582710266113
Epoch  150  loss  0.703227407268501 correct 50
epoch_time: 0.0911250114440918
Epoch  160  loss  1.089422702801251 correct 50
epoch_time: 0.10401797294616699
Epoch  170  loss  0.7449483017650855 correct 50
epoch_time: 0.11442208290100098
Epoch  180  loss  0.6687126057148692 correct 50
epoch_time: 0.10988688468933105
Epoch  190  loss  0.7199744553517259 correct 50
epoch_time: 0.10456395149230957
Epoch  200  loss  0.4119565489110504 correct 49
epoch_time: 0.10875225067138672
Epoch  210  loss  0.5864933491778787 correct 50
epoch_time: 0.10627985000610352
Epoch  220  loss  0.3501649727765585 correct 49
epoch_time: 0.1086280345916748
Epoch  230  loss  0.18115957366858726 correct 49
epoch_time: 0.09923696517944336
Epoch  240  loss  0.336117489482747 correct 50
epoch_time: 0.10610508918762207
Epoch  250  loss  0.985788529140595 correct 50
epoch_time: 0.09052276611328125
Epoch  260  loss  1.2205943485929638 correct 49
epoch_time: 0.11292791366577148
Epoch  270  loss  0.2126188622800736 correct 50
epoch_time: 0.10412907600402832
Epoch  280  loss  0.6746481412499922 correct 50
epoch_time: 0.10424113273620605
Epoch  290  loss  0.09388580054399104 correct 50
epoch_time: 0.10902571678161621
Epoch  300  loss  0.16046056330419908 correct 50
epoch_time: 0.11506485939025879
Epoch  310  loss  0.14066526803184037 correct 50
epoch_time: 0.11208271980285645
Epoch  320  loss  0.26674486277678217 correct 50
epoch_time: 0.10839605331420898
Epoch  330  loss  0.06537516948645616 correct 50
epoch_time: 0.10368895530700684
Epoch  340  loss  0.6281481560898502 correct 50
epoch_time: 0.10086393356323242
Epoch  350  loss  0.39178531533179656 correct 50
epoch_time: 0.1150658130645752
Epoch  360  loss  0.21541487966443362 correct 50
epoch_time: 0.12190628051757812
Epoch  370  loss  0.30323031178123405 correct 50
epoch_time: 0.09541916847229004
Epoch  380  loss  0.8177080870761314 correct 50
epoch_time: 0.11521792411804199
Epoch  390  loss  1.036776885454755 correct 50
epoch_time: 0.10518002510070801
Epoch  400  loss  0.23446528862894284 correct 50
epoch_time: 0.11055803298950195
Epoch  410  loss  0.19376342059143167 correct 50
epoch_time: 0.10440516471862793
Epoch  420  loss  0.30442654279611714 correct 50
epoch_time: 0.0899660587310791
Epoch  430  loss  0.39708169167508567 correct 50
epoch_time: 0.11031389236450195
Epoch  440  loss  0.22346875892318813 correct 50
epoch_time: 0.11462712287902832
Epoch  450  loss  0.05820934462691193 correct 50
epoch_time: 0.11456418037414551
Epoch  460  loss  0.17773106594633487 correct 50
epoch_time: 0.10389089584350586
Epoch  470  loss  0.34222931145127317 correct 50
epoch_time: 0.10330915451049805
Epoch  480  loss  0.3244390516219912 correct 50
epoch_time: 0.10539078712463379
Epoch  490  loss  0.11940301260102762 correct 50
epoch_time: 0.10615706443786621


# training result(bigger model): python run_fast_tensor.py --BACKEND cpu --HIDDEN 250 --DATASET split --RATE 0.05
Epoch  0  loss  62.0137652093469 correct 26
Epoch  10  loss  2.0070184296994658 correct 44
Epoch  20  loss  2.321647393303683 correct 48
Epoch  30  loss  2.002363554066822 correct 50
Epoch  40  loss  2.1353809032976723 correct 50
Epoch  50  loss  1.8278922037233982 correct 46
Epoch  60  loss  0.5451656363738347 correct 50
Epoch  70  loss  0.6336765298650221 correct 50
Epoch  80  loss  0.6601595353357246 correct 49
Epoch  90  loss  1.377673990964113 correct 49
Epoch  100  loss  0.8211288653421968 correct 50
Epoch  110  loss  0.41302864393261546 correct 50
Epoch  120  loss  0.9574258283774593 correct 49
Epoch  130  loss  0.1863261644864488 correct 50
Epoch  140  loss  0.2226299391793173 correct 50
Epoch  150  loss  0.8690309908591938 correct 49
Epoch  160  loss  0.7183737765810317 correct 50
Epoch  170  loss  0.8517729990950437 correct 50
Epoch  180  loss  0.04280968482312693 correct 50
Epoch  190  loss  0.5602617806652957 correct 50
Epoch  200  loss  0.7698667385427231 correct 49
Epoch  210  loss  0.6042704869819052 correct 50
Epoch  220  loss  1.0334119355913156 correct 49
Epoch  230  loss  0.4098860713460407 correct 50
Epoch  240  loss  0.08446755059780271 correct 50
Epoch  250  loss  0.10255823085287147 correct 49
Epoch  260  loss  0.5299554077841727 correct 50
Epoch  270  loss  0.38306299433508706 correct 50
Epoch  280  loss  0.1738283376204281 correct 49
Epoch  290  loss  0.5057635854607219 correct 50
Epoch  300  loss  0.5962269410257884 correct 50
Epoch  310  loss  0.5902587036369955 correct 50
Epoch  320  loss  0.10242360310102706 correct 50
Epoch  330  loss  0.05530780707394624 correct 50
Epoch  340  loss  0.7897754227815197 correct 49
Epoch  350  loss  0.6755849779595673 correct 50
Epoch  360  loss  0.42068638408256087 correct 50
Epoch  370  loss  0.23487660970244267 correct 50
Epoch  380  loss  0.3816495262668425 correct 50
Epoch  390  loss  0.0676216718612071 correct 50
Epoch  400  loss  0.11307009819769498 correct 50
Epoch  410  loss  0.14576934603409364 correct 50
Epoch  420  loss  0.0566817416692542 correct 50
Epoch  430  loss  0.5312072584874707 correct 50
Epoch  440  loss  0.0463581708796455 correct 50
Epoch  450  loss  0.3618550110904429 correct 50
Epoch  460  loss  0.3241513716082847 correct 50
Epoch  470  loss  0.10483727313548302 correct 50
Epoch  480  loss  0.06690086292916073 correct 50
Epoch  490  loss  0.3158355100440084 correct 50

# training result(bigger model): python run_fast_tensor.py --BACKEND gpu --HIDDEN 250 --DATASET split --RATE 0.05
Epoch  0  loss  39.629843439072935 correct 33
Epoch  10  loss  4.93904997188946 correct 42
Epoch  20  loss  4.3218337591078 correct 40
Epoch  30  loss  1.717225549942774 correct 47
Epoch  40  loss  5.293099293533993 correct 44
Epoch  50  loss  1.2534050163013675 correct 47
Epoch  60  loss  1.396163092116289 correct 49
Epoch  70  loss  0.9433888702549166 correct 49
Epoch  80  loss  3.651705669392034 correct 50
Epoch  90  loss  3.396994192802379 correct 45
Epoch  100  loss  2.085896488545394 correct 47
Epoch  110  loss  1.0718533272018989 correct 49
Epoch  120  loss  0.1964551928793838 correct 49
Epoch  130  loss  0.6281428381365278 correct 49
Epoch  140  loss  0.8249348082678247 correct 49
Epoch  150  loss  0.22073926646878644 correct 47
Epoch  160  loss  1.2660541933271838 correct 49
Epoch  170  loss  0.19675416812006166 correct 48
Epoch  180  loss  0.34488878372670534 correct 49
Epoch  190  loss  2.5747084768083037 correct 47
Epoch  200  loss  1.7995362606050596 correct 49
Epoch  210  loss  0.8867364945662785 correct 50
Epoch  220  loss  0.08802875585643638 correct 47
Epoch  230  loss  0.32185768161959083 correct 48
Epoch  240  loss  1.4857347942778472 correct 49
Epoch  250  loss  0.13755564345001567 correct 49
Epoch  260  loss  0.6525592963012462 correct 48
Epoch  270  loss  0.21038823787909824 correct 49
Epoch  280  loss  2.017385831600231 correct 48
Epoch  290  loss  0.679790962792134 correct 50
Epoch  300  loss  1.8990094976860004 correct 49
Epoch  310  loss  2.3975780378503266 correct 49
Epoch  320  loss  0.048360833071931925 correct 49
Epoch  330  loss  1.0872618665195402 correct 49
Epoch  340  loss  0.7792865477776286 correct 49
Epoch  350  loss  1.6912014093751246 correct 49
Epoch  360  loss  1.1703946332641366 correct 49
Epoch  370  loss  2.188112294402277 correct 48
Epoch  380  loss  2.2476851946603427 correct 48
Epoch  390  loss  1.6281481294554963 correct 47
Epoch  400  loss  0.5682297697161588 correct 49
Epoch  410  loss  0.17271111698710745 correct 49
Epoch  420  loss  0.4172859927062233 correct 49
Epoch  430  loss  0.4877486597495276 correct 49
Epoch  440  loss  0.15547779598827313 correct 48
Epoch  450  loss  2.0266848854899386 correct 47
Epoch  460  loss  2.221689897120455 correct 48
Epoch  470  loss  0.07103355202806311 correct 49
Epoch  480  loss  1.6702788708950398 correct 49
Epoch  490  loss  0.05735634208709884 correct 49


#diagnostics output
python project/parallel_check.py
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/karen/Desktop/Cornell
Tech/CS5781_MLE/workspace/mod3-JiayiCai152k/minitorch/fast_ops.py (164)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/karen/Desktop/Cornell Tech/CS5781_MLE/workspace/mod3-JiayiCai152k/minitorch/fast_ops.py (164)
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                |
        out: Storage,                                                        |
        out_shape: Shape,                                                    |
        out_strides: Strides,                                                |
        in_storage: Storage,                                                 |
        in_shape: Shape,                                                     |
        in_strides: Strides,                                                 |
    ) -> None:                                                               |
        # TODO: Implement for Task 3.1.                                      |
        isStrideAligned = list(out_shape) == list(in_shape) and list(        |
            in_strides                                                       |
        ) == list(out_strides)                                               |
        if isStrideAligned:                                                  |
            for i in prange(len(out)):---------------------------------------| #2
                out[i] = fn(in_storage[i])                                   |
            return                                                           |
        else:                                                                |
            for i in prange(len(out)):---------------------------------------| #3
                out_index: Index = np.zeros(MAX_DIMS, np.int32)--------------| #0
                in_index: Index = np.zeros(MAX_DIMS, np.int32)---------------| #1
                to_index(i, out_shape, out_index)                            |
                broadcast_index(out_index, out_shape, in_shape, in_index)    |
                o = index_to_position(out_index, out_strides)                |
                j = index_to_position(in_index, in_strides)                  |
                out[o] = fn(in_storage[j])                                   |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--0 has the following loops fused into it:
   +--1 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial, fused with loop(s): 1)



Parallel region 0 (loop #3) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/karen/Desktop/Cornell
Tech/CS5781_MLE/workspace/mod3-JiayiCai152k/minitorch/fast_ops.py (182) is
hoisted out of the parallel loop labelled #3 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/karen/Desktop/Cornell
Tech/CS5781_MLE/workspace/mod3-JiayiCai152k/minitorch/fast_ops.py (183) is
hoisted out of the parallel loop labelled #3 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: in_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/karen/Desktop/Cornell
Tech/CS5781_MLE/workspace/mod3-JiayiCai152k/minitorch/fast_ops.py (216)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/karen/Desktop/Cornell Tech/CS5781_MLE/workspace/mod3-JiayiCai152k/minitorch/fast_ops.py (216)
-----------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                            |
        out: Storage,                                                                    |
        out_shape: Shape,                                                                |
        out_strides: Strides,                                                            |
        a_storage: Storage,                                                              |
        a_shape: Shape,                                                                  |
        a_strides: Strides,                                                              |
        b_storage: Storage,                                                              |
        b_shape: Shape,                                                                  |
        b_strides: Strides,                                                              |
    ) -> None:                                                                           |
        # TODO: Implement for Task 3.1.                                                  |
        isStrideAligned = list(out_shape) == list(a_shape) == list(b_shape) and list(    |
            a_strides                                                                    |
        ) == list(out_strides) == list(b_strides)                                        |
        if isStrideAligned:                                                              |
            for i in prange(len(out)):---------------------------------------------------| #7
                out[i] = fn(a_storage[i], b_storage[i])                                  |
            return                                                                       |
        else:                                                                            |
            for i in prange(len(out)):---------------------------------------------------| #8
                out_index: Index = np.zeros(MAX_DIMS, np.int32)--------------------------| #4
                a_index: Index = np.zeros(MAX_DIMS, np.int32)----------------------------| #5
                b_index: Index = np.zeros(MAX_DIMS, np.int32)----------------------------| #6
                to_index(i, out_shape, out_index)                                        |
                o = index_to_position(out_index, out_strides)                            |
                broadcast_index(out_index, out_shape, a_shape, a_index)                  |
                j = index_to_position(a_index, a_strides)                                |
                broadcast_index(out_index, out_shape, b_shape, b_index)                  |
                k = index_to_position(b_index, b_strides)                                |
                out[o] = fn(a_storage[j], b_storage[k])                                  |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--4 has the following loops fused into it:
   +--5 (fused)
   +--6 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial, fused with loop(s): 5, 6)



Parallel region 0 (loop #8) had 2 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/karen/Desktop/Cornell
Tech/CS5781_MLE/workspace/mod3-JiayiCai152k/minitorch/fast_ops.py (237) is
hoisted out of the parallel loop labelled #8 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/karen/Desktop/Cornell
Tech/CS5781_MLE/workspace/mod3-JiayiCai152k/minitorch/fast_ops.py (238) is
hoisted out of the parallel loop labelled #8 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: a_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/karen/Desktop/Cornell
Tech/CS5781_MLE/workspace/mod3-JiayiCai152k/minitorch/fast_ops.py (239) is
hoisted out of the parallel loop labelled #8 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: b_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/karen/Desktop/Cornell
Tech/CS5781_MLE/workspace/mod3-JiayiCai152k/minitorch/fast_ops.py (272)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/karen/Desktop/Cornell Tech/CS5781_MLE/workspace/mod3-JiayiCai152k/minitorch/fast_ops.py (272)
---------------------------------------------------------------------|loop #ID
    def _reduce(                                                     |
        out: Storage,                                                |
        out_shape: Shape,                                            |
        out_strides: Strides,                                        |
        a_storage: Storage,                                          |
        a_shape: Shape,                                              |
        a_strides: Strides,                                          |
        reduce_dim: int,                                             |
    ) -> None:                                                       |
        # TODO: Implement for Task 3.1.                              |
                                                                     |
        reduce_size = a_shape[reduce_dim]                            |
                                                                     |
        for i in prange(len(out)):-----------------------------------| #10
            out_index: Index = np.zeros(MAX_DIMS, np.int32)----------| #9
            to_index(i, out_shape, out_index)                        |
            o = index_to_position(out_index, out_strides)            |
                                                                     |
            result = out[o]                                          |
            a_pos = index_to_position(out_index, a_strides)          |
            # Perform the reduction along the specified dimension    |
            for s in range(reduce_size):                             |
                result = fn(result, a_storage[a_pos])                |
                a_pos += a_strides[reduce_dim]                       |
                                                                     |
            # Write the final reduced value to `out`                 |
            out[o] = result                                          |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #10, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--10 is a parallel loop
   +--9 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (serial)



Parallel region 0 (loop #10) had 0 loop(s) fused and 1 loop(s) serialized as
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/karen/Desktop/Cornell
Tech/CS5781_MLE/workspace/mod3-JiayiCai152k/minitorch/fast_ops.py (286) is
hoisted out of the parallel loop labelled #10 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/karen/Desktop/Cornell
Tech/CS5781_MLE/workspace/mod3-JiayiCai152k/minitorch/fast_ops.py (303)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/karen/Desktop/Cornell Tech/CS5781_MLE/workspace/mod3-JiayiCai152k/minitorch/fast_ops.py (303)
------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                              |
    out: Storage,                                                                         |
    out_shape: Shape,                                                                     |
    out_strides: Strides,                                                                 |
    a_storage: Storage,                                                                   |
    a_shape: Shape,                                                                       |
    a_strides: Strides,                                                                   |
    b_storage: Storage,                                                                   |
    b_shape: Shape,                                                                       |
    b_strides: Strides,                                                                   |
) -> None:                                                                                |
    """NUMBA tensor matrix multiply function.                                             |
                                                                                          |
    Should work for any tensor shapes that broadcast as long as                           |
                                                                                          |
    ```                                                                                   |
    assert a_shape[-1] == b_shape[-2]                                                     |
    ```                                                                                   |
                                                                                          |
    Optimizations:                                                                        |
                                                                                          |
    * Outer loop in parallel                                                              |
    * No index buffers or function calls                                                  |
    * Inner loop should have no global writes, 1 multiply.                                |
                                                                                          |
                                                                                          |
    Args:                                                                                 |
    ----                                                                                  |
        out (Storage): storage for `out` tensor                                           |
        out_shape (Shape): shape for `out` tensor                                         |
        out_strides (Strides): strides for `out` tensor                                   |
        a_storage (Storage): storage for `a` tensor                                       |
        a_shape (Shape): shape for `a` tensor                                             |
        a_strides (Strides): strides for `a` tensor                                       |
        b_storage (Storage): storage for `b` tensor                                       |
        b_shape (Shape): shape for `b` tensor                                             |
        b_strides (Strides): strides for `b` tensor                                       |
                                                                                          |
    Returns:                                                                              |
    -------                                                                               |
        None : Fills in `out`                                                             |
                                                                                          |
    """                                                                                   |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                |
                                                                                          |
    # TODO: Implement for Task 3.2.                                                       |
    # raise NotImplementedError("Need to implement for Task 3.2")                         |
    out_batch_stride = out_strides[0]                                                     |
                                                                                          |
    # Extract dimensions                                                                  |
    batch_size = out_shape[0]                                                             |
    m = out_shape[-2]  # Rows of the result                                               |
    n = out_shape[-1]  # Columns of the result                                            |
    k = a_shape[-1]  # Inner dimension                                                    |
                                                                                          |
    # Outer loop in parallel over batches and output matrix positions                     |
    for b in prange(batch_size):  # Iterate over batch------------------------------------| #11
        for i in range(m):  # Iterate over rows of the result                             |
            for j in range(n):  # Iterate over columns of the result                      |
                # Calculate the position in the output tensor                             |
                out_pos = (                                                               |
                    b * out_batch_stride + i * out_strides[-2] + j * out_strides[-1]      |
                )                                                                         |
                                                                                          |
                # Initialize accumulator for the dot product                              |
                result = 0.0                                                              |
                                                                                          |
                # Compute the dot product for the current position                        |
                for p in range(k):                                                        |
                    # Calculate positions in `a_storage` and `b_storage`                  |
                    a_pos = b * a_batch_stride + i * a_strides[-2] + p * a_strides[-1]    |
                    b_pos = b * b_batch_stride + p * b_strides[-2] + j * b_strides[-1]    |
                                                                                          |
                    # Perform multiplication and accumulate                               |
                    result += a_storage[a_pos] * b_storage[b_pos]                         |
                                                                                          |
                # Write the accumulated result to the output tensor                       |
                out[out_pos] = result                                                     |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None