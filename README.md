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
Epoch  0  loss  6.717728481848455 correct 28
epoch_time: 4.015519380569458

Epoch  10  loss  4.582037345272636 correct 25
epoch_time: 1.545206069946289

Epoch  20  loss  5.634057536625928 correct 42
epoch_time: 1.5217540264129639

Epoch  30  loss  3.706966512811714 correct 50
epoch_time: 1.536975383758545

Epoch  40  loss  3.144123141677587 correct 48
epoch_time: 1.5894992351531982

Epoch  50  loss  3.334143554888687 correct 49
epoch_time: 1.5434613227844238

Epoch  60  loss  3.4778938625771048 correct 48
epoch_time: 1.5026631355285645

Epoch  70  loss  2.8428905443268704 correct 49
epoch_time: 1.5111711025238037

Epoch  80  loss  1.422143759907747 correct 47
epoch_time: 1.5105068683624268

Epoch  90  loss  1.4109767604015468 correct 49
epoch_time: 1.5863728523254395

Epoch  100  loss  2.036335914386677 correct 50
epoch_time: 1.5373501777648926

Epoch  110  loss  1.8954810796588142 correct 50
epoch_time: 1.507638931274414

Epoch  120  loss  0.6092754776042346 correct 50
epoch_time: 1.5215339660644531

Epoch  130  loss  1.8337589796199212 correct 48
epoch_time: 1.565504550933838

Epoch  140  loss  1.8754895828650748 correct 50
epoch_time: 1.5316178798675537

Epoch  150  loss  0.8119687052764666 correct 50
epoch_time: 1.511343002319336

Epoch  160  loss  0.9378536083467957 correct 49
epoch_time: 1.5388267040252686

Epoch  170  loss  1.5195569247774139 correct 49
epoch_time: 1.584627389907837

Epoch  180  loss  0.9476787021942626 correct 50
epoch_time: 1.5406525135040283

Epoch  190  loss  0.47955306733326086 correct 49
epoch_time: 1.5260612964630127

Epoch  200  loss  0.7354020146109536 correct 50
epoch_time: 1.5257198810577393

Epoch  210  loss  0.7598778660430875 correct 50
epoch_time: 1.6466975212097168

Epoch  220  loss  1.0330245592521183 correct 50
epoch_time: 1.523576021194458

Epoch  230  loss  1.1973829990550304 correct 49
epoch_time: 1.527214527130127

Epoch  240  loss  0.17059482660352643 correct 50
epoch_time: 1.5170457363128662

Epoch  250  loss  1.2414200854557291 correct 50
epoch_time: 1.5813508033752441

Epoch  260  loss  0.6228572389852884 correct 50
epoch_time: 1.532425880432129

Epoch  270  loss  0.907125207794287 correct 50
epoch_time: 1.5338213443756104

Epoch  280  loss  1.1110993524937784 correct 50
epoch_time: 1.516005516052246

Epoch  290  loss  0.8512186467538913 correct 50
epoch_time: 1.6183795928955078

Epoch  300  loss  0.14729819553619575 correct 50
epoch_time: 1.5331408977508545

Epoch  310  loss  0.08598953558972486 correct 50
epoch_time: 1.5247235298156738

Epoch  320  loss  0.9044110246071403 correct 49
epoch_time: 1.53857421875

Epoch  330  loss  0.7553859308733526 correct 50
epoch_time: 1.518317699432373

Epoch  340  loss  0.052698748280475625 correct 50
epoch_time: 1.5733671188354492

Epoch  350  loss  0.7137343317490632 correct 50
epoch_time: 1.5450201034545898

Epoch  360  loss  0.206959364063608 correct 50
epoch_time: 1.5347654819488525

Epoch  370  loss  0.28669074969178066 correct 50
epoch_time: 1.5068490505218506

Epoch  380  loss  0.15179671323240854 correct 50
epoch_time: 1.5997393131256104

Epoch  390  loss  0.9215843764684617 correct 49
epoch_time: 1.5431885719299316

Epoch  400  loss  0.8800524556632704 correct 49
epoch_time: 1.542062759399414

Epoch  410  loss  0.14844338247260247 correct 50
epoch_time: 1.5224010944366455

Epoch  420  loss  0.08753516407250234 correct 50
epoch_time: 1.5919578075408936

Epoch  430  loss  0.10040418831267703 correct 50
epoch_time: 1.5005970001220703

Epoch  440  loss  0.448225609486411 correct 50
epoch_time: 1.6086390018463135

Epoch  450  loss  0.8524166254283521 correct 50
epoch_time: 1.536520004272461

Epoch  460  loss  0.35590297051006364 correct 50
epoch_time: 1.4986732006072998

Epoch  470  loss  0.10927234878573203 correct 50
epoch_time: 1.5161786079406738

Epoch  480  loss  0.8562199608478566 correct 50
epoch_time: 1.510741949081421

Epoch  490  loss  0.772350348152507 correct 50
epoch_time: 1.5090112686157227


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
Epoch  0  loss  5.81109258925818 correct 43  

epoch_time: 4.736538887023926  

Epoch  10  loss  4.820453667362473 correct 47  

epoch_time: 1.5161447525024414  

Epoch  20  loss  2.6111075164787447 correct 49  

epoch_time: 1.5952422618865967  

Epoch  30  loss  0.8659659226981535 correct 49  

epoch_time: 1.5494053363800049  

Epoch  40  loss  1.5565931585138426 correct 49  

epoch_time: 1.5565311908721924  

Epoch  50  loss  0.30481255161243076 correct 49  

epoch_time: 1.5058917999267578  

Epoch  60  loss  0.34138252785942064 correct 50  

epoch_time: 1.519963264465332  

Epoch  70  loss  0.7388958397338877 correct 50  

epoch_time: 1.5001416206359863  

Epoch  80  loss  0.6079043453213785 correct 49  

epoch_time: 1.5020177364349365  

Epoch  90  loss  0.46919154098309546 correct 50  

epoch_time: 1.606436014175415  

Epoch  100  loss  1.31893585089622 correct 50  

epoch_time: 1.5577714443206787  

Epoch  110  loss  0.13175369473898554 correct 50  

epoch_time: 1.688108205795288  

Epoch  120  loss  0.6839479994117982 correct 50  

epoch_time: 1.5209879875183105  

Epoch  130  loss  0.39161537748788044 correct 50  

epoch_time: 1.5665771961212158  

Epoch  140  loss  1.2235387247155591 correct 49  

epoch_time: 1.7704510688781738  

Epoch  150  loss  0.44356752103237673 correct 50  

epoch_time: 1.5540375709533691  

Epoch  160  loss  0.27410023073697093 correct 50  

epoch_time: 1.565624713897705  

Epoch  170  loss  0.6894385573136292 correct 50  

epoch_time: 2.166369915008545  

Epoch  180  loss  0.7957617927799696 correct 50  

epoch_time: 1.5536339282989502  

Epoch  190  loss  0.172591766246461 correct 50  

epoch_time: 1.5548880100250244  

Epoch  200  loss  0.7438731648059108 correct 50  

epoch_time: 2.2868924140930176  

Epoch  210  loss  0.8250976907572676 correct 50  

epoch_time: 1.614004135131836  

Epoch  220  loss  0.2454566274963353 correct 50  

epoch_time: 1.5159759521484375  

Epoch  230  loss  0.019035854913605522 correct 50  

epoch_time: 2.1440060138702393  

Epoch  240  loss  0.014865565885734481 correct 50  

epoch_time: 1.5338385105133057  

Epoch  250  loss  0.05223922500293619 correct 50  

epoch_time: 1.5889508724212646  

Epoch  260  loss  0.21959972233885383 correct 50  

epoch_time: 2.130535125732422  

Epoch  270  loss  0.07177507793419269 correct 50  

epoch_time: 1.5324921607971191  

Epoch  280  loss  0.30433430414702595 correct 50  

epoch_time: 1.5083537101745605  

Epoch  290  loss  0.4697604167001581 correct 50  

epoch_time: 2.296480417251587  

Epoch  300  loss  0.3006288292174264 correct 50  

epoch_time: 1.5290427207946777  

Epoch  310  loss  0.07368418784812546 correct 50  

epoch_time: 1.5141422748565674  

Epoch  320  loss  0.05295042527879265 correct 50  

epoch_time: 2.267526149749756  

Epoch  330  loss  0.24994792339428504 correct 50  

epoch_time: 1.5228700637817383  

Epoch  340  loss  0.06627633106684737 correct 50  

epoch_time: 1.566927433013916  

Epoch  350  loss  0.06666962523872819 correct 50  

epoch_time: 2.2676620483398438  

Epoch  360  loss  0.20363446166836058 correct 50  

epoch_time: 1.51279878616333  

Epoch  370  loss  0.009208443450994613 correct 50  

epoch_time: 1.5109403133392334  

Epoch  380  loss  0.26853925256965777 correct 50  

epoch_time: 2.1449928283691406  

Epoch  390  loss  0.27725895062625466 correct 50  

epoch_time: 1.503042459487915  

Epoch  400  loss  0.021753372903922454 correct 50  

epoch_time: 1.5181267261505127  

Epoch  410  loss  0.004547324002729283 correct 50  

epoch_time: 1.9021797180175781  

Epoch  420  loss  0.2676778081633405 correct 50  

epoch_time: 1.5524263381958008  

Epoch  430  loss  0.22468329380069135 correct 50  

epoch_time: 1.5004584789276123  

Epoch  440  loss  0.0016645652568180336 correct 50  

epoch_time: 1.753539800643921  

Epoch  450  loss  0.08516373017864874 correct 50  

epoch_time: 1.505624771118164  

Epoch  460  loss  0.22143172779497725 correct 50  

epoch_time: 1.5021967887878418  

Epoch  470  loss  0.040441416403488306 correct 50  

epoch_time: 1.6299290657043457  

Epoch  480  loss  0.05112305883818996 correct 50  

epoch_time: 1.5059947967529297  

Epoch  490  loss  0.1794494999545364 correct 50  

epoch_time: 1.5151097774505615  


# training result: python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05
Epoch  0  loss  8.294246910367786 correct 27  

epoch_time: 4.234359502792358  

Epoch  10  loss  4.38553308389737 correct 44  

epoch_time: 2.414961576461792  

Epoch  20  loss  2.531462409137741 correct 47  

epoch_time: 1.5739068984985352  

Epoch  30  loss  2.0759634407224805 correct 47  

epoch_time: 1.6196537017822266  

Epoch  40  loss  3.3029979957950424 correct 48  

epoch_time: 1.6305718421936035  

Epoch  50  loss  3.03043401953576 correct 48  

epoch_time: 1.5838005542755127  

Epoch  60  loss  0.9509362505324317 correct 48  

epoch_time: 1.835296392440796  

Epoch  70  loss  0.9786672576606629 correct 48  

epoch_time: 1.585200309753418  

Epoch  80  loss  1.1362224614904126 correct 48  

epoch_time: 1.6144137382507324  

Epoch  90  loss  2.532807838621824 correct 48  

epoch_time: 2.2048745155334473  

Epoch  100  loss  2.846180254493979 correct 48  

epoch_time: 1.6011507511138916  

Epoch  110  loss  1.0099891161536554 correct 48  

epoch_time: 1.5662055015563965  

Epoch  120  loss  1.2083445109934179 correct 48  

epoch_time: 1.5777111053466797  

Epoch  130  loss  1.060776888217573 correct 48  

epoch_time: 1.7004847526550293  

Epoch  140  loss  1.7153687528640484 correct 49  

epoch_time: 2.3173575401306152  

Epoch  150  loss  0.43547744602468763 correct 49  

epoch_time: 1.5962255001068115  

Epoch  160  loss  0.6612122447415499 correct 49  

epoch_time: 1.601360559463501  

Epoch  170  loss  2.279925899278961 correct 49  

epoch_time: 1.7970256805419922  

Epoch  180  loss  0.4201947743658836 correct 48  

epoch_time: 1.576261281967163  

Epoch  190  loss  0.6606936331524483 correct 49  

epoch_time: 1.5760271549224854  

Epoch  200  loss  0.5321296683688608 correct 49  

epoch_time: 1.6279237270355225  

Epoch  210  loss  0.9322860577950421 correct 49  

epoch_time: 1.6207878589630127  

Epoch  220  loss  0.3113094807806439 correct 49  

epoch_time: 2.3915889263153076  

Epoch  230  loss  0.07058167616188726 correct 49  

epoch_time: 1.561945915222168  

Epoch  240  loss  2.4455115358782464 correct 49  

epoch_time: 1.6143934726715088  

Epoch  250  loss  0.21188715304456957 correct 49  

epoch_time: 1.7876877784729004  

Epoch  260  loss  1.2969385802701032 correct 49  

epoch_time: 1.595510482788086  

Epoch  270  loss  0.11735991683957364 correct 49  

epoch_time: 1.8287546634674072  

Epoch  280  loss  0.11431394843318572 correct 49  

epoch_time: 1.5801763534545898  

Epoch  290  loss  0.6516994655964454 correct 49  

epoch_time: 1.623307704925537  

Epoch  300  loss  0.13871353860685992 correct 49  

epoch_time: 2.2665045261383057  

Epoch  310  loss  0.9288550274620835 correct 49  

epoch_time: 1.6112453937530518  

Epoch  320  loss  0.1355345541912926 correct 50  

epoch_time: 1.5605087280273438  

Epoch  330  loss  0.39103090569554766 correct 50  

epoch_time: 1.6030986309051514  

Epoch  340  loss  1.3627947121906456 correct 50  

epoch_time: 1.6850934028625488  

Epoch  350  loss  0.027454081538152632 correct 49  

epoch_time: 1.9552042484283447  

Epoch  360  loss  1.6061248303559092 correct 50  

epoch_time: 1.589548110961914  

Epoch  370  loss  0.24643649798749562 correct 50  

epoch_time: 1.5934560298919678  

Epoch  380  loss  0.1474101210122944 correct 49  

epoch_time: 2.1051037311553955  

Epoch  390  loss  0.5062024780861155 correct 49  

epoch_time: 1.6096551418304443  

Epoch  400  loss  0.7304620609049406 correct 49  

epoch_time: 1.5997424125671387  

Epoch  410  loss  0.22584032685955724 correct 49  

epoch_time: 1.6494719982147217  

Epoch  420  loss  0.4483914331781795 correct 49  

epoch_time: 1.6470727920532227  

Epoch  430  loss  0.16561431222272796 correct 50  

epoch_time: 2.333692789077759  

Epoch  440  loss  0.3120477161994417 correct 49  

epoch_time: 1.5680794715881348  

Epoch  450  loss  0.20308582459095692 correct 49  

epoch_time: 1.6231036186218262  

Epoch  460  loss  0.10878338349288387 correct 49  

epoch_time: 1.8260040283203125  

Epoch  470  loss  0.2227540519686182 correct 50  

epoch_time: 1.5308430194854736  

Epoch  480  loss  0.06719595696965884 correct 49  

epoch_time: 1.6053342819213867  

Epoch  490  loss  1.2047302943271931 correct 50  

epoch_time: 1.6062824726104736  



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
Epoch  0  loss  7.316611685393863 correct 39  

epoch_time: 7.363204002380371  

Epoch  10  loss  3.8673041455904915 correct 35  

epoch_time: 0.249891996383667  

Epoch  20  loss  0.9460066964620966 correct 49  

epoch_time: 0.31812095642089844  

Epoch  30  loss  0.7470041376541555 correct 50  

epoch_time: 0.2756052017211914  

Epoch  40  loss  0.8185111046954201 correct 50  

epoch_time: 0.24865007400512695  

Epoch  50  loss  1.0734311960635048 correct 50  

epoch_time: 0.2728309631347656  

Epoch  60  loss  0.15974068946024572 correct 50  

epoch_time: 0.2523980140686035  

Epoch  70  loss  0.9895109879845742 correct 50  

epoch_time: 0.25618815422058105  

Epoch  80  loss  0.5862915284261478 correct 50  

epoch_time: 0.2478008270263672  

Epoch  90  loss  0.27111884910424505 correct 50  

epoch_time: 0.24985289573669434  

Epoch  100  loss  0.40457076019777233 correct 50  

epoch_time: 0.24332523345947266  

Epoch  110  loss  0.6726016876418988 correct 50  

epoch_time: 0.2403099536895752  

Epoch  120  loss  0.08191851439519597 correct 50  

epoch_time: 0.24979114532470703  

Epoch  130  loss  0.22271337401695868 correct 50  

epoch_time: 0.2933461666107178  

Epoch  140  loss  0.6010311786697443 correct 50  

epoch_time: 0.24342584609985352  

Epoch  150  loss  0.23647299364157706 correct 50  

epoch_time: 0.25012826919555664  

Epoch  160  loss  0.32873574216475293 correct 50  

epoch_time: 0.24888396263122559  

Epoch  170  loss  0.21844110489164592 correct 50  

epoch_time: 0.26287102699279785  

Epoch  180  loss  0.492446237896778 correct 50  

epoch_time: 0.26834774017333984  

Epoch  190  loss  0.1849672429662414 correct 50  

epoch_time: 0.2593111991882324  

Epoch  200  loss  0.33896855392274367 correct 50  

epoch_time: 0.8725488185882568  

Epoch  210  loss  0.08812842482375058 correct 50  

epoch_time: 0.2586357593536377  

Epoch  220  loss  0.19637039532243897 correct 50  

epoch_time: 0.278933048248291  

Epoch  230  loss  0.164973889849615 correct 50  

epoch_time: 0.23794078826904297  

Epoch  240  loss  0.10906020876567027 correct 50  

epoch_time: 0.25045084953308105  

Epoch  250  loss  0.33179086954640646 correct 50  

epoch_time: 0.23948192596435547  

Epoch  260  loss  0.12078397595095164 correct 50  

epoch_time: 0.2420041561126709  

Epoch  270  loss  0.09286032221104518 correct 50  

epoch_time: 0.24813580513000488  

Epoch  280  loss  0.21547429317252198 correct 50  

epoch_time: 0.2449648380279541  

Epoch  290  loss  0.15800337232314943 correct 50  

epoch_time: 0.23555707931518555  

Epoch  300  loss  0.19988848338740672 correct 50  

epoch_time: 0.2790851593017578  

Epoch  310  loss  0.19253543716926852 correct 50  

epoch_time: 0.24466896057128906  

Epoch  320  loss  0.1038174781829502 correct 50  

epoch_time: 0.2541329860687256  

Epoch  330  loss  0.11195006091020157 correct 50  

epoch_time: 0.2499980926513672  

Epoch  340  loss  0.21593678140389067 correct 50  

epoch_time: 0.27196311950683594  

Epoch  350  loss  0.14548296411103542 correct 50  

epoch_time: 0.2527780532836914  

Epoch  360  loss  0.0169039743224738 correct 50  

epoch_time: 0.25463128089904785  

Epoch  370  loss  0.10157128152866834 correct 50  

epoch_time: 0.24590778350830078  

Epoch  380  loss  0.13733984784934297 correct 50  

epoch_time: 0.24280619621276855  

Epoch  390  loss  0.024752534824895908 correct 50  

epoch_time: 0.24676990509033203  

Epoch  400  loss  0.10995143780606174 correct 50  

epoch_time: 0.24924874305725098  

Epoch  410  loss  0.09558909218772865 correct 50  

epoch_time: 0.24092698097229004  

Epoch  420  loss  0.021688994115109397 correct 50  

epoch_time: 0.2478940486907959  

Epoch  430  loss  0.021837920263698705 correct 50  

epoch_time: 0.832301139831543  

Epoch  440  loss  0.012727311126546717 correct 50  

epoch_time: 0.23774409294128418  

Epoch  450  loss  0.09002290904539748 correct 50  

epoch_time: 0.23777198791503906  

Epoch  460  loss  0.07163157309558374 correct 50  

epoch_time: 0.2578752040863037  

Epoch  470  loss  0.09060070283863839 correct 50  

epoch_time: 0.2458810806274414  

Epoch  480  loss  0.16857914336543453 correct 50  

epoch_time: 0.254986047744751  

Epoch  490  loss  0.1186446289397017 correct 50  

epoch_time: 0.24537181854248047  


# training result(bigger model): python run_fast_tensor.py --BACKEND gpu --HIDDEN 250 --DATASET split --RATE 0.05
Epoch  0  loss  17.032865661905838 correct 32  

epoch_time: 4.971484422683716  

Epoch  10  loss  3.1869761209288567 correct 44  

epoch_time: 1.909198522567749  

Epoch  20  loss  1.5481232580467892 correct 48  

epoch_time: 1.770247220993042  

Epoch  30  loss  3.429431248436704 correct 48  

epoch_time: 2.1529629230499268  

Epoch  40  loss  2.3697610350877114 correct 46  

epoch_time: 1.8314759731292725  

Epoch  50  loss  2.3285620552603152 correct 46  

epoch_time: 2.5560975074768066  

Epoch  60  loss  1.6757007031025595 correct 43  

epoch_time: 1.7515506744384766  

Epoch  70  loss  1.8649413670874067 correct 49  

epoch_time: 2.467421531677246  

Epoch  80  loss  0.6126939418192534 correct 49  

epoch_time: 1.755033254623413  

Epoch  90  loss  0.9766188352120756 correct 46  

epoch_time: 2.2752647399902344  

Epoch  100  loss  1.2808223579770321 correct 50  

epoch_time: 1.749701738357544  

Epoch  110  loss  0.28313754352069126 correct 50  

epoch_time: 1.8595283031463623  

Epoch  120  loss  1.9480116744547007 correct 50  

epoch_time: 1.7705392837524414  

Epoch  130  loss  1.674315063613149 correct 49  

epoch_time: 1.8339996337890625  

Epoch  140  loss  0.7904714937017161 correct 49  

epoch_time: 1.7816829681396484  

Epoch  150  loss  0.5093894646889735 correct 50  

epoch_time: 1.7329845428466797  

Epoch  160  loss  0.6170895293718378 correct 47  

epoch_time: 1.7810218334197998  

Epoch  170  loss  0.9196936483997812 correct 49  

epoch_time: 1.848097562789917  

Epoch  180  loss  0.9136326721449785 correct 50  

epoch_time: 1.7880432605743408  

Epoch  190  loss  0.7014139012892073 correct 50  

epoch_time: 1.7766826152801514  

Epoch  200  loss  0.22149622977626468 correct 50  

epoch_time: 2.0130372047424316  

Epoch  210  loss  0.7117331685612669 correct 50  

epoch_time: 1.8305902481079102  

Epoch  220  loss  0.25764817496722947 correct 50  

epoch_time: 2.350722074508667  

Epoch  230  loss  0.4134640853641126 correct 50  

epoch_time: 1.7570793628692627  

Epoch  240  loss  0.7474978102387817 correct 50  

epoch_time: 2.6074466705322266  

Epoch  250  loss  0.08212542016049787 correct 49  

epoch_time: 1.8405728340148926  

Epoch  260  loss  1.379308095721712 correct 48  

epoch_time: 2.280134439468384  

Epoch  270  loss  0.36906984679568233 correct 49  

epoch_time: 1.7592763900756836  

Epoch  280  loss  1.0823385006495274 correct 49  

epoch_time: 1.968402624130249  

Epoch  290  loss  0.44204783148507076 correct 50  

epoch_time: 1.8427789211273193  

Epoch  300  loss  0.08082098753465561 correct 49  

epoch_time: 1.7706799507141113  

Epoch  310  loss  0.4744759132302717 correct 49  

epoch_time: 1.7521486282348633  

Epoch  320  loss  1.303123448469844 correct 49  

epoch_time: 1.7499396800994873  

Epoch  330  loss  0.06455724833779819 correct 50  

epoch_time: 1.7310259342193604  

Epoch  340  loss  0.055019579025716164 correct 49  

epoch_time: 1.8243179321289062  

Epoch  350  loss  0.7558296615426267 correct 49  

epoch_time: 1.778430700302124  

Epoch  360  loss  0.7768750054606868 correct 49  

epoch_time: 1.7884349822998047  

Epoch  370  loss  0.5491476278246619 correct 50  

epoch_time: 1.9334750175476074  

Epoch  380  loss  1.2342653230550948 correct 48  

epoch_time: 1.8356032371520996  

Epoch  390  loss  0.5601713193567529 correct 49  

epoch_time: 2.3966927528381348  

Epoch  400  loss  0.4902287806315427 correct 49  

epoch_time: 1.738591194152832  

Epoch  410  loss  0.6967500522855703 correct 49  

epoch_time: 2.581817388534546  

Epoch  420  loss  0.08761476086519374 correct 49  

epoch_time: 1.8541274070739746  

Epoch  430  loss  0.02760575416771211 correct 49  

epoch_time: 2.4030063152313232  

Epoch  440  loss  0.013677084633519526 correct 50  

epoch_time: 1.7754592895507812  

Epoch  450  loss  0.3670713797106577 correct 50  

epoch_time: 1.980743169784546  

Epoch  460  loss  0.03678943841736381 correct 50  

epoch_time: 1.7433795928955078  

Epoch  470  loss  0.03987122314629547 correct 50  

epoch_time: 1.733795166015625  

Epoch  480  loss  1.492859514874095 correct 47  

epoch_time: 1.7434120178222656  

Epoch  490  loss  0.5280758753180783 correct 50  

epoch_time: 1.766920804977417  



# diagnostics output
![image](https://github.com/user-attachments/assets/26491c24-8ba0-4838-94bb-ffdf77e596c4)
![image](https://github.com/user-attachments/assets/2c3e60ee-abaf-420e-86af-fb31993e3701)
![image](https://github.com/user-attachments/assets/59c736be-1cf3-4698-9171-4b5fac4fe966)
![image](https://github.com/user-attachments/assets/b39cc8e8-b719-46ef-bfe8-c54aefa878f1)
![image](https://github.com/user-attachments/assets/2aa6d67e-ea84-4673-ad58-1d159992d30c)
