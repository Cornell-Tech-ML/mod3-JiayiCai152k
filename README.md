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



# Task 3.4: Performance comparison graph
![image](https://github.com/user-attachments/assets/df8daf45-549c-44c5-947d-a821fe81d4d0)


# training result: python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
Epoch  0  loss  7.187436876970409 correct 29
Epoch  10  loss  4.912696218697055 correct 31
Epoch  20  loss  5.1184989925762405 correct 40
Epoch  30  loss  3.9004170663756446 correct 47
Epoch  40  loss  5.542352474878401 correct 44
Epoch  50  loss  3.4814258310728103 correct 48
Epoch  60  loss  4.102056319130968 correct 41
Epoch  70  loss  2.6593231414563805 correct 46
Epoch  80  loss  3.8896222369748945 correct 49
Epoch  90  loss  3.5328646908926977 correct 48
Epoch  100  loss  1.170678797337431 correct 48
Epoch  110  loss  1.5060796094821418 correct 48
Epoch  120  loss  1.3918993027395539 correct 49
Epoch  130  loss  0.9022941077254526 correct 48
Epoch  140  loss  2.9551371151552686 correct 47
Epoch  150  loss  1.1710663109522883 correct 48
Epoch  160  loss  0.8682919101330002 correct 48
Epoch  170  loss  1.5227010439653892 correct 50
Epoch  180  loss  1.9248235054236178 correct 48
Epoch  190  loss  0.656445643754626 correct 49
Epoch  200  loss  1.8396269046143294 correct 48
Epoch  210  loss  1.7806600996891766 correct 48
Epoch  220  loss  1.5804502430792784 correct 48
Epoch  230  loss  0.6985006940560926 correct 50
Epoch  240  loss  0.70808398728296 correct 49
Epoch  250  loss  1.2055089909602714 correct 49
Epoch  260  loss  1.1346026344395914 correct 47
Epoch  270  loss  0.9814926576793291 correct 50
Epoch  280  loss  1.8194017499348392 correct 49
Epoch  290  loss  1.0855355617046611 correct 50
Epoch  300  loss  1.5939051514494071 correct 49
Epoch  310  loss  1.1687783423765943 correct 49
Epoch  320  loss  0.4735570939092418 correct 49
Epoch  330  loss  0.8770185863271432 correct 50
Epoch  340  loss  1.3537308079312969 correct 47
Epoch  350  loss  0.23359918698378906 correct 49
Epoch  360  loss  1.2782217665718112 correct 50
Epoch  370  loss  1.0464999843588751 correct 49
Epoch  380  loss  0.9382517544925322 correct 50
Epoch  390  loss  0.03224754193507152 correct 50
Epoch  400  loss  0.9052951159259254 correct 49
Epoch  410  loss  0.5621777893193786 correct 49
Epoch  420  loss  0.8411746651627742 correct 49
Epoch  430  loss  0.5781448259872155 correct 50
Epoch  440  loss  0.6162177402663692 correct 48
Epoch  450  loss  0.5596992970357196 correct 50
Epoch  460  loss  0.06539415053059627 correct 49
Epoch  470  loss  1.0849781091406236 correct 50
Epoch  480  loss  0.035576374463522846 correct 50
Epoch  490  loss  0.6294851570978072 correct 50


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