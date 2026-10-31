# BackToBasics-GCL

This repository contains the code for our NeurIPS 2026 accepted paper, \*Return to Basics: Very Simple Graph Contrastive Learning via Noise Cancellation Principle\*.

BackToBasics-GCL pairs GCN and MLP encoders for simple and effective graph contrastive learning guided by the noise cancellation principle.



\## Table of Contents



\- \[Requirements](#requirements)

\- \[Framework](#framework-overview)

\- \[Reproducing Results](#reproducing-results)

\- \[Citation](#citation)



\## Requirements



To install the required dependencies, refer to the `environment.yml` file.





\## Framework overview



!\[Framework overview](images/framework.png)





\## Reproducing Results



To reproduce the results in Table 3 and Table 4, run the following commands:





```bash

python main. py --dataset wisconsin --hidden\_dim 2048 --epoch 100 --lr 0.01 --runtime 10 --classifier\_time 10 --classifier\_epochs 2000 --cuda 3 --classifier\_lr 0.01 --classifier\_decay 0.0001 --batch\_norm --loss cosmean --GCN\_layers\_1 2 --a1 0.3 --a2 0.7



```





Additional comments are provided in `log`. Variants using alternative encoders (i.e., GAT-MLP and GRAND-MLP) are available in `log\\\_pde\\\_lin`.







\## Citation



If you find our helpful, consider to cite us:



```bibtex



@INPROCEEDINGS{Zhao:C26b,

\&#x20;   author = {Yanan Zhao and Feng Ji and Jingyang Dai and Jiaze Ma and Wee Peng Tay},

\&#x20;   title = {Return to Basics: Very Simple Graph Contrastive Learning via Noise Cancellation Principle},

\&#x20;   booktitle = {Advances in Neural Information Processing Systems},

\&#x20;   volume = {},

\&#x20;   pages = {},

\&#x20;   month = {Dec.},

\&#x20;   year = {2026},

\&#x20;   address = {Sydney, Australia},

}






