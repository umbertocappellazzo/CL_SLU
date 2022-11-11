# Exploring the Joint Use of Rehearsal and Knowledge Distillation in Continual Learning for Spoken Language Understanding
This codebase contains the implementation of the experiments on the efficacy of combining rehearsal and knowledge distillation (KD) techniques applied to a Class-Incremental Continual Learning (CiCL) scenario for the Spoken Language Understanding task. This is the official code implementation of a paper submitted to ICASSP 2023.

## General overview

We explore a CiCL scenario for SLU, specifically for Intent Classification. The dataset used for the experiments is Fluent Speech Commands [[1]](#1). We consider multiple KD combinations at different levels in the network (i.e., feature and predictions space), as well as exploiting only rehearsal data or their combination with the current task data. The results supports the effectiveness of concurrently exerting predictions space and feature space KDs. Additionally, our method is expecially suitable for low-resource devices in that it attains larger gains for smaller rehearsal buffer's sizes.

![main figure](CL_SLU_scheme.png)

## Environment setup
The requested libraries for running the experiments are listed in the requirements.txt file. Run the command below to install them.  
**NOTA BENE**: for the experiments the forked continuum library must be used (https://github.com/umbertocappellazzo/continuum.git).

```
pip install -r requirements.txt
```


## References
<a id="1">[1]</a> 
L. Lugosch, M. Ravanelli, P. Ignoto, V. S. Tomar, and Y. Bengio, *Speech model pre-training for end-to-end spoken language understanding*, Interspeech 2019, 2019.
