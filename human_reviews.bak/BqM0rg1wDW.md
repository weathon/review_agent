# PETNet - Coincident Particle Event Detection using Spiking Neural Networks

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 3

## Abstract
Spiking neural networks (SNN) hold the promise of being a more biologically plausible, low-energy alternative to conventional artificial neural networks. Their time-variant nature makes them particularly suitable for processing time-resolved, sparse binary data.
In this paper, we investigate the potential of leveraging SNNs for the detection of photon coincidences in positron emission tomography (PET) data.  PET is a medical imaging technique based on injecting a patient with a radioactive tracer and detecting the emitted photons. One central post-processing task for inferring an image of the tracer distribution is the filtering of invalid hits occurring due to e.g. absorption or scattering processes. Our approach, coined PETNet, interprets the detector hits as a binary-valued spike train and learns to identify photon coincidence pairs in a supervised manner. We introduce a dedicated multi-objective loss function and demonstrate the effects of explicitly modeling the detector geometry on simulation data for two use-cases. Our results show that PETNet can outperform the state-of-the-art classical algorithm with a maximal coincidence detection $F_1$ of 95.2\%. At the same time, PETNet is able to predict photon coincidences up to 36 times faster than the classical approach, highlighting the great potential of SNNs in particle physics applications.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a novel spiking neural network (SNN) architecture for coincident particle event detection in positron emission tomography (PET). Technically, the authors design a dedicated multi-objective loss function for SNNs that is both sensitive to spike counts and timing critical. Meanwhile, they implement large-scale data-parallel SNN training on a multi-node GPU system. Experiments show that PETNet outperforms SOTA algorithms with faster inference speed.

### Strengths
i) The topic of particle event detection using SNNs is very interesting and attractive.

ii) The authors verify a surprising conclusion that SNNs can speed up this task and even improve detection accuracy by learning coincidence patterns.

iii) The writing is straightforward, clear, and easy to understand.

### Weaknesses
i) I am curious and surprised by a sentence “SNN provide a promising alternative with the potential of surpassing conventional ANN prediction accuracy while requiring substantially less computational resources” in the introduction section. Why can SNNs have better performance than the corresponding ANN model and reduce computational complexity? I am curious and surprised by a sentence made in your introduction. Why can SNNs have better performance than the corresponding ANN model and reduce computational complexity? I am curious and surprised by a sentence made in your introduction. Why can SNNs have better performance than the corresponding ANN model and reduce computational complexity? Please prove this point in terms of theoretical explainability and experimentation. To the best of my knowledge, one of the biggest advantages of SNNs over ANNs is low power consumption.

ii) The authors should show more visualization results for better understand the task of particle event detection.

iii) The authors should give the detailed SNN architecture in the manuscript.

### Questions
See weakness.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This article formulates the type II coincidence pairing problem from PET dataset and highlights the time reduction and accuracy improvement over classical SCW algorithm using a well-designed SNN training method (PETNET).

### Strengths
please see Summary

### Weaknesses
1. The motivation of this research has migrated SNN into a new medical classification problem (i.e. PET data). The writer introduce a multi-objective loss function where data sparsity and temporal resolution are both considered. The PET dataset characteristics may well capture SNN inherent spatiotemporal design that could make up good performance. However, in most research topics, ANN is always outperforming SNN, especially in terms of precision (e.g. CIFAR-10, imageNet). It is questionable for directly comparison only between non-ML algorithm (SCW) and SNN. 

2. The multi-node GPU contributions is migrating the classification problem from non-GPU framework to GPU-accelerated framework (i.e. cuda). This acceleration raised the idea that if the problem can be fulfilled using only very simple ML algorithm that also can process in a very fast and accurate manner, such as SVM, decision trees or clustering if considering only single fully connected layer is needed for the SNN. 

3. It is suggested to combine the Figure 1 figure 2 into one figure. Especially the PET dataset, as a subset of biomedical signal, is little covered by research topic. It is at best to illustrate the dataset in picture and visualise 1-2 selected examples from numerical vector/matrix in your dataset.

4. Extending from (1), it is also suggested to include state-of-the-art algorithm which also capture temporal dynamics apart from single-layer LSTM. For example, transformer type model vs SNN approach or CNN based classification models. The expected SNN result may also be deployed on neuromorphic hardware instead of parallel hybrid “supercomputer”. If supercomputer is available, computing resources can be sacrificed for shorter time as there could be many available ML choices apart from SNN.

### Questions
please see the weakness

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
Spiking Neural Networks (SNNs) are explored as an energy-efficient alternative for processing Positron Emission Tomography (PET) data. The study introduces PETNet, a method that uses SNNs to filter invalid photon hits in PET imaging. Results show PETNet has a 95.2% F1 score and is 36 times faster than traditional methods.

### Strengths
The study utilizes Spiking Neural Networks (SNNs) to develop PETNet, which not only achieves an impressive F1 score of 95.2% in photon coincidence detection but also processes data at a remarkable speed.

### Weaknesses
While the author applied SNN technology to PET data processing, the innovative aspects of the research still appear limited. The LSTM mentioned in the article fails to effectively capture long-range dependencies, and the transformer faces challenges related to memory consumption due to sequence length. Regrettably, the author did not compare with these methods in the experiments.

### Questions
The SNN using LIF neurons primarily relies on the constant between membrane potentials to determine its temporal dependency, which may be inadequate. Moreover, it adopts the standard fully-connected SNN design. Compared to architectures like LSTM, RNN, and Transformer, where does the SNN's advantage lie? Can this be demonstrated through experimental results?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
