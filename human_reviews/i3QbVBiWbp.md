# Learning Informative Latent Representation for Quantum State Tomography

- Avg Score: 4.75
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 6, 3

## Abstract
Quantum state tomography (QST) is the process of reconstructing the complete state of a quantum system (mathematically described as a density matrix) through a series of different measurements. These measurements are performed on a number of identical copies of the quantum system, with outcomes gathered as probabilities/frequencies. 
QST aims to recover the density matrix and the corresponding properties of the quantum state from the measured frequencies. 
Although an informationally complete set of measurements can specify the quantum state accurately in an ideal scenario with a large number of identical copies, both the measurements and identical copies are restricted and imperfect in practical scenarios, making QST highly ill-posed. The conventional QST methods usually assume adequate or accurate measured frequencies or rely on manually designed regularizers to handle the ill-posed reconstruction problem, suffering from limited applications in realistic scenarios. 
Recent advances in deep neural networks (DNNs) led to the emergence of deep learning (DL) in QST. However, existing DL-based QST approaches often employ generic DNN models that are not optimized for imperfect conditions of QST. In this paper, we propose a transformer-based autoencoder architecture tailored for QST with imperfect measurement data. Our method leverages a transformer-based encoder to extract \emph{an informative latent representation} (ILR) from imperfect measurement data and employs a decoder to predict the quantum states based on the ILR. We anticipate that the high-dimensional ILR will capture more comprehensive information about the quantum states. To achieve this, we conduct pre-training of the encoder using a pretext task that involves reconstructing high-quality frequencies from measured frequencies. Extensive simulations and experiments demonstrate the remarkable ability of the informative latent representation to deal with imperfect measurement data in QST.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a transformer-based autoencoder architecture for quantum state tomography (QST) with imperfect measurement data. A transformer-based encoder is pre-trained to extract informative latent representation (ILR) with the task of measurement frequency reconstruction, which is succeeded by a transformer-based decoder to estimate quantum states from the measurement operators and frequencies. Extensive simulations and experiments demonstrate the remarkable ability of the proposed model to deal with imperfect measurement data in QST.

### Strengths
- The paper introduces a novel and interesting idea of building Transformer-based neural network for quantum state tomography.

- The paper clearly introduces the QST preliminaries, well presents the ill-posed challenge for QST and insightfully discusses the related works.

- The experiment result shows that transformer auto-encoder can reconstruct quantum states far better than the baseline models from imperfect measurement data.

### Weaknesses
- The idea of using transformer self-attention layers for QST is not strongly motivated, and hence not theoretically sound to me. 

- The model does scales poorly with the number of qubits due to the exponential number of operators in a complete set of QST measurement, so the contribution is limited.

- The latent representation contains a mixture of encoded features and raw input features. This seems not reasonable in principle for transformer-based models, especially when the raw features and encoded features are quite different across different samples.

- The experiment is a bit slim and cannot well show the value of the proposed model. 
  - It appears that the baseline models are linear regression models without pre-training, so it is an unfair comparison because the proposed model is exposed to far more data than the baseline models due to the presence of the encoder.  Are there stronger NN baselines? Is it possible to train the non-pre-training baselines with both pre-training data and training-data for this work?
  - The ablation study is missing, whereas it is necessary for this model to justify the design of different model components, such as i) having the missed operators and ii) training a frequency decoder instead of directing the state decoder in pre-training.

### Questions
Please kindly see the weaknesses above.

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
This manuscript introduces a transformer-based architecture designed to address the challenge of quantum state tomography with imperfect measurement data. The authors present the encoder-decoder framework of their model and illustrate a pre-training technique for the encoder, enabling it to reconstruct high-quality frequencies from imperfectly measured data. Furthermore, the authors show the model's effectiveness by employing it in the reconstruction of arbitrary 2-qubit and 4-qubit quantum states, as well as in the prediction of their properties.

### Strengths
- This manuscript presents a versatile model capable of simultaneously performing quantum tomography and predicting quantum properties.
- The paper introduces a pre-training strategy aimed at enhancing the robustness of the proposed model.
- This paper applies the proposed model to quantum state tomography of arbitrary quantum states rather than focusing on specific states or predefined quantum state sets.

### Weaknesses
- I have doubts about the scalability of the proposed model for large-scale quantum systems, especially considering the exponential growth in the number of cube operators required. This implies that the dimension of the input layer for this model would increase exponentially . If this holds true, the resulting model would become exceedingly large when applied to large-scale quantum systems.

  For another, the experiments about QST in this paper are limited to 2-qubit and 4-qubit quantum states. Even for 4-qubit pure states, when $N_t = 100$ and no operators are masked, the reconstruction fidelity is approximately $1-e^{-2} \approx 0.865$, which is not high. If this limitation is attributed to the relatively small value of $N_t$, the authors may consider conducting additional experiments on 4-qubit states (or even larger quantum system) to address this concern. I was unable to locate such experiments in the appendix, which predominantly shows a series of additional experiments conducted on 2-qubit states. 

- I believe the proposed model lacks novelty in some sense. While it incorporates a transformer architecture, the fundamental encoder-decoder framework closely resembles those found in existing references [1] and [2] for quantum state tomography and quantum state learning. Furthermore, I feel that the pre-training strategy introduced here is similar to the setting in [2], which involves predicting measurement results for unmeasured bases.  I would appreciate it if the authors could clarify the distinction between "masked" operators in this paper and the unmeasured bases described in [2].

  [1] Ahmed, Shahnawaz, et al. "Quantum state tomography with conditional generative adversarial networks." *Physical Review Letters* 127.14 (2021): 140502.

  [2] Zhu, Yan, et al. "Flexible learning of quantum states with generative query neural networks." *Nature Communications* 13.1 (2022): 6222.

### Questions
Major concerns:

- I have stated two major concerns in the "Weakness" section above, with one relating to scalability and the other relating to novelty.

Minor questions:

- In the part of predicting quantum properties, the authors utilize locally rotated GHZ states and W states. Could the authors provide information on the range of values associated with the properties to be predicted for these two types of states?

- I have some doubts about the motivation of using the model for property prediction. Why not compute the properties directly from the predicted density matrix, especially considering that the state decoder is designed to generate the density matrix as the output?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors a transformer-based autoencoder architecture tailored for quantum state tomography with imperfect measurement data. However, the introduction of quantum mechanics is not explicit. In addition, some important points should be emphasized.

### Strengths
One significant advantage of this method is its capability to provide more comprehensive information when dealing with imperfect measurement data. By using a transformer-based encoder, it effectively extracts latent information from imperfect measurement data, improving the accuracy of quantum state estimation.

### Weaknesses
Please review the comments below.

### Questions
1. On Page 1, the authors have mentioned that, "To uniquely identify a quantum state, the measurements must be informatively complete to provide
all the information about ρ (Jeˇ zek et al., 2003). The exponential scaling of parameters in $\rho$ requires
an exponentially increasing number of measurements, each of which requires a sufficient number
of identical copies (Gebhart et al., 2023)." However, this statement is not entirely precise. When the density matrix is low-rank [1] or takes the form of a matrix product operator [2], the POVM may not be informatively complete. Consequently, when a low-dimensional structure exists within the density matrix, many traditional methods can be applied with significantly fewer repeated measurements, which is an important direction to explore compared to neural network-based approaches. The reviewer suggests that this structural aspect should be included in the introduction.

[1] J. Haah, A. Harrow, Z. Ji, X. Wu, and N. Yu, “Sample-optimal tomography of quantum states,” IEEE Transactions on Information
Theory, vol. 63, no. 9, pp. 5628–5641, 2017.

[2] Zhen Qin, Casey Jameson, Zhexuan Gong, Michael B Wakin, and Zhihui Zhu.  “Stable tomography for structured quantum states,” arXiv preprint arXiv:2306.09432, 2023.

2. In Section 3.1, PRELIMINARIES ABOUT QST, it is advisable to use the notation $2^n$ instead of just $d$. This change is necessary to establish the proper context for the definition of a qubit as introduced in Section 3.2, THE ILL-POSED QST PROBLEM. Additionally, it would be beneficial to introduce the concepts of Hermitian, positive semidefinite (PSD) structure, and unit trace in the density matrix earlier in the section for improved clarity.

3. In Figure 2, due to the missing definition of qubit, for readers without any quantum background, it is hard to compute the total number of density matrices. Consequently, the number of missed measurements will be meaningless.

4. In part "QST process using a transformer-based autoencoder", should the architecture need to be designed anew for different qubits and POVMs, the authors should underscore this requirement.

5. The reviewers suggests that the authors should add the convergence rate of infidelity for different algorithms.

6. In the section 4.2 RECONSTRUCTING DENSITY MATRICES, the use of 2-qubit and 4-qubit examples may be considered limited. It would be beneficial to include discussions involving at least 8-qubit systems for a more comprehensive analysis.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The submission extends the concept of the masked autoencoder to enhance the sample complexity of quantum state tomography. The authors have conducted numerical simulations involving systems of up to 12 qubits to assess the performance of their proposal. Nonetheless, several statements throughout the paper and the configurations used in the numerical simulations introduce confusion, making it challenging to discern the precise contributions of the submission.

### Strengths
The utilization of deep learning techniques to improve quantum state tomography (QST) represents an emerging and promising field. Nevertheless, the current body of work focused on designing specialized learning models for quantum state tomography remains relatively limited. The submission effectively addresses this gap and presents intriguing results.

### Weaknesses
The primary weakness of the submission stems from inaccuracies in statements and the presence of confusing settings. The presence of incorrect or imprecise statements obscures the novelty and technical contributions of the proposed method. Additionally, while the authors have conducted a series of numerical simulations, the absence of a comparative analysis with state-of-the-art methods hinders our ability to gauge the practical advancements offered by the proposed method.

### Questions
1)  The motivation behind designing the auto-decoder structure is not entirely clear. It remains uncertain whether the authors aim to directly adapt the concept of Masked autoencoders to tackle QST tasks or if deeper insights are guiding this choice. Providing more context on this decision would enhance the submission's coherence.

2) The use of a state decoder to predict state properties appears to introduce confusion. If a user's primary interest lies in estimating specific properties, more efficient methods may be available than the proposed approach. It is essential to consider that state reconstruction, even with the inclusion of masked operations, can be resource-intensive and time-consuming.  

3) The numerical simulations are limited to older methods for QST. Consequently, it remains uncertain whether the purported contributions and advantages can be effectively realized in practical applications. To establish the practicality and competitiveness of the proposed approach, a systematic examination involving a wider spectrum of advanced deep learning methods is imperative. For instance, recent studies [Ahmed, Shahnawaz, et al. "Quantum state tomography with conditional generative adversarial networks." Physical Review Letters 127.14 (2021): 140502.]  have explored the use of incomplete POVM information in conjunction with a generative adversarial learning scheme to address QST tasks, and a thorough comparative analysis with such contemporary approaches would greatly enhance the submission's value and relevance. 

4) In Table 3, the authors benchmark the proposed method for estimating coherence and entanglement of GHZ and W states with 8/12 qubits. Given that this task has also been investigated in a study by Zhu et al. in 2022, a comparative study becomes imperative. The relevant results would provide valuable insights into the relative strengths and weaknesses of these two methods for the specified task.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
