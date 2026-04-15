# SAFE: Spiking Neural Network-based Audio Fidelity Evaluation

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 3, 5

## Abstract
Recent advancements in generative AI have enabled the creation of highly realistic synthetic audio, posing significant challenges in voice authentication, media verification, and fraud detection. While deep learning models are frequently used for fake audio detection, they often struggle to generalize to unseen and complex manipulations, particularly partial fake audio, where real and synthetic segments are seamlessly combined. This paper explores the use of Spiking Neural Networks (SNNs) for fake and partial fake audio detection, an area that has not yet been investigated. SNNs, known for their energy-efficient computation and ability to process temporal data, offer a promising alternative to traditional Artificial Neural Networks (ANNs). We propose an SNN-based approach for fake audio detection and comprehensively evaluate its performance through a series of experiments, including hyperparameter tuning, cross-dataset generalization and partial fake audio detection. 
Our results show that SNNs achieve accuracy comparable to state-of-the-art ANN models with fewer number of parameters. Although, SNNs did not offer significant improvements in generalization capabilities, they provided advantages such as reduced model sizes and computational efficiency, making them more suitable for resource-constrained and real-time voice authentication applications.
This study lays the groundwork for further exploration of SNNs in audio spoofing countermeasures, providing a foundation for future advancements in security-critical voice applications.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes a fake audio detection method based on SNN, aimed at detecting fully and partially fake audio. By applying SNN to this task, the paper demonstrates the potential advantages of this approach in terms of energy efficiency and model parameter reduction. The authors develope two types of SNN models (a fully connected SNN and a convolutional SNN) and evaluate their generalization performance across various datasets, such as ASVspoof and Fake or Real. Additionally, a partially fake audio dataset is created to test the models' fine-grained detection capabilities.

### Strengths
1. This study is the first to explore the use of SNN for audio deepfake detection, introducing a novel neural network architecture choice to the field.

2. Two SNN-based models are proposed in this research, demonstrating superior performance in parameter efficiency and energy consumption compared to traditional ANN, making them more suitable for resource-constrained applications.

3. Although the number of datasets tested is limited, the study provides a comprehensive evaluation of the approach, including hyperparameter tuning, cross-dataset testing, and detection of partially fake audio, thoroughly assessing the model's performance.

### Weaknesses
1. The application of SNN for audio deepfake detection brings a degree of novelty; however, simply employing SNNs for this task may limit the paper's depth of contribution. A more detailed explanation is needed to justify the choice of SNNs for audio deepfake detection, particularly considering the success of ANN-based models in this area. It is recommended to design and optimize the SNN model in alignment with the specific characteristics of audio deepfake detection to demonstrate an innovative approach beyond the straightforward application of SNN.

2. The dataset tested in this paper is limited, which can not reflect the effectiveness and generalization of the method proposed in this paper.

3. The comparative analysis is insufficient, especially in terms of comparisons with the latest deepfake detection methods. 

4. The descriptions of SNN architecture and implementation details are relatively brief. Including more detailed explanations would enhance the method's interpretability and reproducibility.

### Questions
1.It is hoped that the architecture and design of SNN model can be further improved in combination with the characteristics of audio depfake detection, rather than simple application.

2.To enhance the clarity of the paper's contributions, the authors could address the following aspects:

a.Conduct additional experimental analysis and comparisons to demonstrate the necessity of using SNN models for deepfake tasks.

b.Compared with the direct application of SNN, the model in this paper can reduce the amount of calculation and required parameters, which is the advantage of SNN model, and can not explain the contribution of the method in this paper. It needs to be compared with the direct application to illustrate how the improvement of the architecture in this work can further improve the performance and increase the generalization ability.

3.Provide more implementation details, such as pseudocode descriptions, to improve reproducibility.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper presents a novel approach to fake audio detection using Spiking Neural Networks (SNNs), specifically focusing on the challenges of detecting both fully and partially synthetic audio. The study's strengths include the development of an energy-efficient SNN model with comparable accuracy to state-of-the-art Artificial Neural Networks (ANNs) and its unique application to partial fake audio detection at the frame level, which is underexplored in existing research. Another strength is the detailed experimental analysis, which includes hyperparameter tuning and cross-dataset testing, demonstrating the robustness of SNNs across multiple datasets. However, limitations include novelty, insufficient baselines, and few evaluation datasets.  Overall, the writing is clear and technically thorough, effectively presenting both the potential and the limitations of SNNs in fake audio detection.

### Strengths
- **Comprehensive Hyperparameter Tuning**: The paper presents thorough hyperparameter tunings for SNN and CSNN models with various loss functions (e.g., CE-rate, CE-count) and surrogate gradients (e.g., Arctangent, Fast Sigmoid). This systematic tuning strengthens model performance.

- **Cross-Dataset Generalization**: Experiments evaluate model generalization across different datasets, testing models trained on Fake or Real against ASVspoof-2019 and vice versa. 

- **Partial Fake Audio Detection**: The paper introduces a new partial fake audio dataset (PFA dataset) to explore model effectiveness in detecting partially manipulated audio at the frame level. This innovative frame-level classification demonstrates SNNs' potential for temporal precision in handling complex audio manipulations

### Weaknesses
- **Insufficient Baselines**: The methods compared in this study are relatively simple, lacking more advanced baseline methods that are widely recognized as state-of-the-art in synthetic speech detection. For a more comprehensive evaluation, SOTA methods like AASIST [1], AASIST-L [1], RawNet2 [2], and RawGATST [3] should be included, as they provide robust comparisons with current advanced approaches in audio anti-spoofing.

- **Few Evaluation Datasets**:  The experiments rely on only two datasets, ASVspoof-2019 and Fake or Real. While these are important datasets for this field, other available datasets, such as WaveFake [4] and In-the-Wild [5], would enhance the study’s scope by testing the model’s generalization across a broader array of synthetic audio sources, further verifying its performance on diverse data.

- **Lack of Robustness Evaluation**: The study does not explore the robustness of the SNN and CSNN models under common real-world audio perturbations, such as MP3 compression or Gaussian noise. Evaluating the models with these types of perturbations would provide a more practical understanding of their performance and adaptability in noisy or degraded audio environments.

### References
1. Jee-weon Jung, Hee-Soo Heo, Hemlata Tak, Hye-jin Shim, Joon Son Chung, Bong-Jin Lee, Ha-Jin Yu, and Nicholas Evans. AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks. *ICASSP*, 2022.

2. Hemlata Tak, Jose Patino, Massimiliano Todisco, Andreas Nautsch, Nicholas Evans, and Anthony Larcher. End-to-end anti-spoofing with RawNet2. *ICASSP*, 2021.

3. Hemlata Tak, Jee-weon Jung, Jose Patino, Madhu Kamble, Massimiliano Todisco, and Nicholas Evans. End-to-end spectro-temporal graph attention networks for speaker verification anti-spoofing and speech deepfake detection. *ASVspoof 2021 Workshop*, 2021.

4. Joel Frank and Lea Schonherr. WaveFake: A dataset to facilitate audio deepfake detection. *NeurIPS Datasets and Benchmarks Track*, 2021.

5. Nicolas M. Muller, Pavel Czempin, Franziska Dieckmann, Adam Froghyar, and Konstantin Bottinger. Does audio deepfake detection generalize? *Interspeech*, 2022.

### Questions
- Would it be possible to conduct additional experiments incorporating these SOTA baselines to better contextualize the performance of the SNN and CSNN models? Given these models are considered state-of-the-art, including them could provide a stronger comparison. 

-  Would the authors consider adding datasets like WaveFake or In-the-Wild, which feature different synthetic audio sources? Such datasets could better reveal the models’ generalization capability across various audio manipulation techniques. 

- Since real-world audio is often subject to distortions such as compression and noise, how might the SNN and CSNN models perform under these common perturbations (e.g., MP3 compression, Gaussian noise)? Have the authors considered conducting robustness tests to simulate these conditions? Exploring this could add substantial practical relevance to the study, indicating the models’ adaptability in more variable environments.

- The SNN and CSNN models show parameter efficiency, but could the authors provide more details on their real-time applicability, especially in comparison to more complex models like AASIST? Given the reduced model size, what latency and computational gains are observed, and how might these impact deployment in resource-constrained settings?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This study explores the application of Spiking Neural Networks (SNNs) in detecting both fully fake and partially fake audio. It specifically evaluates and compares the performance of two proposed SNNs against several self-implemented Artificial Neural Networks (ANNs) and existing models, focusing on their ability to generalize across different datasets

### Strengths
This work clearly introduces the proposed method, including feature extraction and various SNN designs. I particularly appreciate how the Experiments section is organized with a clear presentation, making it very easy for the reviewer to follow.

### Weaknesses
1. The related work mentioned in this paper may not fully represent the current research status, as deep fake audio detection has been a long-standing and ongoing research direction. As stated in Section 1, "the detection of partially fake audio—where real and synthetic audio segments are seamlessly merged—remains an underexplored area", which may not be accurate. For example, research on identifying or localizing fake segments within a piece of audio can be considered as detecting partially fake audio, according to the taxonomy of this work. The following literature from 2022 to 2024 provides just a few examples that are not included in this paper:
 - Yadav, Amit Kumar Singh, et al. "Mdrt: Multi-domain synthetic speech localization." ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024.
 - Xie, Yuankun, et al. "An Efficient Temporary Deepfake Location Approach Based Embeddings for Partially Spoofed Audio Detection." ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024.
 - Zhang, Lin, et al. "Range-Based Equal Error Rate for Spoof Localization." arXiv preprint arXiv:2305.17739 (2023).
 - Khan, Awais, and Khalid Mahmood Malik. "Securing voice biometrics: One-shot learning approach for audio deepfake detection." 2023 IEEE International Workshop on Information Forensics and Security (WIFS). IEEE, 2023.

2. The motivation for introducing SNNs for detecting fake audio could have been more convincingly stated. For example, does the architecture of SNNs make them particularly suitable for this task? According to Section 1, "SNNs are inherently designed to process temporal data due to their ability to capture the timing and sequence of events, which makes them particularly suitable for tasks involving audio, which has rich temporal dynamics." However, the experimental results shown in Table 2 demonstrate that SNNs actually perform worse than the self-implemented CNN in most cases. Additionally, from an efficiency perspective, do SNNs offer other benefits beyond having fewer parameters that make them more suitable for practical use? If so, in what scenarios do existing continuous-parameterized neural networks fail to meet the detection latency requirements?

3. Since the reviewed related work is limited, this may also make the evaluation less comprehensive and representative. For example, as stated in Section 4.2, "Due to the lack of cross-dataset evaluation studies on assessing the generalization ability of ANNs for the fake audio detection problem...", this statement may not fully consider the necessity of cross-dataset generalization performance evaluation for existing deep neural network-based fake audio detectors. If the evaluation is limited to simple ANN architectures, such as feed-forward and convolutional architectures, then the considered evaluation scope could have been significantly expanded.

### Questions
N/A

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
In response to the evolving need for fake audio detection, the authors proposed a Spiking Neural Network (SNN) based detection method for detecting fake and partially fake audio.

### Strengths
1. The idea is interesting, and the research motivation is clear.
2. Language is mostly accessible, though with some minor issues

### Weaknesses
My suggestion is as follows:
1. The author proposes in the contribution that "combining real and synthetic samples from the Fake or Real dataset, Demonstrating the effectiveness of SNNs in detecting partial fake audio at the frame level, "but no specific experimental data or demonstration examples were provided for the detection effect at the frame level. It is recommended that the author further improve this section.
2. The author points out in the introduction the limitations of existing ANN models in generalization and processing complex forged audio, but lacks data support or experimental evidence for the specific manifestations of these limitations. Suggest citing specific literature or experimental data to quantify the shortcomings of existing methods in detecting counterfeit audio.
3. The model proposed in the paper contains multiple key modules, but the role of these modules is not fully emphasized in the overall framework diagram or textual description. It is suggested that the author highlight the core role of each module in the methodology section to highlight the logical structure of the model and deepen readers' understanding of its working mechanism.
4. The author designed multiple experimental plans, but the organization of experimental design and results is somewhat scattered. To enhance logical coherence, suggest clearly separating the objectives, methods, and results of each experiment into sections.
5. The experimental results presented in Table 2, especially the EER index, perform worse than the existing methods, which the authors must explain in the paper.

### Questions
Please refer to the above suggestions

### Soundness
2

### Presentation
3

### Contribution
2
