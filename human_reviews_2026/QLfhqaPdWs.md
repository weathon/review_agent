# Self-Diagnosing Neural Networks: A Causal Framework for Real-Time Anomaly Detection in Training Dynamics

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Monitoring neural network training via scalar curves can obscure early indicators of failure in high-dimensional optimization dynamics. This work studies a framework that treats training as a spatiotemporal signal, converting sequences of internal activations and gradients into internal-state videos. A Dynamics Masked Autoencoder (DynamicsMAE) is pretrained on these videos to learn dynamics-aware representations, and a Temporal Vision Diagnostician (TeViD) with an evidential classification head is then fine-tuned for streaming, open-set diagnosis of training runs under a past-only constraint. Given a short window of internal-state frames, TeViD predicts diagnostic labels (Healthy, Overfitting, Instability, Catastrophic Forgetting, Concept Bias) and can abstain via an Unknown category. Evaluation uses time-to-detect, event-time AUPRC, risk--coverage analysis, and a simple decision-theoretic cost model. On >500 held-out runs from a curated data-factory benchmark with factorially held-out architectures, datasets, optimizers, and anomaly types, the method attains an event-time AUPRC of $0.96 \pm 0.01$ and issues alerts a median of $6.2$ epochs earlier than scalar rule-based baselines at a fixed $5\$% false-alarm rate, suggesting that internal-state video diagnosis can serve as a useful training-time signal alongside existing Machine Learning Operations (MLOps) tools on this benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper reframes neural training as a spatio-temporal signal and proposes a three-phase system: a data factory performs “video-ization” to the model internal states (activations and gradients) during training; dynamic-MAE, which is a self-supervised pretraining that learn dynamics-specific representations as a reconstruction task (masks the input); and TEVID, a diagnostician with an evidential head (EDL) for open-set decisions under strict causal constraints (no look-ahead, no validation/test at inference). On 500+ held-out runs spanning unseen architectures/datasets/optimizers, TEVID outperforms scalar-curve and generic video baselines.

### Strengths
1.	Novel problem framing: treating training internals as a video signal with strict streaming constraints is clear and practically relevant.
2.	Broad evaluation set: held-out splits across architectures, datasets, and optimizers demonstrate generalization rather than overfitting to a specific training setup.

### Weaknesses
1.	The introduction is full of statements that frame the motivation for this work and the gap it attempts to close, but  includes only 2 citations. Many statements are not justified or referenced.
2.	Limited ablations: the method combines several choices (activations and gradients as inputs, masked MAE pretraining, evidential head), yet ablations isolate these components only partially, making it unclear which parts actually drive performance.
3.	The narrative oversells real-world deployability without enough reported cost (memory/latency/storage). Overall clarity is good at a high level, but some details are underspecified.

### Questions
1.	Why are both activations and gradients required? Please include ablations showing performance of activations-only, gradients-only, and different window lengths W.
2.	What is the actual runtime and memory overhead of instrumenting a large model to dump activations + gradients every step? Please report wall-clock slowdown and storage requirements.
3.	How robust is TEVID when the anomaly labels are defined without validation-based hindsight? Maybe a testing scenario good be to remove one of the labels and test performance without it.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors propose a novel method for early detection of training instability that uses the network’s hidden states rather than a single scalar metric, such as the validation or training loss. The classification is performed in an open-set framework, with the option to refrain from prediction when uncertainty is high. The contributions are motivated from an information-theoretical perspective and supported by additional ablation studies. The authors demonstrate the performance of their method across multiple architectures, holding out some new architectures and a new domain (text) for the test set, and their method outperforms the scalar baselines on classic changepoint detection metrics.

### Strengths
The idea of analyzing the network’s inner state to stop training early is novel, interesting, and well-based in information theory. The authors correctly identify domain transfer as a key aspect of their contribution and demonstrate the performance of their methods within this framework. The chosen labels succinctly capture the various options of training failure. The choice of training and test samples (and labeling) is well-motivated by prior literature.

### Weaknesses
The main quality metrics are misaligned with the production task. With early stopping methods, the main question is “how much computation time will this method save?” If it can stop training a few epochs earlier on a 1000-epoch training run, saving us a few minutes of training but adding hours to the run, the method is useless.

The domain transfer is a key element here: if one needed to re-train the network for every domain, the contribution would be significantly limited, as the computational overhead would increase. However, from the paper’s text, it is not entirely clear how the authors handle differences in shape, which evidently arise when working with multiple architectures spanning both text and visual data. It is important that such handling should not involve any additional training, since this would nullify the benefits of domain transfer.

### Questions
1. What proportion of computational time does the method take up during training? How much time is saved on training, relative to the total training time, taking into account the time it took to run their method (i.e., the total time saved, minus the total runtime of this method throughout this training run)?
2. How do you handle the difference in shapes across varying architectures and domains, without additional training? Especially considering that the architecture and the weights of the proposed method remain the same across all runs. This is a key point - that should be better articulated in the paper’s text.
3. How is your paper positioned relative to the BANANAS [1] and similar frameworks [2]? Can you improve the hyperparameter optimization process by utilizing the developed approach?
4. Why did you avoid representation-based CPD baselines [3, 4]? For complex problems, they should work the best, according to their experimental results, given the complexity of the problem.

References: 
1. https://arxiv.org/pdf/1910.11858
2. Li, Lisha, et al. "Hyperband: A novel bandit-based approach to hyperparameter optimization." Journal of Machine Learning Research 18.185 (2018): 1-52.
3. Bazarova, Alexandra, Evgenia Romanenkova, and Alexey Zaytsev. "Normalizing self-supervised learning for provably reliable Change Point Detection." 2024 IEEE International Conference on Data Mining (ICDM). IEEE, 2024.
4. Chang, Wei-Cheng, et al. "Kernel Change-point Detection with Auxiliary Deep Generative Models." International Conference on Learning Representations. 2019.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces real-time anomaly detection in neural network training dynamics. Previous methods use error (or loss) as the diagnostic signal, similar to the bias-variance tradeoff. But the scalar value is insufficient to discover more complex anomalies including instability, catastrophic forgetting among others. To mitigate this problem, the paper introduces a high-dimensional internal state of a network (activations and gradients) as the diagnostic signal. The novelty of the framework is in its *causal* nature whereby the current decision is only dependent on past internal states. The results showcase an open-set streaming diagnosis framework and maintains a high performance across diverse architectures, datasets, and optimizers.

### Strengths
1. The paper proposes multiple types of anomalies that are important to detect while training.

2. Utilizing operation theory metrics including Median time-to-detect, event-time area under the precision-recall curve, and risk coverage curves are generally not used in representation learning settings. The paper brings diversity to the field, which is a definite strength.

3. As the authors state, the proposed method attains high diagnosis performance for unseen architectures, datasets, and optimizers.

### Weaknesses
**Clarity**: The Introduction, Related work, and Problem formulation do not convey the goal of the paper clearly. After reading through the full paper, I believe that the authors overpromise in these sections. The utility of concepts of information bottleneck, MLOps, UQ, training dynamics etc. are quite shallow. Neither are they explained with relation to the ultimate application (a supervised prediction of whether training runs are unstable, forgetting, healthy etc.), nor are they compared in the baselines. Some of the word choices (training *pathologies* in line 72, performance degrades *gracefully* in line 400), analogies (line 54) etc. are strange. This leads me to believe that the work uses LLMs more often than acknowledged in the appendix. I might be wrong on the LLM usage, and if so correct me. However, that does not deflect from my comments regarding the lack of clarity.

**Positioning the work**: The primary reason for my rating is that the paper does not position itself correctly. The authors mention *reconceptualization* and use training dynamics (the meaning of which is slightly different in literature [1][2]) and activations and gradient evolution (well researched topics for OOD, anomaly detection [3][4]) to discuss the novelty in the paper as compared to scalar signals (loss functions). Gradient comparison against loss is also shown in literature [3]. The difference from the listed works is that the paper does the diagnosis on training runs rather than training data. But the paper fails to make this clear anywhere and I only understood it in Section 4. Additionally, I don't see a reason why existing works' features cannot be applied in the way that the authors are applying. Infact some of them are used within an active learning framework [5]. I believe that construction of the data factory in the proposed framework is excellent. But the paper makes no attempt to position itself with relevance to any community. Moreover, as I mention this below, details regarding the dataset construction are relegated to the appendix.

**Terminologies**: The authors utilize a constrained definition of causality where only the previous time series inputs can influence future outputs. However, existing literature on causality has a more generic interventionist definition. The definition is not clarified anywhere. The writing can be tightened and can concentrate more on the technical aspects of the paper.    

**Reliance on appendix**: I felt like the most relevant and interesting details (for the data factory) are relegated to the appendix. The main paper can be substantially rewritten to remove all the unnecessary descriptions. It can concentrate on constructing the labeled healthy and unhealthy training sets and concentrate on the choices that the authors have to make to construct this labeling.

**Ablation studies:** 

1. The authors suggest that the plausible default weighting of false alarm cost, missed detection cost, and lead-time award is (1, 10, -0.5). Is this not mere assumption? At minimum, there needs to be an ablation study here. 

2. There is no reasons, ablations, or empirical results provided for extremely important choices that are made for the construction of the dataset (ex: a run is labeled as catastrophic forgetting if accuracy drop is > 0.3 over a 10-epoch window. Why 0.3 and 10 epochs?)

**Technical soundness:** 

1. Several anomalies are generated based on a predefined scenarios of training runs. A priority ordering is defined (line 613) - *Instability > Catastrophic Forgetting > Concept Bias > Overfitting > Healthy. This hierarchy prioritizes acute, systemic failures
over more subtle or late-stage ones*. How did the authors come to this conclusion? Based on effect on test data? If so which of the designed metrics? The paper makes a very large set of such assumptions and will benefit by looking into the details of a few (or one) of such assumptions. As of now, the paper seems quite shallow.

2. The information bottleneck and DPI analyses are, at best, not surprising. No new bounds or insights are provided to showcase when the proposed framework must be used in lieu of the loss function. Saying that the loss is deterministic (when we know that ultimately loss is calculated from the internal states) does not showcase much.

[1] Achille et al. "Critical learning periods in deep neural networks."

[2] Schneider et al. "Understanding and leveraging the learning phases of neural networks." 

[3] Kwon, Gukyeong, et al. "Backpropagated gradient representations for anomaly detection." European conference on computer vision. Cham: Springer International Publishing, 2020.

[4] Lee, Jinsol, et al. "Probing the purview of neural networks via gradient analysis." IEEE Access 11 (2023): 32716-32732.

[5] Benkert, Ryan, et al. "Gaussian switch sampling: a second-order approach to active learning." IEEE Transactions on Artificial Intelligence 5.1 (2023): 38-50.

### Questions
Please see the Weakness section. I have listed a few choices made by the authors that are not well substantiated, either empirically or theoretically.

1. Usage of LLMs for more than typos and grammar checks

2. How did the authors define the priority list?

3. How were the choices for weighting of false alarm cost, missed detection cost, and lead-time award made?

4. How were the choices for anomaly construction made?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
A framework is proposed for reconceptualision neural network training as a high-dimensional spatiotemporal signal. By employing masked autoencoding on internal activations and gradients, a vision-based diagnostician is pretrained to perform open-set classification of training failures in real time, adhering to strict causal constraints. This approach achieves earlier and more reliable detection than conventional scalar-curve or generic video-based baselines across a diverse range of unseen models, datasets, and optimizers.

### Strengths
Strengths
1.	This paper recasts the neural network training monitor from scalar loss to monitoring a high-dimensional spatio-temporal process, which is innovative and conceptually powerful.
2.	The use of the Data Processing Inequality to justify information asymmetry between scalar loss and internal states is elegant and well-motivated.
3.	Evaluation spans architectures (CNNs, Transformers, MLP-Mixers), datasets (CIFAR, Tiny-ImageNet, WikiText), and optimizers (SGD, AdamW, Lion, Adafactor).

### Weaknesses
1.	One major concern is the computational and storage overhead of capturing and storing layer activations and gradients every 50 training steps across multiple layers. This process is inherently expensive and scales poorly for large models, where training is already highly resource-intensive. As a result, the proposed diagnostic framework may be impractical for real-world monitoring of large-scale models. The paper lacks a detailed efficiency, applicability, and scalability analysis, for example, measurements of runtime overhead, memory usage, or the impact on training throughput. A quantitative study of these aspects is essential to assess whether the proposed method can realistically be integrated into large-model training pipelines.
2.	Anomaly labels are generated using post-hoc privileged information (validation loss, optimizer states, etc). This labeling procedure risks encoding biases that may not generalize to real-world online settings.
3.	While the paper reports model GFLOPs, it omits key practical metrics such as end-to-end training time and additional storage or I/O overhead introduced by the diagnostic system. These are essential for assessing the real-world applicability and scalability of the proposed method.
4.	The proposed approach relies on capturing both gradients and activations for diagnosis, but the paper does not analyze their individual contributions. An ablation study using only gradients or only activations would clarify how each signal impacts diagnostic performance. This would help determine whether both modalities are necessary.
5.	The evidential head’s “Unknown” threshold is fixed and tuned on validation data. No theoretical justification or adaptive mechanism is proposed. This could lead to inconsistent behavior under distribution shifts.
6.	Direct comparison with the architectures (TimeSformer, ViViT) that TEVID builds upon is expected. Without these comparisons, it’s unclear whether TEVID’s superior performance arises from novel dynamics modeling or simply from using a stronger backbone and richer input signal.
7.	The paper fixes the causal window size to 10 frames but does not justify this choice. A window size sensitivity analysis is needed to understand how performance varies with different window sizes.
8.	The paper captures the states after every 50 steps. It would be important to evaluate the impact of capturing states more or less frequently (e.g., every 25 or 100 steps) on both performance and efficiency.
9.	The masking ratio of 90% iused in DYNAMICS-MAE is presented without empirical justification; an ablation over different masking ratios would help validate this design choice.
10.	As the paper is focused on anomaly detection application, performance on image based anomaly detection datasets such as MVTec AD, VisA, and video based anomaly detection such as UCFCrime and XD-Violence  would be required.

### Questions
Please respond to the queries asked in weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
