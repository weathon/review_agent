# Quan-dorcet: Tournament-Based One-vs-One Quantum Classification for Robust Single-Shot Inference

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Quantum machine learning (QML) promises powerful classification capabilities, but suffers from fragile output encodings and high sampling demands—especially in multiclass settings. Traditional schemes such as one-hot and binary encoding either produce interpretable outputs too rarely or require many shots to achieve reliable predictions. We propose a decision aggregation framework for quantum multiclass classification based on round-robin tournament scoring. Each output qubit represents a binary comparison between class pairs, and the final prediction is determined by majority wins—yielding a Condorcet-style winner when one exists. This structure improves both the resolvability and accuracy of single-shot predictions, outperforming standard encodings under few-shot conditions. Our method retains global entanglement while localizing decision tasks, enabling interpretable inference that remains reliable under intrinsic quantum randomness, without sacrificing expressivity. Empirical results show that this approach achieves high accuracy and interpretability with significantly fewer measurements, suggesting a promising direction for future quantum classifiers.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a novel output encoding technique for QML in multi-class classification tasks. Specifically, the authors propose a decision aggregation framework based on round-robin tournament scoring, offering an alternative to traditional encoding schemes such as Gray and binary encoding. A key feature of the proposed approach is its reliance on one-shot encoding, where a single measurement is used to determine the class label, rather than computing expectations over multiple shots. This design aligns well with practical constraints in QML, contributing to more efficient inference.
In addition to the theoretical framework, the paper presents empirical evaluations across multiple datasets, demonstrating that the proposed method consistently outperforms existing encoding strategies.

### Strengths
Reducing the sample complexity or shot complexity of QML for multiclass problems is an interesting and important problem that this paper addresses. The numerical experiments clearly demonstrate that the proposed method outperforms existing approaches for output encoding. Overall, the paper is relatively well written and easy to follow

### Weaknesses
The paper focuses on a single-shot approach, which is nice. But I wonder if a comparison with a few-shot measurements would make sense and give improvements. Essentially, a comprehensive discussion on the number of measurement shots for output encoding seems lacking in this paper.

### Questions
Can you provide any theoretical justification comparing the single shot with the fixed multi-shot approach?

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
4

### Summary
This paper presents a new decision aggregation framework, QUAN-DORCET to address the critical sampling bottleneck and fragile output encodings in multiclass quantum machine learning under limited measurement budgets, especially for single-shot inference. The main contribution is the replacement of traditional global output schemes with a round-robin tournament structure where each output qubit performs a binary comparison between a pair of classes. The final prediction is a Condorcet-style winner determined by majority wins across all pairwise comparisons which theoretically converges to full resolvability as the number of classes increases. The authors also develop a unique, differentiable training method that embeds the pairwise comparisons into a continuous simplex using a symmetric cross-entropy loss and empirically demonstrate that this approach significantly improves both shot resolvability and accuracy compared to baselines in few-shot regimes.

### Strengths
The paper introducing a novel output encoding for VQC based on external political/tournament theory (Condorcet's criterion) to solve the intrinsic quantum problem of low single-shot resolvability. The core innovation is leveraging the statistical robustness of the tournament structure, which, unlike one-hot encoding does not suffer from exponentially vanishing resolvability and, unlike binary encoding is less susceptible to single-bit noise causing large semantic misclassifications. The quality of the work is substantiated by the comprehensive theoretical analysis (Section 3.2) and the technical development of a non-trivial, end-to-end differentiable training procedure that successfully maps the $K(K-1)/2$ binary outputs to a continuous simplex for gradient-based optimization. The writing is clear and provides sound motivation explicitly highlighting the trade-offs of all incumbent encoding methods in Table 1.

### Weaknesses
The most significant weakness is the fundamental issue of quadratic scaling in the required number of qubits as $K$ classes require $K(K-1)/2$ output qubits meaning the approach is constrained to small class counts (e.g., $K\le6$ in the experiments) and cannot scale to large classification problems on near-term hardware. While the method aims for robustness on real-world devices, all model training and performance evaluations are conducted exclusively under noiseless simulation. Also, a critical and missing piece of the empirical quality is a systematic study of how the Condorcet-style aggregation handles realistic hardware noise (e.g., bit-flip or depolarizing channels) in comparison to the binary/Gray codes it claims to outperform in terms of noise robustness. Overall, the paper should address the potential for Condorcet cycles (tournament paradox) in the measured results as the existence and frequency of these cycles would determine the fundamental practical limit of the method's resolvability under non-ideal (noisy) conditions.

### Questions
A key question for the authors concerns the empirical analysis of the Condorcet paradox,

a) Could the authors provide data on the frequency of non-unique winners (cycles or ties) in the tournament aggregation? As this is a vital component of the non-resolvable shots and have they considered using a tie-breaking rule or a ranking method (like the Schulze method) to maximize the practical resolvability? 

b) Given that the robustness to bit-level noise is a central claim against binary encodings, the authors should perform an actionable study by including results for all encodings under a simulated hardware noise model (e.g., $1\%$ depolarizing noise on the output measurement qubits) to confirm the robustness advantage in the setting where it is most needed. As a suggestion to overcome the $\mathcal{O}(K^2)$ qubit requirement, have the authors explored an alternative sparse tournament structure, such as a hierarchical elimination or a Tournament-of-Champions (ToC) scheme, and can they provide an analysis on the trade-off between the decreased resource cost and the expected decrease in single-shot resolvability?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The papr introduces "Quan-dorcet" a round-robin tournament aggregation technique for multiclass quantum classification. Quan-dorcet targets the fragility of existing output encodings in QML, specifically under single.few-shot measurement constraints. In the following I provide the strengths and weaknesses of the paper.

### Strengths
1. **Addressing an important bottleneck:** Authors address an important bottleneck in current QML architectures which is poor reliability and the requirement of high shot.

2. **The encoding** The tournament is well motivated. Going beyond ad hoc output mappings and leveraging robust statistical voting mechanisms.

3. **Comparison** Empirical comparisons against standard output encodings (one-hot, binary, Gray) on quantum circuits for MNIST digits/fashion datasets are extensive, covering multiple circuit types and parameter regimes.

4. **Code availability** The authors have publicly released their code, demonstrating a commitment to open-source practices and enabling reproducibility.

### Weaknesses
## **Quadratic resource scaling:** 
The method requires a number of quantum wires/qubits that scales quadratically with the number of classes: for K classes ~K(K-1)/2, pairwise comparisons are encoded. This makes the approach impractical for even moderate sized output spaces and restricts applicability to small problems. The text neither addresses this limitation nor empirical studies on the maximal class count achievable on actual hardware.

## **Lacking QPU execution:** 
All the main experiments are performed under noiseless simulation; only limited inference ablations are reported using noise models from IBM Qiskit. There is no demonstration of end-to-end training or inference on real quantum processors. The authors should provide a detail characterization of the Quan-dorce's robustness under device noise and decoherence effects.

## **Scalability:** 
The computational cost is high around "100 kCPU-hours". As stated by the authors, the approach necessitated "unforeseen compute limitations" that prevented reporting results for all circuit blocks and larger K in time for the submission. I believe this demonstrates not only scalability barriers for NISQ devices but also for classical simulation pipelines.

## **Lacking analysis of class imbalance and semantic overlap:** 
The framework does not explore: 
- Class imbalance, where some classes are much less represented than others, can severely impair prediction accuracy because minority classes may rarely win pairwise matchup. I believe this can lead to majority-win bias and poor generalization. 
- Semantic overlap, where different classes share similar characteristics, can lead to  ambiguous or non-separable boundaries.

## **Clarity:** 
The manuscript is technically sound but sometimes impenetrable to non-specialists.  Some claims about tournament theory and Condorcet aggregation would benefit from clearer intuitive explanations and more concrete worked examples. 

## **Lack of references** 
In introduction: 
- The second sentence does not provide any reference. Such as the claim that PQC used for encoding input data. 
- The term "tunable gate operations" is vague. 
- The sentence `practical implementations face a significant challenge in the form of a sampling bottleneck` neither provide any justification why this bottleneck appears nor it provide any relevant information. 
- The claim `the proportion of resolvable outputs.. vanishes exponentially with the number of classes, making inference increasingly unreliable` is made without any references or explanation. I encourage the author should provide more references and explanations in the introduction.

### Questions
As noted in the weaknesses above, I would like to pose the following questions and suggestions to the authors:

## **Resource Scaling:**  
   - Can you propose, analyze, or empirically test methods to reduce the quadratic scaling of qubit requirements? 
- For example, could some pairwise decisions be encoded or aggregated classically, or can hybrid output encodings balance accuracy with qubit economy? 
- What is the largest class count (K) for which the method remains practical on current or near-term hardware?

## **Quantum hardware execution:**  
   - Do you have plans to implement the tournament method on real quantum processors, and if so, what are the expected resource bottlenecks and noise impacts? 
- Could you provide results, or at least simulated characterizations, for your method under realistic device noise and decoherence, especially regarding accuracy and resolvability?

## **Scalability:**  
   - Can you clarify the extent of computational resources required across architectures, and suggest optimizations to the classical simulation pipeline? 
- What are the directions for scaling up your method on either quantum or classical backends?

## **Class Imbalance and semantic overlap:**  
   - How does your framework handle class imbalance, where some classes are underrepresented and risk being overlooked in majority voting? 
- Could you design experiments with imbalanced or overlapping class distributions, and report rates of ties and ambiguous predictions (such as `Condorcet cycles`)?

## **Clarity:**  
   Can you expand the descriptions of tournament theory and Condorcet mechanisms with visual example (e.g., for a small 3-class case)?

## **References:**  
   Can you add more references to support the foundational claims in the introduction, especially regarding PQC input encoding, sampling bottlenecks, and exponential validity decay with the number of classes?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a novel output encoding framework, "Quan-Dorcet," for quantum multiclass classification. The method is based on pairwise round-robin tournament comparisons, aiming to improve the accuracy and "resolvability" of single-shot and few-shot inference. The authors introduce "shot resolvability" as a key metric and demonstrate through simulation that their method outperforms one-hot and binary encodings for a small number of classes (K≤6).

### Strengths
The paper addresses the challenge of multiclass classification in quantum machine learning (QML), particularly the issues of fragile output encodings and high sampling demands under few-shot or single-shot inference. The authors propose a tournament-based one-vs-one encoding scheme wherein each qubit corresponds to a binary comparison between a pair of classes; class prediction is obtained via a round-robin vote (Condorcet winner) across all pairwise outcomes. This decouples the multiclass output into many simpler binary decisions embedded within a shared entangled quantum state, which the authors argue improves “shot resolvability” (probability a single measurement yields a valid class) and accuracy under measurement-limited settings. Empirical results demonstrate that the scheme outperforms standard one-hot or binary encoding architectures in few-shot/single-shot regimes, suggesting this paradigm as a robust path for quantum classifiers with fewer measurements.

### Weaknesses
Scalability Problem: This is the most critical issue. As the authors admit in Section 5 ("Limitations"), the required number of qubits (wires) $W$ scales quadratically with the number of classes $K$ ($W = \binom{K}{2} = K(K-1)/2$).

Impact on Practicality: This makes the method practically infeasible. While binary encoding requires only $\lceil \log_2 K \rceil$ qubits, the proposed method requires 45 qubits for $K=10$ and nearly 5,000 qubits for $K=100$. This is unattainable on near-term (or even mid-term) quantum hardware.

Poor Trade-off: The paper claims to solve the "sampling bottleneck" but introduces a far more severe "qubit resource bottleneck." This trade-off (exchanging sampling efficiency for an exponentially increasing qubit requirement) is unacceptable in practice.

Incomplete Experiments: As noted in the footnotes of Table 2 and Table 3, the authors admit that "Due to unforeseen compute limitations," the results for $K=6$ are incomplete, tested on only one circuit, and that the rest "will be ready by rebuttal period." This indicates the submission is incomplete work.

### Questions
While this contribution is creative and potentially impactful, several limitations and open concerns remain:

1. Limited demonstration of practical quantum advantage

The experiments, while promising, appear to be simulation-based (no demonstrated real quantum hardware results). Thus, real-world factors (noise, decoherence, measurement error, circuit overhead) are not fully addressed.

The improvement in “shot resolvability” is compelling, but it remains unclear how that metric translates into end-to-end system performance, especially when scaling to larger class sets or higher dimensional inputs.

2. Scalability issues not thoroughly addressed

For  𝐾 classes, one-vs-one induces  K*(K-1)/2 binary comparisons/qubits (or circuits). The paper should more clearly analyse the resource scaling (qubits, gates, measurement overhead) for large 𝐾, and whether the tournament cost outweighs encoding benefits.

The assumption that a unique Condorcet winner will emerge reliably may break down in practice under noisy or ambiguous class boundaries; the authors should discuss scenarios where majority voting may fail or require tie-break strategies.

3. Baseline comparisons and alternative encodings

Although one-hot and binary encodings are compared, more recent and sophisticated quantum multiclass classifier encoding schemes (e.g., amplitude encoding, mixed‐state discriminators) are not deeply benchmarked. Without comparison to strong state-of-the-art quantum multiclass methods, the improvement claim is less convincing.

Moreover, many classical multiclass frameworks (ensemble binary classifiers, one-vs-one classical SVMs) employ similar architectural breakdowns; a comparison of quantum vs classical one-vs-one paradigms would strengthen significance.

4. Theoretical justification of single-shot improvements

The notion of “shot resolvability” is interesting but currently heuristic. A deeper theoretical analysis of how measuring fewer shots yields reliable class output (given quantum measurement statistics, error rates) would improve confidence.

Also, the impact of entanglement and shared statewide encoding on error propagation among the binary comparators is not fully addressed.

5. Application and realism of datasets/inputs

The datasets used, while unspecified here in detail, likely involve small-scale toy problems under simulation. It remains unclear how the method performs on large-input real-world classification tasks (e.g., image datasets with many classes and high dimensionality).

The authors should discuss how measurement budget, class imbalance, noise, and decoherence would influence performance in near-term quantum devices.

**Relevant References for Inclusion**

To strengthen the literature context and demonstrate awareness of related work, the authors should consider citing the following:

Du, Y., Yang, Y., Hsieh, M.-H., & Tao, D. (2023). Problem-Dependent Power of Quantum Neural Networks on Multi-Class Classification. Phys. Rev. Lett. 131, 140601. 

Bokhan, D., Mastiukova, A. S., Boev, A. S., Trubnikov, D. N., & Fedorov, A. K. (2022). Multiclass classification using quantum convolutional neural networks with hybrid quantum-classical learning. arXiv:2203.15368. 

Useche, D. H., Quiroga-Sandoval, S., Molina, S. L., Vargas-Calderón, V., Ardila-García, J. E., & González, F. A. (2025). Quantum generative classification with mixed states. arXiv:2502.19970. 

Delilbasic, A., Le Saux, B., Riedel, M., Michielsen, K., & Cavallaro, G. (2023). A Single-Step Multiclass SVM based on Quantum Annealing for Remote Sensing Data Classification. arXiv:2303.11705. 

Cruzeiro, E. Z., De Mol, C., Massar, S., & Pironio, S. (2023). Quantum-inspired classification based on quantum state discrimination. arXiv:2303.15353. 

Riaz, F., Abdulla, S., Suzuki, H., Ganguly, S., Deo, R. C., & Hopkins, S. (2023). Accurate Image Multi-Class Classification Neural Network Approaches including quantum variations. Sensors, 23(5):2753.

### Soundness
2

### Presentation
2

### Contribution
2
