# Adversarial Robustness of Continuous Time Dynamic Graphs

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Real-world relations are dynamic and often modeled as temporal graphs, making Temporal Graph Neural Networks (TGNNs) crucial for applications like fraud detection, cybersecurity, and social network analysis. However, our study reveals critical vulnerabilities in these models through three types of adversarial attacks: structural, contextual, and temporal perturbations. We introduce Temporally-aware Randomized Block Coordinate Descent (TR-BCD), a novel gradient-based evasion attack framework for continuous-time dynamic graphs. Unlike previous approaches that rely on heuristics or require training data access, TR-BCD optimizes adversarial edge selection through continuous relaxation while maintaining realistic temporal patterns. Through extensive experiments on six temporal networks, we demonstrate that TGNNs are highly vulnerable to TR-BCD attacks, reducing Mean Reciprocal Rank (MRR) by up to 53% while perturbing only 5% of edges. Our attacks are highly effective against state-of-the-art models, including TGN and TNCN, highlighting the importance of studying adversarial robustness for temporal graph learning methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper investigates the adversarial robustness of Temporal Graph Neural Networks (TGNNs) in the continuous-time dynamic graph (CTDG) setting. The authors introduce TR-BCD (Temporally-aware Randomized Block Coordinate Descent), a novel gradient-based evasion attack that perturbs structural, contextual, and temporal dimensions of a temporal graph during inference without modifying training data. TR-BCD greedily injects adversarial edges over time using randomized block coordinate descent to keep the optimization scalable. Experiments on six real-world temporal graph benchmarks demonstrate that the proposed attack can reduce Mean Reciprocal Rank (MRR) by up to 53% while perturbing only 5% of edges. The paper further analyzes the attack’s stealthiness using anomaly detection (SPOTLIGHT) and finds TR-BCD to be both effective and evasive.

### Strengths
Strengths:
* Novel and timely topic: This is one of the first works systematically studying adversarial evasion attacks on continuous-time dynamic graphs, a problem of growing relevance for fraud detection, cybersecurity, and temporal recommendation systems.
* Methodological soundness: The TR-BCD algorithm is clearly formulated, combining continuous relaxation, randomized block coordinate descent, and temporal consistency constraints.
* Strong empirical results: Extensive experiments on six datasets and two state-of-the-art TGNNs (TGN, TNCN) convincingly demonstrate the vulnerability of temporal graph models.
* Practical considerations: The discussion on memory efficiency, time complexity, and unnoticeability constraints makes the work well-grounded.
* Evasion realism: The use of anomaly detection evaluation adds credibility by showing that TR-BCD attacks can remain stealthy.

### Weaknesses
Weaknesses:

* Limited defense discussion: The paper focuses entirely on the attack side; exploring or even briefly analyzing potential defenses (e.g., adversarial training, temporal regularization) would make the study more complete.

* Comparative baselines: Although MemStranding and heuristic attacks are included, more recent or stronger gradient-based baselines (e.g., temporal variants of PGD or PR-BCD) could strengthen the evaluation.

* Sensitivity analysis: The paper could analyze how performance varies with hyperparameters like block size, time perturbation variance, or contextual budget.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

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
This work presents Temporally-aware Randomized Block Coordinate Descent (TR-BCD), a novel gradient-based evasion attack tailored for continuous-time dynamic graphs. TR-BCD formulates adversarial edge selection via a continuous relaxation and optimizes it with temporally aware block coordinate updates that preserve realistic timing patterns. Experiments show that TR-BCD substantially degrades the performance of temporal graph neural networks (TGNNs), demonstrating its effectiveness as a targeted evasion strategy on dynamic graphs.

### Strengths
1. The paper discusses real-world scenarios where continuous-time dynamic graphs commonly occur, underscoring the practical relevance and significance of the proposed framework.

2. It focuses on adversarial attacks against Temporal Graph Neural Networks (TGNNs), aiming to assess and enhance their robustness for real-world applications.

3. The proposed TR-BCD framework optimizes adversarial edge selection through continuous relaxation and directly optimizes the adversarial objective during inference.

### Weaknesses
1. The proposed method relies on a relatively standard Randomized Block Coordinate Descent framework without introducing a sufficiently innovative or compelling design.
2. The paper lacks a comprehensive discussion of the method’s limitations, particularly regarding performance variations across different TGNN architectures and graph data characteristics.

### Questions
1. The authors do not provide a comprehensive comparison of TR-BCD with other existing adversarial attacks on TGNNs. It is recommended to include comparisons with more recent baseline methods. 

2. The experimental evaluation is limited to small datasets. The authors should consider testing TR-BCD on larger-scale datasets to demonstrate its generalizability and effectiveness in real-world applications.

3. The study evaluates TR-BCD using only a limited set of TGNN models. Recent advances have introduced more powerful TGNNs, such as ROLAND [1]; including these models in the evaluation would strengthen the paper.

4. The experiments only assess TR-BCD against raw TGNNs without incorporating existing GNN defense mechanisms. It would be valuable to test TR-BCD against TGNNs equipped with defense strategies to evaluate its robustness under defended settings.

5. How do the authors ensure the unnoticeability of perturbations in continuous-time dynamic graphs (CTDGs), given that up to 5% of edges are added? Please clarify how such perturbations remain realistic and imperceptible.

6. Please report the runtime or computational cost of TR-BCD to better understand its efficiency compared to other attack methods.

7. The results show that TR-BCD causes more than 50% performance degradation on some datasets but less than 10% on some datasets. Could the authors explain the factors contributing to this large performance variation?

8. In the TGNNs victim model, combining all types of perturbations leads to lower attack effectiveness than using only structural perturbations (see Table 5 in the Appendix). This suggests that integrating temporal and contextual perturbations may reduce TR-BCD’s performance. Could the authors elaborate on the reason for this behavior?

9. Table 6 indicates that edge deletion has a stronger impact than edge addition, which contrasts with typical adversarial attack findings where edge addition tends to be more influential. Could the authors explain why TR-BCD exhibits this reverse effect?


[1] You, Jiaxuan, Tianyu Du, and Jure Leskovec. "ROLAND: graph learning framework for dynamic graphs." Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2022.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the adversarial robustness of Temporal Graph Neural Networks (TGNNs) on Continuous-Time Dynamic Graphs (CTDGs). The authors propose Temporally-aware Randomized Block Coordinate Descent (TR-BCD), an evasion (test-time) attack that optimizes adversarial edge insertions through gradient-based continuous relaxation. Additionally, TR-BCD introduces temporal consistency by modeling timestamps with Gaussian noise, enabling it to craft realistic, temporally coherent perturbations. The attack considers three perturbation types—structural, contextual, and temporal—and is evaluated across six datasets (Wikipedia, Reddit, MOOC, Enron, etc.) and two TGNNs (TGN, TNCN). Results show that TR-BCD significantly reduces performance (up to 53% MRR drop with 5% perturbation budget), outperforming heuristic baselines and prior TGNN-specific attacks.

### Strengths
The paper has the following strengths:
- Robustness of TGNNs under adversarial conditions remains a pressing and underexplored area.
- The attack’s optimization objective, constraints, and procedures are clearly presented.
- Evaluations span multiple datasets and perturbation types (structural, contextual, temporal), and provide decent experimental support.

### Weaknesses
The paper has the following weaknesses:
- The attack model is not explicitly stated, making it difficult to gauge the attack’s practicality or compare it with prior white-box or black-box works.
- Only two TGNNs (TGN and TNCN) are evaluated. Broader coverage would better support the general claims of effectiveness.
- The paper compares the proposed method with limited state-of-the-art GNN attacks.
- The method is an adaptation of existing coordinate-descent GNN attacks, with the primary novelty being timestamp regularization. While meaningful, it sounds incremental.

### Questions
This paper investigates the adversarial robustness of Temporal Graph Neural Networks (TGNNs) in continuous-time dynamic graphs (CTDGs). The authors propose Temporally-aware Randomized Block Coordinate Descent (TR-BCD), an evasion attack that extends gradient-based GNN perturbations to the temporal domain. TR-BCD jointly optimizes structural, contextual, and temporal perturbations while maintaining time consistency via Gaussian timestamp modeling. Experiments on six datasets (Wikipedia, Reddit, MOOC, Enron, etc.) and two TGNN architectures (TGN, TNCN) demonstrate significant performance degradation compared to prior TGNN attacks.

However, a few questions may help clarify the generalizability and novelty of the proposed approach:

1. Could you explicitly describe the attacker’s knowledge assumptions? For example, does TR-BCD assume access to model parameters, gradients, or node embeddings (i.e., a white-box setting), or does it operate under a limited or black-box setting? If the attack assumes a white-box setup, could you discuss how it might transfer or adapt to a more restricted setting (e.g., black-box or limited-feedback environments)

2. The evaluation includes TGN and TNCN, which share similar memory-update structures. Have you considered testing TR-BCD on other representative TGNNs, such as DyRep, JODIE, or ROLAND, as discussed in prior work [1]?

3. Many of these static graph attacks, such as TDGIA[2], could be temporally adapted with minor modifications. Have you attempted such adaptations, or can you discuss why they may not be directly applicable to the CTDG setting?

4. Beyond introducing timestamp regularization, how does TR-BCD fundamentally differ from prior coordinate-descent-based GNN attacks? In what way does temporal smoothness change the optimization landscape or attack transferability compared to static or snapshot-based graph settings?

5. The paper demonstrates strong attack performance in standard settings, but it is unclear how TR-BCD behaves when common defenses are applied. Could the authors discuss whether they evaluated TR-BCD under known GNN or TGNN defense strategies? If not, do they expect the attack to remain effective, or would temporal regularization make it more vulnerable to such countermeasures?

[1] Dai, Yue, et al. "MemFreezing: A Novel Adversarial Attack on Temporal Graph Neural Networks under Limited Future Knowledge." Forty-second International Conference on Machine Learning.

[2] Zou, Xu, et al. "Tdgia: Effective injection attacks on graph neural networks." Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper propose a Randomized Block Coordinate Descent method to attack white-box TGNNs. The attacker applies gradient decent based method to increase the target model loss and pick components with higher gradients greedily. To reduce the memory cost of such optimization on whole edges, the method is further adapted by choosing a small randomized fraction of edges as candidates to cut down the range. To better obtain the candidates the author further greedily incorporate historical negative edges. Experiments across several datasets and victim models validates the performance of the method.

### Strengths
**S1.** The paper study the vulnerabilities of TGNN models against adversarial attack, which is a not well explored topic but has many potential impact since TGNN relates to many practical applications.

**S2.** Abundant experiments are done to show the effectiveness and practicability. The method is not only evaluated decreased performance of the method compared with several baselines, but also show deceiving ability on mitigating anomaly detection method and ablation study on different components and different budget. There is also code provided to support its reliability. 

**S3.** The randomized entry samples cut down the memory complexity , which makes the method more friendly to device requirement.

### Weaknesses
**W1.** The practicability of the proposed method is doubtful. The proposed attack is a gradient based attack with knowledge of both the white-box victim model and all historical temporary graphs, both of which seems not accessible in real world application. 

**W2.** The technique contribution is somewhat limited. The proposed Randomized Block Coordinate Descent is also a simple adaptation of typical gradient descent optimization over edges with so called randomized samples, mainly relying on a greedy idea of highly picking historical negative edges. There's no deduction nor theoretical guarantees on the proposed method with only considerable optimization efficiency due to the small size of random sampled edge candidates.  

**W3.** The presentation of the paper should be improved.

3.1. The methodology of the  proposed method are fully expressed in textual description throughout section 4 without formulations for illustrate. Considering there's actually large space remaining in the paper, simply copying expression from the algorithm flow would solve. 

3.2. The memory mechanism is lacked of illustration. There is a introduced mechanism of model memory that help the TGN to encode temporal graph. Since this is a mechanism that not held widely by general GNN or other ML models, its definition and formally illustration at problem statement is needed since it tightly relates to the designed algorithm. An only reference in the related works would make the reader ignore and feel confused when encountering it in latter section 4.

3.3. In the technique design a lot of mechanisms are proposed to tackle the memory problem of the attack process, while this is not clearly stated in motivation and contribution aspect in introduction session. Such technique challenge and solution should be briefly mentioned in the section 1 so reader could expect content related.

### Questions
Please see weakness.

### Soundness
3

### Presentation
2

### Contribution
2
