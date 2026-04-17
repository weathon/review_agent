# CocoRNA: Collective RNA Design with Cooperative Multi-agent Reinforcement Learning

- Decision: Reject
- Scores: 4, 2, 8, 4

## Abstract
Designing RNA sequences that reliably fold into specific secondary structures is essential for understanding their biological functions but remains a challenging computational problem. We propose CocoRNA, a cooperative multi-agent reinforcement learning framework for RNA inverse design. CocoRNA simplifies the design task by decomposing it into smaller sub-problems, each solved collaboratively by multiple agents. This approach reduces the complexity of the problem and improves the exploration of design policies. During training, a centralized critic uses global structural information to guide the agents, enabling them to jointly optimize their design strategies. As a result, CocoRNA learns high-quality RNA design policies that generalize effectively to unseen structures without additional training. Experiments on the Rfam dataset demonstrate that CocoRNA substantially outperforms state-of-the-art methods in both success rate and design speed. Further experiments on other biological sequence design tasks highlight the effectiveness and broad potential of CocoRNA for complex design tasks. Visualization examples are available on https://sites.google.com/view/cocorna/home.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes CocoRNA, a multi-agent cooperative framework for secondary structure based RNA design. The authors subdivide an RNA structure in dot-bracket format into several sub-tasks, each solved by an individual agent, while using a global critic network to ensure that the design considers the overall structure during design. The method is evaluated on the Rfam dataset and compared to five different competitors. On this benchmark CocoRNA outperforms all other approaches even when using a much stricter time limit. Furthermore, CocoRNA is also adapted and evaluated on other design tasks provided with the Design-Bench benchmark with promising results. In an ablation study, the authors confirm the gains of a multi-agent system compared to a single agent as well as their newly introduced SAE strategy for early training.

### Strengths
- Compared to existing RL approaches for secondary structure based RNA design, the multi-agent approach appears interesting and makes sense.
- The reported performance on the Rfam dataset is strong.
- The decomposition of the problem into sub-tasks while keeping a global measure of performance is a good idea.

### Weaknesses
1. I’m a bit concerned about the timeliness of the approach. Specifically, using the dot-bracket notation is a clear limitation and excludes design for pseudoknots, or other pairing schemes.
2. When using the data from [1], why do the authors not evaluate against [1]? In the paper, it seems that libLEARNA is superior compared to Meta-LEARNA-Adapt on the task of RNA design for nested structures. Also RNAPond [2] could be an interesting competitor here. I think there are also further competitors that could be evaluated, e.g. [3,4] but it might be too much to run all these in the limited amount of time during rebuttal.
3. One motivation in the introduction is about delayed rewards and misleading auxiliaries. However, the proposed reward formulation seems to match the formulation used in [5] quite exactly. Or am I missing something?
4. While the authors show strong results on the Rfam dataset, the structural diversity might be rather low. A good additional benchmark could be the Eterna100 (version 2) [6] in this case. While not the best benchmark for generative approaches, I would recommend evaluations on it here, since it specifically covers corner cases of the thermodynamic model of RNAfold, which could provide further insights into CocoRNA’s performance. 
5. The authors provide visual examples on an external website. Looking at the designs, it seems that most sequences rarely contain ‘A’ nucleotides. At least there is a clear bias towards G, C, and surprisingly U. The authors should at least mention this somewhere. But I think this could also be a general limitation of the method, since the authors do not include any measure to avoid bad sequences (GC biases can strongly influence function). In the best case, a desired GC-content of the designed sequences would be included (which was e.g. done in [1] in a very simple way that seemed to work well already).
6. In this regard, the diversity measure might also be a bit misleading since it is only based on the Hamming Distance between the designed sequences. 
7. It would also be interesting to see evaluations based on families. Using RNAfold, it is not too computationally expensive to even fold all sequences in the Rfam database and provide a family based evaluation scheme.

[1] Runge, F., Franke, J., Fertmann, D., Backofen, R., & Hutter, F. (2024). Partial RNA design. Bioinformatics, 40(Supplement_1), i437-i445.

[2] Yao, H. T., Waldispühl, J., Ponty, Y., & Will, S. (2021, April). Taming disruptive base pairs to reconcile positive and negative structural design of RNA. In RECOMB 2021-25th international conference on research in computational molecular biology.

[3] Minuesa, G., Alsina, C., Garcia-Martin, J. A., Oliveros, J. C., & Dotu, I. (2021). MoiRNAiFold: a novel tool for complex in silico RNA design. Nucleic acids research, 49(9), 4934-4943.

[4] Yang, X., Yoshizoe, K., Taneda, A., & Tsuda, K. (2017). RNA inverse folding using Monte Carlo tree search. BMC bioinformatics, 18(1), 468.

[5] Runge, F., Stoll, D., Falkner, S., & Hutter, F. Learning to Design RNA. In International Conference on Learning Representations.

[6] Koodli, R. V., Rudolfs, B., Romano, J., Wayment-Steele, H. K., Dunlap IV, W. A., Eterna Participants, & Das, R. (2021). Redesigning the EteRNA100 for the Vienna 2 folding engine. BioRxiv, 2021-08.

### Questions
1. It seems like the sequence is randomly initialized. Have the authors thought about / tested different initialization methods? SAMFEO for example, uses an initialization with GC pairs and A for unpaired nucleotides if I remember correctly which itself could already solve many tasks without further predictions when using RNAfold. I would guess that this would further improve performance of CocoRNA (although it doesn’t solve the problem with GC-content during design described above).
2. In the plots of Figure 3 (and similarly Figure 10). It seems that CocoRNA’s runtime scales quadratic with sequence length (although being much faster than the other methods). If I remember correctly, linear scaling with sequence length was one of the claims of the original LEARNA paper. Can the authors elaborate on that?
3. While already stated above, what do the authors think about including e.g. pseudoknots in the design? Using dot-bracket notation, this would blow-up the state-space I guess? What about using matrix representations as many recent DL folding engines output L x L matrices like done in the cited work of RNAinformer? 
4. Is there any limitation on sequence length for the designs?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces COCORNA, a cooperative multi-agent reinforcement learning framework for RNA inverse folding, designing RNA sequences that fold into a given secondary structure. The key innovation is task decomposition. The RNA sequence is divided into segments or structural parts. These agents work together under a "CTDE" paradigm, using "MAPPO" for policy optimization and search-augmented exploration. COCORNA achieves a 97.75% success rate on the Rfam dataset, and it significantly outperforming state-of-the-art methods in both accuracy and speed. It also generalizes well to other biological sequence design tasks.

### Strengths
- Here utilizes the multi-agent reinforcement learning framework for RNA inverse folding, offering a fresh perspective on tackling complex biological sequence design problems.
- This work achieves an impressive 97.75% success rate on the Rfam dataset, significantly outperforming existing methods (e.g., LEARNA, SAMFEO, antaRNA) by large margins.
- Besides, provides theoretical convergence guarantees under standard assumptions, enhancing the credibility and understanding of the proposed method.

### Weaknesses
- Rewards signal depends on external tools like ViennaRNA. The rewards can become misleading, if the predicted structure is inaccurate. No uncertainty modeling or robustness to predictor error.
- Here only uses secondary structure as the design target, while 3D structure is more crucial for RNA function in reality. Ignoring it may lead to biologically non-functional designs.
- Theoretical analysis only guarantees local convergence, not global optimality. Like most RL methods, it can get stuck in suboptimal policies, especially in sparse reward setting.

### Questions
- Why did authors choose fixed position-based or structure-based decomposition? Or have explored adaptive or learnable decomposition strategies? Is there exist other decomposition schemes and their impact on the performance?
- Search-augmented exploration improves early training but involves local greedy search. Could this mislead the policy towards short-term improvements?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work targets the inverse design of RNA secondary structures, an important biological sequence design task. The main contribution is the novel application of multi-agent RL to this task. The proposed approach, CocoRNA, achieves an impressive empirical performance.

### Strengths
* RNA design is an important problem to study and an interesting application of RL
* The main strength of the paper is the strong empirical performance in both number of solved sequences and time-to-solution
* Applying RL to RNA design has been explored before, but the application of multi-agent RL is novel and has several advantages. Decomposing the RNA design problem into sub-problems for different agents to solve to reduce state dimensionality and policy spaces is also an interesting idea.
* The authors identify specific shortcomings of current RL strategies and motivate the use of multi-agent RL and the decomposition into sub-tasks well.
* Both success rate and design speed are evaluated
* The ablation study for the multi-agent architecture and proposed exploration heuristic confirm the design choices behind CocoRNA.
* Related methods are covered well and key differences discussed
* Written with a very clear language and structure
* The illustrative Figures 1 & 2 are quite helpful
* The authors cover limitations in Appendix G2. Moving this discussion into the main paper would be good.

### Weaknesses
* Previous work evaluated also on the Eterna100 dataset, why is this omitted here?
* Code to reproduce the experiments or even an algorithm implementation is not provided. Some hyperparameter settings and details on architecture and optimization is provided though.
* Novelty in terms of methodology and problem setting is limited. Due to the strong performance, I still think this work is interesting for the ICLR community to be aware of which methods work well in applications.

### Questions
* "As demonstrated by our experiments in Section 4, existing approaches typically require hours of computation", where do approaches require hours of computation?
* Does the Rfam dataset you use differ from the one used in previous works?
* How were the hyperparameter settings for CocoRNA chosen?
* Table 1 compares CocoRNA with a 30s time limit (the most strict one) to baselines at 30s time limits and above. A plot that shows the #solved samples across different time limits would improve the evaluation and likely underline the strong performance and quick solution time (as seen in Figure 2) of CocoRNA.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes COCORNA, a multi-agent reinforcement learning framework for RNA inverse design. In their proposed method, the task is split into smaller sub-problems handled by multiple agents. A centralized critic coordinates learning, and a search-augmented exploration (SAE) improves early training. The method achieves a 97.75% success rate on Rfam, 70× faster than strong baselines. It also generalizes to other biological sequence design benchmarks.

### Strengths
They proposed the first multi-agent framework for RNA reverse design with solid theoretical foundation and relevant proof.

They demonstrated significant improvements in success rate and speed, supporting their claim with evidence.

Comprehensive ablation study and experimental details are presented and well-organized.

### Weaknesses
Missing comparison with new foundation models (e.g., RNAinformer).

Lacking deeper analysis of unsuccessful design cases.

Limited tests on long sequences (>500 nt).

### Questions
Could you involve more comparison to recent foundation models for RNA reverse design?

For the ~2.25% of structures that COCORNA fails to design within the time limit, could you provide a deeper analysis? What characteristics do these challenging structures share (e.g., length, complexity, specific structural motifs)?

While the cross-oracle validation with RibonanzaNet (94.90% success rate) is impressive, the 2.85 percentage point drop suggests some overfitting to ViennaRNA's MFE predictions. Could you elaborate on: (a) which types of structures show the largest performance gaps between oracles, and (b) potential strategies to make the approach more oracle-agnostic?

### Soundness
3

### Presentation
3

### Contribution
3
