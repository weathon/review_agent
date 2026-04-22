# On the Fragility of Graph Backdoor Defenses: A Robust Strategy via Layer-wise Feature Divergence

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Recent studies have revealed the high susceptibility of GNNs against backdoor attacks, which poses a significant threat to their practical applications. In order to deal with the threats posed by backdoors, a series of targeted defense measures have been proposed, which have effectively alleviated the harm of backdoor attacks to a certain extent. However, do these methods really completely eliminate the threat of backdoors? Inspired by related research in the DNN field, we conduct the first systematic robustness analysis of backdoor defenses in the GNN domain. Our experiments reveal that even fine-tuning the defense model for only five epochs with a small fraction of poisoned data can cause a sharp resurgence in its ASR, indicating that residual backdoor features persist and can be readily reactivated. Recognizing the unique message-passing paradigm in GNNs, we leverage Layer-wise Linear Feature Connectivity (LLFC) to uncover the root cause of this pronounced fragility in current graph backdoor defenses. To enhance the robustness of these defenses, we also propose a novel strategy termed \textbf{Layer-wise Feature Divergence (LFD)}, which forces the defense model to diverge from the original backdoor model by maximizing the distance between their respective layer-wise features during retraining. Extensive experiments demonstrate that LFD significantly enhances the robustness of defense models, achieving state-of-the-art performance in defense capabilities while maintaining high accuracy on clean data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the robustness of graph neural network (GNN) backdoor defenses, which aim to remove or mitigate malicious triggers embedded during training. The authors find that existing defenses, though seemingly effective, are fragile—their attack success rate (ASR) can quickly rebound when the model is fine-tuned even slightly with poisoned data. To explain this, the authors analyze feature-space similarity between purified and backdoored models using Layer-wise Linear Feature Connectivity (LLFC), revealing persistent feature-level correlations. Based on this insight, they propose a novel defense strategy called Layer-wise Feature Divergence (LFD), which explicitly maximizes divergence in layer-wise feature representations between the defense model and the original backdoored model. Experiments on standard datasets (Cora, PubMed, OGB-Arxiv) and several attack settings (UGBA, GTA, DPGBA) show that LFD achieves improved robustness (low ASR and RA-ASR) while maintaining high clean accuracy.

### Strengths
1. Novel and well-motivated problem formulation.   
The authors are the first to systematically evaluate the robustness of GNN backdoor defenses beyond one-shot purification. The observation that existing defenses are easily “reinfected” by minimal fine-tuning is original and practically important.

2. Insightful analysis via LLFC.  
Using LLFC to quantify feature-space similarity provides a fine-grained, interpretable diagnostic for understanding residual backdoor effects—an advancement over prior DNN-based analyses that rely only on parameter-space linear mode connectivity.

3. Principled method (LFD).  
The proposed defense—explicitly encouraging divergence in layer-wise feature representations—is conceptually clear, well-grounded in the LLFC analysis, and relatively easy to integrate with existing training pipelines.

4. Strong empirical evidence.  
Comprehensive experiments across datasets and attacks demonstrate consistent improvements in robustness metrics, with ablation studies showing each loss component’s contribution.

5. Clear writing and reproducibility.  
The manuscript is well-structured, follows ICLR conventions, and provides detailed implementation information with open-source code.

### Weaknesses
1. Limited novelty of defense formulation.  
While the idea of maximizing feature divergence is reasonable, it may be viewed as an extension of existing regularization methods (e.g., feature orthogonalization, representation disentanglement). The theoretical justification of why this ensures durable robustness could be deepened.

2. Restricted attack scope.    
Experiments only consider dirty-label backdoor attacks. The method’s effectiveness against clean-label attacks or more adaptive triggers is untested (acknowledged by the authors).

3. LLFC computation cost and scalability.    
The paper does not discuss the computational overhead of LLFC-based loss during training, especially on large graphs (e.g., OGB datasets). A runtime or memory analysis would strengthen practical relevance.

4. Lack of theoretical analysis or guarantees.  
The defense remains empirically motivated. There is no theoretical argument connecting feature divergence maximization to guaranteed reduction in feature-space backdoor correlation.

5. Missing visualization or qualitative results.   
Some intuitive visualization of feature-space shifts (e.g., via t-SNE or cosine-similarity maps) could make the concept of “layer-wise divergence” more tangible.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the robustness of graph neural network (GNN) backdoor defenses and introduces a new defense strategy termed Layer-wise Feature Divergence (LFD). The authors first show that many existing backdoor defense methods for GNNs, despite appearing successful, remain fragile — as fine-tuning with even a small portion of poisoned data can reactivate hidden backdoor features. They attribute this fragility to residual feature similarity between the purified and the original backdoored models, analyzed via Layer-wise Linear Feature Connectivity (LLFC). To counteract this, the proposed LFD method enforces divergence between layer-wise features of the purified model and the backdoored one, aiming to achieve stronger robustness. The paper provides experimental results on standard datasets and compares against several existing defenses (RIGBD, DShield), reporting superior robustness under retraining attacks.

### Strengths
+ Provides the first empirical study on the robustness of graph backdoor defenses under reactivation attacks.
+ Offers clear empirical evidence that existing defenses can be easily re-poisoned, highlighting an underexplored vulnerability.

### Weaknesses
- Limited novelty — LFD is a direct adaptation of existing LMC ideas from DNNs, without deeper theoretical insights.
- Evaluation lacks breadth and rigor: only small datasets, missing modern or large-scale baselines, and no statistical testing.
- Heuristic method design — hyperparameter choices, loss weighting, and poisoned-node selection are ad-hoc and unexplained.
- Clean-label and black-box attack settings ignored, which weakens practical relevance.

### Questions
1. Is the divergence enforced equally across all layers, or do certain layers (e.g., deeper GCN layers) contribute more to robustness? Would focusing the divergence on high-level representations yield better results?

2. The paper uses LLFC mainly as a visualization tool — is there any quantitative evidence that LLFC distance actually correlates with robustness metrics such as RA-ASR?

3. Can the proposed method generalize beyond dirty-label settings? Specifically, how would it perform under clean-label or federated graph backdoor attacks where trigger visibility and label control are limited?

4. How would LFD handle structural perturbations — for instance, attacks that modify edges rather than node features? Would the same feature divergence principle apply?

5. What is the computational cost of computing layer-wise divergence, especially for deeper GNNs or large-scale datasets?

6. Finally, the core assumption that maximizing feature divergence leads to true backdoor removal seems intuitive but unproven — could it instead just push the model into a different yet still vulnerable region of the feature space?

### Soundness
2

### Presentation
2

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
This paper presents the systematic robustness analysis of graph backdoor defenses, revealing that existing methods only achieve superficial security and can be easily reactivated. The authors propose a novel defense strategy called Layer-wise Feature Divergence (LFD), which enhances robustness by maximizing feature-space divergence from the backdoored model during retraining.

### Strengths
1)	First to systematically investigate the fragility of graph backdoor defenses.
2)	Leverages LLFC to explain vulnerabilities in the feature space.
3)	The paper is well-structured, clearly written, and supported by informative figures and tables.

### Weaknesses
1)	The core of LFD requires a clean, backdoored reference model to maximize feature divergence. This creates a fundamental paradox: in a real-world defense scenario, such a pristine backdoored model is unavailable. If the defender already possesses it, why not use it directly for training? 

2)	The analysis of LLFC remains superficial, with limited theoretical contribution.

3)	The vulnerabilities of the poisoning node detection module have not been adequately discussed.

### Questions
1、It is unclear whether the method proposed by the author is applicable to large-scale graph data, such as the OGB-arxiv dataset mentioned in Section 6, as it was not demonstrated in the experiments.

2、From Table 2, it can be seen that in the clean node classification task, using only clean_loss actually performs worse compared to using w/o suppress_loss. The authors don't seem to have paid attention to this point. Does keeping the defense model away from the original backdoor model have a greater impact on the clean node classification task?

3、Weakness1.

4、Although the author introduced the LLFC tool, its usage is basically descriptive rather than diagnostic, without exploring which layers' features are most critical for backdoor recovery?

5、The authors acknowledge that LFD is only evaluated against dirty-label attacks. However, more advanced and stealthy clean-label attacks pose a significant threat【1】. Does the core assumption of LFD—that poisoned nodes rely on triggers and are insensitive to neighbors—hold under clean-label attacks? Would the defense mechanism of LFD be rendered ineffective in such scenarios?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a systematic robustness analysis of existing graph backdoor defense methods. The authors identify a critical weakness that many defenses that appear to be effective only achieve superficial security. This security is easily broken by a simple retuning attack, where the defended model is fine-tuned for just a few epochs on a small amount of poisoned data, causing the ASR to rebound sharply. To explain this fragility, the authors leverage Layer-wise Linear Feature Connectivity  to show that "purified" models remain highly connected to the original backdoored model in the feature space. Building on this insight, they propose a novel defense strategy, Layer-wise Feature Divergence , which explicitly maximizes the feature-space divergence between the defense model and the backdoored model during retraining. Extensive experiments across multiple datasets and attacks demonstrate that LFD achieves significantly more robust defense compared to existing state-of-the-art methods.

### Strengths
1. This is the first work to systematically reveal and analyze the superficial security problem in graph backdoor defenses. The finding that defenses can be easily broken is significant for the community and highlights a critical blind spot in current research.

2. The use of LLFC as an analytical tool to explain the root cause of the fragility is innovative and well-motivated. It moves beyond simple parameter-space analysis and provides a convincing, feature-space perspective on why backdoor knowledge persists.

3.  The proposed LFD defense is a direct and elegant solution to the identified problem. The comprehensive experiments across multiple datasets and attacks, along with thorough ablation studies, robustly validate its effectiveness and the contribution of each component.

### Weaknesses
1. As noted in the limitations, the work currently focuses only on dirty-label attacks. Furthermore, the practical threat model of the Retuning Attack requires clarification. The paper's background positions RA as an attacker's action, yet the scenario often implies a defender using it for detection. This ambiguity weakens the motivation. Moreover, if an attacker has the capability to perform RA, it is unclear why they wouldn't simply mount a new backdoor attack from scratch to achieve similar or better effects, which questions the practical necessity of RA.

2. The core mechanism of simulating backdoor activation without knowing the true trigger is not sufficiently justified. The paper claims that adding random noise to node features can help distinguish poisoned nodes, but the theoretical or empirical basis for why this perturbation reliably mimics the effect of the true, structured trigger is lacking. Experimental validation specifically supporting this design choice is needed to confirm its validity over simple random sampling.

3. The description of the poisoned node detection algorithm and the captions for key figures lack clarity, hindering reproducibility. Additionally, the computational overhead of LFD, which requires forward passes through three models, is not discussed, which is a practical concern for adoption.

### Questions
1. The paper's core vulnerability analysis relies on the Retuning Attack. However, the threat model seems ambiguous. Is RA an action performed by an attacker who has obtained a purified model, or is it a diagnostic tool for the defender? If it's an attacker's action, what is the practical advantage of RA over simply training a new backdoored model, given that both require poisoning the training data?

2. In your poisoned node detection method, you use random feature perturbation on all nodes to simulate potential trigger effects. What is the theoretical or empirical evidence that this general perturbation effectively activates the specific, often structure-based, backdoor features implanted by attacks like GTA or UGBA? Are there ablation studies or comparisons that demonstrate the superiority of this specific perturbation strategy?

### Soundness
2

### Presentation
2

### Contribution
2
