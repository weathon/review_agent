# Curation Leaks: Membership Inference Attacks against Data Curation for Machine Learning

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 2

## Abstract
In machine learning, curation is used to select the most valuable data for improving both model accuracy and computational efficiency. Recently, curation has also been explored as a solution for private machine learning: rather than training directly on sensitive data, which is known to leak information through model predictions, the private data is used only to guide the selection of useful public data. The resulting model is then trained solely on curated public data. It is tempting to assume that such a model is privacy-preserving because it has never seen the private data. Yet, we show that without further protection, curation pipelines can still leak private information. Specifically, we introduce novel attacks against popular curation methods, targeting every major step: the computation of curation scores, the selection of the curated subset, and the final trained model. We demonstrate that each stage reveals information about the private dataset and that even models trained exclusively on curated public data leak membership information about the private data that guided curation. These findings highlight the previously overlooked inherent privacy risks of data curation and show that privacy assessment must extend beyond the training procedure to include the data selection process. Our differentially private adaptations of curation methods effectively mitigate leakage, indicating that formal privacy guarantees for curation are a promising direction.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper uncovers a novel and important privacy vulnerability in machine learning pipelines: membership leakage during data curation. The authors demonstrate both theoretically and empirically that data curation, even before model training, can expose sensitive information about which samples were part of the raw dataset.

### Strengths
1. The paper addresses a new angle on privacy leakage. Most membership inference works focus on model outputs; This work focus shifts to data preprocessing and curation.
2. The problem is well motivated and clearly defined.
3. The empirical results are convincing and well-presented.

### Weaknesses
1. While the proposed curation leak is novel and conceptually important, its real-world applicability is limited by the assumption that the adversary can access or infer the curation outcome. In many production pipelines, the raw uncurated data or discarded samples are not observable.
2. The paper briefly mentions DP sampling and randomization but does not propose new defense mechanisms.
3. Ablation studies (e.g., effect of dataset size or embedding dimension) are not included.

### Questions
1.Clarify the assumption
2.Add necessary ablation studies.

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
The paper argues that “privacy via curation” (only using sensitive data to select public data and then training solely on the public subset) is not automatically private. They design membership inference attacks against: (i) curation scores; (ii) the selected public subset; and (iii) the final model (via a small number of crafted public “fingerprint” samples).The results show that image-embedding/nearest-neighbor style curation is much more vulnerable; TRAK-style gradient-averaged curation is more robust but still leaky for small target sets.

### Strengths
The problem itself seems interesting.

### Weaknesses
1. I’m familiar with LiRA and recognize you use an online variant, but the three attack surfaces are hard to follow because the threat model isn’t explicitly stated up front. Please spell out—on one page—(i) the adversary’s goal (membership in the private target set used for curation vs. classic training-set membership), (ii) what the adversary can observe at each stage (scores, selection mask, final model), and (iii) what the adversary can do (e.g., can they inject public items?). A single “who-sees-what” diagram for the 3 stages would save readers a lot of guesswork. Also, there are many typos in the main text and appendix; a thorough copy edit would help.

2. The paper does not convincingly explain why leakage at the scoring and subset-selection stages matters when, in many realistic deployments, only the final model is exposed. In such cases, standard membership inference on the trained model already demonstrates risk. The authors should clarify concrete scenarios (and access assumptions) where Sections 3.1 and 3.2 introduce new or additional threat surface beyond what model-only exposure already entails.

3. The end-to-end attack hinges on the ability to insert crafted samples into the public pool. This reads as logically circular: relying on an existing injection threat to establish a new leakage threat. The paper should justify the realism and scope of this assumption (e.g., where such insertion is feasible, at what rates, and under what defenses or deduplication) and disentangle the novelty of the proposed leakage from the prerequisite capability.

4. The underlying question is important: if a model is trained on curated public data, where the curation used a private target set, does the deployed model leak membership about that private set? However, the current presentation makes the answer hard to find. A concise figure (or two) that lays out the threat model for all three stages—attacker view, defender assets, and leakage channel—would greatly improve readability. With a clearly defined problem and threat model, the empirical results would be far more persuasive; without that foundation, even sophisticated methods risk failing to convince readers.

### Questions
Please address the “Weaknesses” above in your rebuttal—especially by (1) stating the problem crisply, (2) explaining why it matters in realistic deployments, and (3) laying out clear threat models for all three attack stages. If those pieces are clarified and make sense, I’m inclined to raise my score.

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
3

### Summary
This paper investigate the privacy risks of data curation and show effective membership inference attacks on all the major steps of data curation. This includes similarity score evaluation of the data points in the public dataset (which is the easiest), and the construction of the curated public dataset (which is harder), and the training of the ML model on the curated dataset. The paper builds upon popular and well-known membership inference attacks and provided versions that works regardless of the data curation method used. Furthormore, custom attack methods are designed for image similarity score-based curation and TRAK-based curation. Experiments show that while data curation does not train any model on the private dataset, but all its major steps still leak the membership of the private domain dataset, and it is worse for smaller domain dataset. This reveals new privacy risks of data curation.

### Strengths
1. The paper reveals new privacy risks of data curation via membership inference attacks.
2. Systematic review of MIAs in all steps in the data curation pipeline, and attacks are proposed and validated for all of them.
3. Existing MIAs work for agnostic with modifications, but the paper also proposes custom attacks are proposed to target concrete data curation methods, including TRAK and image-based data curation.

### Weaknesses
1. Choice of parameters are not clear or discussed in the main text.
2. Computational complexity of the attacks is not discussed.
3. Limited experiments for end-to-end model MIAs.

Please see the questions listed in the section below.

### Questions
1. Some important parameter and implementation details are not explained. For example, how many shadow models $m$ are used in the attacks? What are the parameters in the data curation procedure and what is the setting of the model training procedure? When training the model, is regularization used? If so, how heavy is that?
2. The paper does not discuss the effects of $m$ on the attack performance. I would like to see the relation between the compute spent by the attacker and the attack effectiveness.
3. For end-to-end model MIAs, all results are when the selected target subset is 1% of the target set. I understand that model training is time consuming, but I would like to see the attack performance for more TPRs.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents the first comprehensive privacy analysis of data curation pipelines in machine learning. The authors challenge the common assumption that training models on curated public data inherently protects private target datasets used for curation. Through systematic membership inference attacks across three pipeline stages (curation scores, selected subsets, and final models), the paper demonstrates that each stage leaks information about the private target set.

### Strengths
- This is the first systematic analysis of privacy leakage in data curation pipelines, addressing a genuine blind spot in the community. it systematically evaluates privacy leakage across three critical stages of the curation pipeline (curation scores, selected data subsets, and the final trained models) using diverse datasets and curation methods.

- The analysis of influence sparsity in image-based curation provides valuable insights into why certain methods are more vulnerable.

- The paper delivers actionable insights: simple max-based curation is fundamentally insecure, while average-based curation (like TRAK) provides innate robustness except in the small-dataset regime .

### Weaknesses
- The threat model is not clearly defined. The adversary's assumed capabilities and knowledge are not clearly stated upfront and appear to change depending on the attack. The adversary knowledge could be escalates to extreme, white-box levels. For the end-to-end TRAK attack, the adversary is assumed to have profound knowledge of the curation mechanism, including the model architecture, the ability to compute gradients, and the need to calculate the Gram matrix $G$. This level of white-box access is not clearly stated or justified.

- The paper's core attack methodology is its adaptation of LiRA. This adaptation is (a) confusingly explained and (b) methodologically questionable. Running an attack algorithm on subsets of the private data to build in/out distributions  is not standard and is not justified as a sound method for simulating the true 'in' and 'out' worlds.

- The paper's strongest end-to-end attack is weakened by its reliance on a proxy metric. The authors state the full end-to-end attack is 'computationally intractable'. Their solution is to 'assume that the adversary can measure whether $f \in \tilde{D}$.' This assumption effectively equates selection with detection, completely ignoring the possibility that ML training dynamics (e.g., SGD noise, and the influence of thousands of other samples) could 'drown out' the fingerprint’s signal, making it undetectable in the final model even if it had been selected.

### Questions
- Can you please provide a single, clear definition of the threat model?

- Can you please provide a more detailed justification for your 'shadow' methodology? Specifically, why is re-running the curation algorithm on random subsets of the private data $\mathcal{T}$  a sound way to construct the in/out distributions for a LiRA attack? This seems to be a custom algorithm that borrows the LiRA name, and its statistical validity is not obvious.

- Can you provide any evidence that this assumption of selection equals detection holds? For example, can you show that a fingerprint sample, when selected, always creates a detectable signal in the final model, even when trained as part of a large curated set?

- Could you please analyze the query budget/cost or computational cost of the attack? How would it affect the scalability of the method?

### Soundness
2

### Presentation
2

### Contribution
3
