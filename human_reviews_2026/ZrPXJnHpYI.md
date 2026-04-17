# Cross-Model Deception: Transferable Adversarial Attack for Code Search

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Reliable code retrieval is crucial for developer productivity and effective code reuse, significantly impacting software engineering teams and organizations. However, the current neural code language models (CLMs) powering search tools are susceptible to adversarial attacks targeting non-functional textual elements. We introduce a language-agnostic transferable adversarial attack method that exploits this vulnerability of CLMs. Our approach perturbs identifiers within a code snippet without altering its functionality to deceptively align the code with a target query. In particular, we demonstrate that modifications based on smaller models, such as CodeT5+, are highly transferable to larger or closed-source models, like Nomic-emb-code or Voyage-code-3. These modifications can increase the similarity between the query and an arbitrary irrelevant code snippet, consequently degrading key retrieval metrics like Mean Reciprocal Rank (MRR) of the state-of-the-art models by up to 40\%. The experimental results highlight the fragility of current code search methods and underscore the need for more robust, semantic-aware approaches. Our codebase is available at https://github.com/AdvAttackOnNCC/Code_Search_Adversarial_Attack.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes to attack code search-oriented large models in a black-box transfer manner. The adversary aims to increase the ranking of the desired code in the recommendation results of the model, given a natural language query. The adversary can add perturbations to identifiers to craft adversarial examples while maintaining the semantic consistency of the code. To optimize the transferability of the adversarial examples generated on the local surrogate model, the authors propose to approximately linearize the attack loss function to obtain the influence score of each token and then aggregate them to derive the influence score of the identifier. Experimental results show that the proposed attack technique exhibits stable transferability across different datasets and architectures, revealing the possibility of attacking large code models with a relatively small surrogate model.

### Strengths
1. The proposed transfer attack problem is very meaningful, especially for attacking large code models with restricted computational resources.
2. The mathematical and algorithmic techniques presented in this paper are elegant.

### Weaknesses
1. The linearity assumption may not hold for code models. 
A minor perturbation in token space may induce a large divergence in the model feature space (e.g., "Desert" and "Dessert" differ by only a single token but diverge significantly in semantics), while a large perturbation in token space may induce only a small divergence in the model feature space (e.g., "Buy" and "Purchase" are very different in token space but may be very similar in semantic feature space). Thus, the fundamental assumption of this paper may not hold at all.
However, the proposed transfer attack heavily relies on the linear assumption:
(1) The first-order Taylor expansion requires the code model to be approximately linear;
(2) Aggregating the influence scores of tokens into the influence score of identifiers requires the code model to be approximately linear (Algorithm 1 in Appendix). Otherwise, simply summing the influence scores of individual tokens cannot accurately reflect the influence score of an identifier.
The reviewer suggests that the authors add two baseline attacks (random search and greedy search) to examine whether the linearity-based attack algorithm improves over them.


2. The attack perturbations are perceptible. 
Appendix G-1.2 and G-2 show some clean code examples vs. adversarial code examples. For instance, the identifier "glm_ease_back_inout" in the clean code is replaced with "TemporaryBGCWigClassAttributeharitutdowntreat", which is excessively long. Given such lengthy perturbations, any type of attack can succeed very easily. The overly long perturbations can also be easily detected by humans and defensive detectors.

3. Missing related work
The following papers also discuss transfer attacks against code models:

Yang Y, Fan H, Lin C, et al. Exploiting the adversarial example vulnerability of transfer learning of source code. IEEE Transactions on Information Forensics and Security, 2024, 19: 5880-5894.

Liu Q, Ji S, Liu C, et al. A practical black-box attack on source code authorship identification classifiers. IEEE Transactions on Information Forensics and Security, 2021, 16: 3620-3633.

4. Missing baselines
The following papers are query-based black-box attacks. I suggest the authors also compare with them by attacking the surrogate model using these query-based attacks and then transferring the adversarial code to attack the target model to evaluate the transfer attack success rate.

Yang Z, Shi J, He J, et al. Natural attack for pre-trained models of code. Proceedings of the 44th International Conference on Software Engineering. 2022: 1482-1493.

Jha A, Reddy C K. Codeattack: Code-based adversarial attacks for pre-trained programming language models. Proceedings of the AAAI Conference on Artificial Intelligence. 2023, 37(12): 14892-14900.

Pu X, Xiong X, Li Y, et al. CodeSearchAttack: Enhancing soft-label black-box adversarial attacks on code. Journal of Information Security and Applications, 2025, 94: 104258.

Nguyen T D, Zhou Y, Le X B D, et al. Adversarial attacks on code models with discriminative graph patterns. arXiv preprint arXiv:2308.11161, 2023.

5. Missing threat model definition

6. Missing equation numbers

### Questions
1. About Tab. 6, it says, "In the black-box scenario, our attack transfer is more effective and requires just 0.01% of the API calls used by CodeAttack." Why does the black-box transfer attack still consume queries?

### Soundness
2

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
This paper introduces a language-agnostic transferable adversarial attack method that exploits the vulnerability of CLMs, which perturbs (renames) identifier tokens in code snippets (without changing functionality) to make an irrelevant snippet appear highly relevant to the target query.
Experiments demonstrate that adversarial code snippets generated using smaller models are highly transferable to larger or closed-source models, thus significantly reducing the accuracy of mainstream models in code retrieval tasks.
Furthermore, it reveals that current state-of-the-art code search models rely on lexical features rather than deeper code understanding, exposing significant robustness gaps despite high benchmark performance and highlighting the need for future improvement.

### Strengths
1. Well-defined and important problem: the paper reveals a critical vulnerability previously overlooked in code search models.

2. Strong cross-model transferability: adversarial examples crafted on small models effectively fool large and even closed-source models, with >95% success rates.

3. Detailed evaluation: experiments cover multiple models, datasets, and five programming languages with detailed analysis.

### Weaknesses
1. Lack of methodological novelty: the core attack adapts the well-known gradient-based token substitution without major algorithmic novelty (Q1).

2. Lack of rigorous theoretical analysis on attack transferability: there is no formal proof on the transferability of adversarial attacks, which leaves the generalizability and robustness of this work in question (Q2-Q3).

3. Limited quantification on attack margins: identifier renaming may introduce unnatural names, potentially making adversarial code detectable (Q4).

4. Insufficient experimental validation: the evaluation lacks a comparison with random attacks as well as a more comprehensive assessment of the impact on downstream tasks, which are necessary to validate the effectiveness and generalizability of the attack (Q5).

5. Some minor writing problems.

### Questions
1. Given that the proposed approach aims to heuristically increase the similarity between a code snippet and a target query, while leaving the initial embedding space unchanged, I am wondering whether the constraints of the embedding space or vocabulary introduce inherent limitations. For instance, in the example outlined in Section 1, where an attacker aims to push malicious code to users with high computational resources, if the malicious code segment does not exist within the model's embedding space, how can this method be effectively deployed?

2. The authors defer the investigation into the causes of the strong correlation in attack transferability across different models to future work. However, I believe that an in-depth theoretical analysis and discussion of this correlation, supported by controlled ablative studies, should constitute a key contribution of this paper. For instance, comparing the transferability performance on models pre-trained on distinctly different corpora but with identical architectures, or vice versa, would be highly informative. Even experiments involving locally-trained, relatively simple models could substantiate the conclusions.

3. While the attacks crafted with a small model are mostly effective on larger ones, one case (attacking with OASIS and evaluating on CodeT5+) showed lower success (e.g. ~82% precision on CosQA). Could the authors elaborate on conditions where the transferability does not hold or be less robust? For example, if the target model uses a drastically different embedding space or tokenization, would the attack still work?

4. How realistic is it for an attacker to introduce adversarial code without being noticed? The attack alters identifier names in code, which could introduce odd or semantically meaningless names to increase similarity with the target query. In practice, a code snippet with bizarre variable names (chosen to match a query’s keywords) might alert a vigilant developer or be flagged by simple static analysis.

5. How to prove that the performance change observed under the proposed method is different from the baseline that merely makes random perturbations in the embedding space? For example, in the Experiments RQ4 of CodeAttack [1], the authors compared CodeAttack against a variant, CodeAttackRAND, which randomly samples tokens from the input code for substitution. Similarly, a comparative experiment involving such random attacks is expected.

[1] Jha, A., & Reddy, C. K. (2023). CodeAttack: Code-Based Adversarial Attacks for Pre-trained Programming Language Models. Proceedings of the AAAI Conference on Artificial Intelligence, 37(12), 14892-14900.

6. There are some writing and formatting errors in the paper, for example:
   - In Introduction, "closed-sourced (Voyage-code-3)" –> "closed-source (Voyage-code-3)".
   - In Introduction, "for example pushing malicious cryptomining code..." –> "for example, pushing malicious cryptomining code...".
   - In the title of Table 5 "...that the attack transfer effects, while present, is less predictable...", "is"->"are"

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
This paper proposes an adversarial attack method for code search, which perturbs the code snippet to maximize its similarity with the target query while preserving functionality. The authors show that changes made using smaller models, such as CodeT5+, are highly transferable to larger or closed-source models like Nomic-emb-code or Voyage-code-3. These changes can increase the similarity between a query and an irrelevant code snippet, reducing key retrieval metrics such as Mean Reciprocal Rank (MRR) of state-of-the-art models by up to 40%. Experimental results show the vulnerability of current code search methods and the need for more robust, semantic-aware approaches.

### Strengths
- This work presents a dedicated adversarial attack against code search models, identifying a significant threat vector.

- The paper demonstrates the vulnerability of current code search methods, as attacks transfer effectively across model sizes and architectures, including closed-source systems.

### Weaknesses
1. Lack of Demonstrated Practical Impact: The threat model is narrowly defined. The paper relies on embedding similarity as a proxy for success, failing to demonstrate consequential failures in downstream tasks (e.g., corrupting code retrieval or evading a vulnerability scanner). The threat model should be refined and grounded in practical scenarios to establish true utility and severity.

2. Limited Evaluation Setup: The attack is highly targeted, requiring a specific code snippet as an objective. This overlooks the more general and practical threat of non-targeted attacks, which aim to cause arbitrary misclassification or degradation, and should be evaluated to assess the full scope of the vulnerability.

3. Insufficient Comparisons: The chosen baseline (Code Attack) is designed for classification, not retrieval. A comparison against state-of-the-art adversarial retrieval methods is missing, undermining the claim of effectiveness. Furthermore, there are many recent code LLMs and code search models, which the paper fail to evaluate.

Previously work, such as (Zhang et al., 2023) and (Tian et al., 2023), already described semantic-preserving code transformations for evaluating machine learning-based code models. The paper can discuss and compare.

W. Zhang, et al.,  "Challenging Machine Learning-Based Clone Detectors via Semantic-Preserving Code Transformations," in IEEE Transactions on Software Engineering, vol. 49, no. 5, pp. 3052-3070, 1 May 2023.

Tian Z, Chen J, Jin Z. Code difference guided adversarial example generation for deep code models, ASE 2023.

4. Narrow Evaluation Focus: The assessment relies solely on ranking metrics (e.g., MRR and Recall), failing to evaluate the semantic correctness or potential maliciousness of the retrieved adversarial code, which is critical for assessing real-world impact.

5. This paper introduces a scenario where ~10% of irrelevant snippets in a small candidate pool are adversarially edited, but the evaluations rely on offline pools (e.g., ~500 items in CosQA) and sampled pairs. The setup shows sensitivity under controlled conditions, but it does not establish exploitability in real repositories with large-scale indexing, deduplication, and ingestion pipelines.

6. This paper claims to preserve functionality by ensuring the consistency of the AST (Abstract Syntax Tree) structure. However, this cannot fully guarantee semantic consistency. The reason is that identifier renaming may lead to naming conflicts or break code that relies on reflection and specific naming conventions—for instance, code that requires specific identifier names to interact with external APIs.

7. Presentation Issues: The paper suffers from some presentation issues, such as  lack of figures, and grammatical errors.

### Questions
Please refer to “Weaknesses”.

### Soundness
2

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
4

### Summary
This paper investigates the robustness of neural code language models (CLMs) used for code retrieval against adversarial attacks. The authors propose a language-agnostic, transferable adversarial attack that perturbs non-functional code elements (mainly identifiers) to mislead code search models without changing the program’s semantics. They demonstrate that adversarial examples generated using smaller open-source models such as CodeT5+ can successfully transfer to larger or closed-source models, including Nomic-emb-code and Voyage-code-3. Experimental results show that such perturbations can degrade retrieval performance—reducing MRR by up to 40%—thus revealing significant vulnerabilities in current CLM-based code search systems.

### Strengths
+ The experimental evaluation is comprehensive. The authors conduct extensive experiments on both CodeT and OASIS, demonstrating the consistency and robustness of their findings across different code search models.
+ The transferability results are especially interesting — adversarial examples produced with small models effectively deceiving large or closed-source systems is a convincing and practically relevant finding.

### Weaknesses
- Incremental contribution. The authors claim “We propose one of the first adversarial attack methods for code search”; however, prior work (e.g., [1,2]) has already explored adversarial attacks on code retrieval systems. Although the authors argue that previous studies mainly targeted classification tasks, this distinction is not convincing, as those works share similar attack mechanisms and objectives. Consequently, the novelty of the proposed approach appears rather incremental.

- Weak motivation. The motivation is weak and somewhat overstated. Although the authors claim that adversarial attacks on code search present “unique challenges”, the differences from classification tasks are superficial. The example of “targeted manipulation of search results” is speculative and lacks empirical support, and the reliance on “offline embeddings” does not constitute a fundamentally new threat model, as similar vulnerabilities exist in other embedding-based systems. I suggest the authors provide a detailed, realistic example to better illustrate the practical relevance of their claimed challenges.

- Unclear Method. The authors seem to have placed much of the theoretical or preliminary content within the method section (e.g., Section 3.1 Gradient-Based Method), which mainly contains derivations rather than concrete methodological descriptions. I suggest adding an overall schematic figure to clearly illustrate the attack pipeline — including the inputs, outputs, and the overall process of how the attack is performed.


[1] Codeattack: Code-based adversarial attacks for pre-trained programming language models

[2] You see what I want you to see: poisoning vulnerabilities in neural code search

### Questions
1. Could the authors clarify how their proposed attack method fundamentally differs from prior adversarial attacks on code retrieval or classification models?

2. Can the authors offer a concrete, realistic scenario or case study demonstrating the practical impact of such attacks in real-world code search systems?

### Soundness
3

### Presentation
3

### Contribution
2
