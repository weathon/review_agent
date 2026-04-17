# Randomized Antipodal Search Done Right for Data Pareto Improvement of LLM Unlearning

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Large language models (LLMs) sometimes memorize undesirable knowledge, which must be removed after deployment. Prior work on machine unlearning has focused largely on optimization methods that adjust parameters to enforce forgetting while preserving retention. However, these approaches assume that the forget and retain sets are readily available, which rarely holds in practice. Unlearning is typically triggered by an undesired generation at inference time, making the retrieval of relevant data the central challenge. 
We introduce the notion of data Pareto improvement for LLM unlearning, which formalizes how retrieval can expand the achievable trade-off frontier between forgetting and retention. To realize this principle, we propose Randomized Antipodal Search on Linearized Influence Kernel (RASLIK), a retrieval algorithm that combines permutation–projection hashing with randomized antipodal search. RASLIK reduces selection variance, achieves sublinear complexity, and yields a double gain in both quality and efficiency. Across multiple models, datasets, and unlearning algorithms, RASLIK consistently outperforms deterministic baselines and even oracle sampling, establishing randomized search as a principled and scalable solution for data-centric unlearning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles an important but underexplored aspect of LLM unlearning: data retrieval. The authors argue that existing work focuses on optimization algorithms while assuming forget/retain sets are given, which doesn't reflect real-world scenarios where unlearning is triggered by problematic generations at inference time. They introduce the concept of "data Pareto improvement" and propose RASLIK (Randomized Antipodal Search on Linearized Influence Kernel), which uses permutation-projection hashing to efficiently retrieve both aligned samples (to forget) and anti-aligned samples (to retain) from training data. The method is evaluated on two models (OLMo-2-1124-7B and Pythia-2.8B) across two scenarios (trigger-based and domain-specific forgetting) with two unlearning algorithms (GA_GDR and GA_KLR), showing consistent improvements over baselines including oracle sampling.

### Strengths
Problem formulation: Reframing unlearning as a data retrieval problem is insightful and addresses a real practical gap. The "data Pareto improvement" concept provides a principled way to think about retrieval quality.

Theoretical contribution: Theorem 3.3 formally establishes variance reduction properties, which is more rigorous than purely empirical methods. The proof sketch makes intuitive sense (randomization smooths boundary decisions).

Comprehensive evaluation: Testing across 2 models × 2 algorithms × 2 datasets with multiple baselines shows thoroughness. The ablation study (RASLIK-F, CR-x variants) is informative.

Practical efficiency: Achieving sublinear complexity O(|X|k) vs O(|X|d) is important for large-scale applications. The antipodal search via sign-flipping is an elegant computational trick.

### Weaknesses
Missing critical baselines: The paper discusses RapidIn, DataInf, and Alinfik in related work (Section 5) but doesn't compare against them experimentally. These are the most relevant recent methods for influence-based retrieval in LLMs. Without this comparison, it's difficult to assess whether RASLIK truly advances the state-of-the-art or simply outperforms weaker baselines like BM25 and random selection.

Limited scale validation: Only testing on 2.8B and 7B models is a significant limitation for a method claiming to be "scalable." Modern deployed LLMs are often 13B, 70B, or larger. Will the sketch-based approach remain efficient as d grows to hundreds of billions of parameters? Even one experiment on a 13B+ model would strengthen the scalability argument considerably.

Oracle sampling paradox under-explored: The claim that RASLIK outperforms oracle sampling (Table 2, Table 3) is counterintuitive and potentially the paper's most interesting finding, but the explanation is insufficient. The CR-x ablation in Table 3 shows that some noise helps, but why does randomization beat having ground-truth labels? Is this about regularization, avoiding overfitting, or something more fundamental? This deserves deeper analysis rather than a brief mention in Section 4.4.

### Questions
Why no comparison with RapidIn/DataInf/Alinfik? These are cited in your related work as recent advances in influence-based retrieval for LLMs. Were there technical obstacles to including them, or practical constraints? Even if full experiments aren't feasible, could you provide a conceptual comparison explaining how RASLIK differs from and improves upon these methods?

Can you validate Assumption 3.2 empirically? The boundary mass assumption (Λ > 0, margin Γ > γ) is central to Theorem 3.3's variance reduction guarantee. Can you show that your experimental datasets actually satisfy this? For example, plot the distribution of ρ_x values around thresholds τ_F and -τ_R to demonstrate non-zero boundary mass.

What explains the oracle paradox? Why does randomized retrieval consistently outperform oracle sampling? Is this result stable across different random seeds, or could it reflect favorable cherry-picking? Have you analyzed which specific samples RASLIK retrieves differently from Oracle, and why those differences lead to better forgetting-retention trade-offs? This finding challenges conventional wisdom and deserves thorough investigation.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on Large Language Models (LLMs) unlearning, especially the data perspective of achieving satisfactory unlearning. Sepcifically, this work assumes that the forget and retain sets are readily available but rarely holds in practice, so the critical question comes to retrival of relevant data for unlearning. To this end, this work propose randomized antipodal search on linearized influence kernel, a retrieval algorithm that combines permutation-projection hashing with randomized antipodal search. Various experiments are conducted to verify the effectiveness of the proposed method.

### Strengths
1. This paper considers one critical problem setting in unlearning, i.e., the data accessibility for forget and retain set. Which is important but rarely considered in LLM unlearning problem.
2. The proposed RASLIK is novel and reasonable to tackle the targeted problem of effective retrieval of relevant data.
3. The experiments including verfiication on different model and different retriving methods are comprehensive.

### Weaknesses
1. Is there any experimental verification on the computational efficiency.
2. It is unclear how we can decide the threshold in practice, and are there any practical implications for the derived theorem?
3. In addition to the various related works on LLM unlearning, the authors should also consider discussing one work that first considered the partial accessibility of forgetting data in machine unlearning in decoupling the concept and target decoupling [1]. [1] Decoupling the Class Label and the Target Concept in Machine Unlearning. arXiv, 2025

### Questions
Please consider the questions in the weakness part for revision suggestions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the concept of data pareto improvement. Specifically, instead of assuming the forget set and retain set are available for an unlearning request, the authors assume access to only a target generation, which indicates the behavior that we want to forget. Based on this target, the authors retrieve a corresponding forget set and retain set from the training set of the LLM. For retrieval, the paper proposes randomized antipodal search, which first randomly projects the gradient into a lower-dimensional space and then retrieves data samples based on the cosine similarity between the query and training data in the projected space. Experiments on two benchmarks show that the proposed method is effective across different unlearning methods and base models.

### Strengths
1. This paper is novel in the sense that it touches on a new perspective of LLM unlearning. The proposed new setting, where the forget and retain sets need to be retrieved, is also more practical in real-world applications. In general, the reviewer thinks this is a good direction for LLM unlearning.
2. The authors propose a method for this problem based on randomized projection, which is efficient in large-scale applications.
3. The writing and presentation are clear.

### Weaknesses
1. My main concern is the benchmarks used for evaluation. Based on my understanding, in both benchmarks, the knowledge we want to forget is not learned during the LLM's pre-training. Instead, they are behaviors that need to be fine-tuned on a poisoned dataset, or knowledge that needs to be learned from a synthetic dataset. In other words, the knowledge we aim to forget comes from the fine-tuning stage and is purely synthetic. The paper will be greatly improved if the experiments are done to forget the real-world knowledge in LLMs, and the forget and retain sets are retrieved from the actual pre-training corpus. For example, datasets like [1-3] target real-world knowledge for unlearning.

2. The proposed method does not consistently outperform the baselines. For example, in Table 2, the proposed method is worse than the embedding similarity baseline on the Howdy-Alpaca Dataset for the OLMo model and GA_GDR unlearning. In other cases, the performance is worse in either forget rate or retain rate.

3. The cost of the method is not reported. Based on my guess, the proposed method might be more expensive in time compared to baselines like embedding similarity because of the gradient calculation. Can the authors compare the actual time complexity and justify the use of the proposed method?

[1] Shi et al., MUSE: Machine Unlearning Six-Way Evaluation for Language Models.

[2] Li et al., The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning.

[3] Liu et al., Revisiting Who’s Harry Potter: Towards Targeted Unlearning from a Causal Intervention Perspective.

### Questions
Please see weaknesses

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
This paper proposes a new LLM unlearning setting, where the forget data and retain data is not complete. It frames unlearning as a retrieveal problem where given an example generation for unlearn, the task is to identify suitable forget data and retain data from corpus to achieve effective unlearning.

### Strengths
* Novel data-centric perspective for LLM unlearning. The proposed retrieval setup to build best forget and retain data is novel and interesting.
* The proposed randomized search method has a theoratical guarantee.

### Weaknesses
* Incomplete evaluation setup. This paper synthetically constructs two new dataset for evaluating the performance of proposed method. While the experiment show promising performance, it's unclear how this setup approximates real-life unlearning request. For example, the trigger forgetting seems to be a easy pattern matching for retrieval, and the domain-specific forgetting seems also obvious for retrieval.
* Constructing this setup in some commonly used LLM unlearning dataset like TOFU (one author information sentence and the retrieval can target at other factual knowledge of that author) and WMDP should be helpful.
* Missing retrieval example. It's unclear that what kind of sentence or knowledge can be retrieved via the proposed method, raising some concerns about why the performance is better compared to other method.

### Questions
* What's the fictional world knowledge is for Virtual-Alpaca? There seems not to be examples in the main paper or appendix.
* What's the search run-time for proposed method?
* The proposed synthetic dataset seems both on question answering or short textual response, can the framework be applied to large knowledge/concept unlearning like the one in RWKU paper about all information about a person?

### Soundness
3

### Presentation
2

### Contribution
3
