# HalluGuard: Demystifying Data-Driven and Reasoning-Driven Hallucinations in LLMs

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
The reliability of Large Language Models (LLMs) in high-stakes domains such as healthcare, law, and scientific discovery is often compromised by hallucinations. These failures typically stem from two sources: *data-driven hallucinations* and *reasoning-driven hallucinations*. However, existing detection methods usually address only one source and rely on task-specific heuristics, limiting their generalization to complex scenarios. To overcome these limitations, we introduce the *Hallucination Risk Bound*, a unified theoretical framework that formally decomposes hallucination risk into data-driven and reasoning-driven components, linked respectively to training-time mismatches and inference-time instabilities. This provides a principled foundation for analyzing how hallucinations emerge and evolve. Building on this foundation, we introduce **HalluGuard**, an NTK-based score that leverages the induced geometry and captured representations of the NTK to jointly identify data-driven and reasoning-driven hallucinations. We evaluate **HalluGuard** on 10 diverse benchmarks, 11 competitive baselines, and 9 popular LLM backbones, consistently achieving state-of-the-art performance in detecting diverse forms of LLM hallucinations. We open-source our proposed \model{} model at https://github.com/Susan571/HalluGuard-ICLR2026.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the problem of hallucinations that occur when large language models generate content, proposing a unified theoretical framework and a novel detection method.

On the theoretical side, the paper introduces an upper bound for hallucination risk, explicitly decomposing it into two components: data-driven and inference-driven risks, which respectively stem from knowledge bias during training and instability during inference.

On the methodological side, based on the NTK theory, the authors propose the HalluGuard scoring method, which leverages the geometric properties of the NTK and the model’s internal representations to jointly detect both types of hallucinations.

### Strengths
1. The paper formally decomposes hallucination risk into data-driven and inference-driven components and derives a theoretical upper bound for the risk, demonstrating significant theoretical value.

2. The method outperforms existing methods on multiple benchmarks, with particularly notable improvements on small models and inference-related tasks.

3. The presentation of the paper is good.

### Weaknesses
1. Potentially high computational complexity: Computing the NTK matrix and related quantities such as its determinant and condition number may introduce significant computational overhead for very large models or long-sequence generation tasks. It is recommended to discuss the scalability of the method or propose approximate computation strategies in the paper.

2. Strong dependence on the “semantic encoder Φ”: HalluGuard relies on a task-specific encoder Φ to map model outputs into a hypothesis space. The paper does not sufficiently analyze how the choice of Φ affects the results. It is suggested to include comparative experiments with different encoding strategies (e.g., BERT, SimCSE, etc.).

3. Strong theoretical assumptions: The theoretical analysis depends on several assumptions (e.g., A1–A3), whose validity in practical models may be questionable. It is recommended to discuss the reasonableness and potential relaxation of these assumptions in the appendix.

4. Insufficient discussion of the interaction between the two types of hallucinations: Although the paper emphasizes that the two types of hallucinations often co-occur, the HalluGuard score is computed as a simple sum of three components, without modeling their interaction effects. The authors could consider introducing interaction terms or dynamic weighting mechanisms.

5. Experimental section could be further enriched: While the experiments are comprehensive, validation on multimodal models or long-text generation tasks (e.g., story writing, long-document summarization) is missing—scenarios where hallucinations are typically more complex. It would also be valuable to add case studies illustrating the evolution process of hallucinations, showing intuitively how HalluGuard captures the “trajectory” of hallucination formation.

### Questions
1. Potentially high computational complexity: Computing the NTK matrix and related quantities such as its determinant and condition number may introduce significant computational overhead for very large models or long-sequence generation tasks. It is recommended to discuss the scalability of the method or propose approximate computation strategies in the paper.

2. Strong dependence on the “semantic encoder Φ”: HalluGuard relies on a task-specific encoder Φ to map model outputs into a hypothesis space. The paper does not sufficiently analyze how the choice of Φ affects the results. It is suggested to include comparative experiments with different encoding strategies (e.g., BERT, SimCSE, etc.).

3. Strong theoretical assumptions: The theoretical analysis depends on several assumptions (e.g., A1–A3), whose validity in practical models may be questionable. It is recommended to discuss the reasonableness and potential relaxation of these assumptions in the appendix.

4. Insufficient discussion of the interaction between the two types of hallucinations: Although the paper emphasizes that the two types of hallucinations often co-occur, the HalluGuard score is computed as a simple sum of three components, without modeling their interaction effects. The authors could consider introducing interaction terms or dynamic weighting mechanisms.

5. Experimental section could be further enriched: While the experiments are comprehensive, validation on multimodal models or long-text generation tasks (e.g., story writing, long-document summarization) is missing—scenarios where hallucinations are typically more complex. It would also be valuable to add case studies illustrating the evolution process of hallucinations, showing intuitively how HalluGuard captures the “trajectory” of hallucination formation.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces HalluGauard, a neural tangent kernel based score for detecting hallucinations in outputs of LLMs. The formulation of the score explicitly considers both hallucinations due to problems with training data and those due to flawed reasoning during model inference. Next, the paper considers several dimensions of efficacy of HalluGuard, and provides empirical justification for each of these through extensive comparison with baselines on well-established benchmarks.

### Strengths
- Admittedly I have not checked the math thoroughly. But I based on my understanding, HalluGuard is well motivated with sound mathematical justification.
- I really appreciate the clarity with which the authors justify the question of efficacy of HalluGuard compared existing methods in the literature. The various dimensions in which they measure the performance of HalluGuard were well stated and more importantly, extensive experimental results were provided in favor of HalluGuard.

### Weaknesses
- Some justification for the assumptions in Section 3.2 would be nice to see. I did not see any references, discussions or proofs for the validity or the practical applicability of the assumptions.

**Minor:**

There is some text overlap in the headings of Table 4

### Questions
- What is the practical computational complexity of HalluGuard? Some details on either FLOPS or runtime would be helpful for readers.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses a central reliability problem in Large Language Models (LLMs) — hallucinations, which are unfaithful or nonsensical outputs that undermine trust in high-stakes applications (e.g., healthcare, law, science). The authors argue that hallucinations originate from two fundamentally distinct sources: Data-driven hallucinations and Reasoning-driven hallucinations. They introduce a Hallucination Risk Bound, formally decomposing total hallucination error into these two components and analyzing them via Neural Tangent Kernel (NTK) geometry and probabilistic concentration bounds. Building on this theory, the authors develop HALLUGUARD, an NTK-based score which jointly captures representational adequacy, rollout amplification, and spectral stability to detect both hallucination types without external references. Experiments on 10 benchmarks, 11 baselines, and 9 LLM backbones show consistent state-of-the-art detection performance and even improved reasoning accuracy when used during inference.

### Strengths
1. The paper introduces a novel theoretical framework, the Hallucination Risk Bound (HRB), that unifies data-driven and reasoning-driven hallucinations under a single mathematical formulation—a first in this research area.
2. The work is methodologically rigorous, combining solid theoretical derivations with extensive empirical validation across 10 benchmarks, 11 baselines, and 9 LLM architectures.
3. The paper is exceptionally clear and well-organized, balancing technical precision with intuitive explanations and well-labeled figures and tables.
4. The research makes a substantial contribution to the understanding and mitigation of hallucinations, offering both theoretical insight and practical tools for safer, more reliable LLMs.

### Weaknesses
1. While the theoretical formulation is elegant, the connection between NTK geometry and hallucination phenomena could be further deepened by clarifying how kernel dynamics specifically capture semantic drift and logical inconsistency beyond representational similarity.
2. Although the experiments are broad, the evaluation largely focuses on detection performance metrics (AUROC, AUPRC), leaving limited insight into causal behavior—whether reducing HALLUGUARD score indeed prevents hallucinations across diverse prompts.
3. The theoretical sections (particularly Proposition 3.1 and Theorem 3.2) could benefit from more intuitive explanations and graphical illustrations of the terms involved (e.g., geometric meaning).

### Questions
1. The Hallucination Risk Bound combines NTK-conditioned data deviation and a Freedman-style reasoning instability term. Could the authors clarify the tightness of this bound—i.e., under what conditions might it become vacuous or fail to predict real hallucination behavior?
2. While the paper cites Inside (Chen et al., 2024a) and MIND (Su et al., 2024), could the authors explicitly compare how their theoretical decomposition differs from these methods’ representation-based or temporal-state analyses?
3. The experiments convincingly show detection improvements, but do HALLUGUARD scores predict future hallucination likelihood in multi-turn or instruction-following dialogues?
4. The theoretical analysis models hallucination as a function of stepwise Jacobians (J_t), but these are intractable in large LLMs. How closely does the NTK-based approximation track actual gradient dynamics observed in smaller, analyzable models?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
In this paper, the authors investigate the phenomenon of hallucinations in Autoregressive LLMs, and propose two specific components to analyze in detail: those that originate due to poor data from the training stage, and the second being inference-time decoding issues that arise in generation. Toward this, the paper introduces HalluGuard, a scoring method that decomposes the hallucination risk into distinct terms by bounds obtained from eigenspectrum of the Neural Tangent Kernel under semantic perturbations. In practice, the paper proposes three scoring components, $det(\mathcal{K})$ i.e the determinant of the NTK gram matrix for the data-driven term, $log~\sigma_{max}$ the supremum Jacobian norm and $-log \kappa(\mathcal{K})^{2}$ utilizing the condition number to account for autoregressive reasoning-driven term. The proposed HalluGuard Score is shown to be effective in practice on standard datasets and LLM models as well, achieving notable improvements over several baseline approaches.

### Strengths
1) The primary strength of the paper is to cast the hallucination detection problem in a formal theoretical framework, to decompose the hallucination risk into data-driven and reasoning-driven sources. This lays appropriate groundwork to then analyse the source of hallucinations themselves, which otherwise is often heuristic at best.

2) The mathematical framework is introduced in a clear and lucid manner, though some terms are introduced with limited motivation (condition number term). Furthermore, the authors empirically demonstrate that the scores proposed for the data-driven and reasoning-driven components vary as expected in a practical setting over standard datasets such as SQuAD, Math-500 and TruthfulQA.

3) The proposed HalluGuard method is shown to be effective on several datasets such as RagTruth, BBH, TruthfulQA, SQuAD, GMS8K and HaluEval. The method is also shown to perform better than numerous baselines, though the choice of comparison points for prior works could be improved (kindly refer to Weaknesses section below).

4) The utilization of HalluGuard toward improving beam search in test-time inference to boost performance is another strong and notable contribution. The paper further presents a detailed case-study in fine-grained hallucination detection, wherein lexically similar yet semantically incorrect outputs are analyzed, and helps demonstrate the efficacy of the proposed scoring method.

### Weaknesses
1) While the core theory as presented in Theorem 3.2  decomposes risk into two terms (data-driven and reasoning-driven), the final score is an additive combination of three terms. The inclusion of the third term ($-log~\kappa^2$), while explored in the appendix as a penalty, appears to be arbitrary and appended without adequate motivation overall. Furthermore, the paper relies on a Freedman inequality to show that the reasoning-driven error term grows exponentially with sequence length T. However, this is extremely loose in practice, and an overly simplistic justification for the reasoning-driven term in the final score.


2) Unexplained Jacobian Proxy: While the paper states that computing the direct step-wise Jacobians for billion parameter LLMs are intractable, the final score still utilizes the eigenspectrum of the NTK Gram Matrix. It is quite unclear from the paper about how this translates to the computational overhead needed to find the final HalluGuard Score in practice. Could the authors kindly clarify how each of the three terms is computed in practice, and most importantly, report the overall run-times for detection and compare this with other other scoring methods? (Computing SVD, the Jacobian matrix possibly requiring a back-propagation step, their supremum norm and condition number)


3) The Appendix mentions that HalluGuard requires the training of lightweight projection layers using AdamW, but this was not introduced in the main paper. Could the author please clarify this critical aspect?


4) The empirical evaluations could be significantly improved by comparing with more relevant baselines like SAPLMA [1] which trains lightweight probes over the network hidden states, and LLM-Check [2] which utilizes the log-determinant of attention kernels for hallucination detection.


5) Semantic Perturbations: The NTK calculation depends entirely on light semantic perturbations. However, the paper does not specify what semantic perturbations are used to compute the NTK kernel? Could the authors also clarify how many perturbations are used? 


6) The proposed method is also mentioned to use “Perturbation Regularization” by using a memory bank of N =3000 token embeddings and set thresholds. This appears to be set-apart from the theoretical motivations entirely, and appears to be a complex overhead. For instance, this suggests that the memory bank would require several thousand samples from the same distribution as that of a sample seen in test-time, which significantly reduces the applicability of the method in most practical settings. Furthermore, if such a large sample set is utilized, prior works such as ITI [3] form a strong baseline of comparison, since the degree of discriminability can be adaptively set for detection.


7) The empirical evaluations shown report only AUROC and AUCPR, however it is crucial to analyze hallucination detection at low False-Positive-Rates (FPR) as well. Could the authors kindly provide standard detection metrics such as True-Positive-Rate (TPR) at 5% or 10% FPR, F1 score etc, since it is difficult to assess the practical efficacy of such detectors without these metrics.


8) Line 458: “Zhou & et al. (2024) proposed EIGENSCORE, which computes… “ points to the reference "Inside: Interpretable self-diagnosis for llm hallucination detection. ICML, 2024." This citation appears to be hallucinated, and I could not find this paper from ICML 2024, and should likely refer to Chen et al., "Inside: Llms’ internal states retain the power of hallucination detection, 2024a", an existing citation that is already used in the paper.



[1] The internal state of an LLM knows when it's lying, A Azaria, T Mitchell, EMNLP 2023


[2] LLM-Check: Investigating Detection of Hallucinations in Large Language Models, Sriramanan et al, NeurIPS 2024 


[3] Inference-Time Intervention: Eliciting Truthful Answers from a Language Model, Li et al. NeurIPS 2023

### Questions
Kindly refer to the questions mentioned in the weaknesses section above. I would be happy to raise my score further if these could be adequately addressed.

### Soundness
2

### Presentation
3

### Contribution
3
