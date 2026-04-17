# ADEPT: Continual Pretraining via Adaptive Expansion and Dynamic Decoupled Tuning

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
Conventional continual pretraining (CPT) for large language model (LLM) domain adaptation often suffers from catastrophic forgetting and limited domain capacity. Existing strategies adopt layer expansion, introducing additional trainable parameters to accommodate new knowledge. However, the uniform expansion and updates still entangle general and domain learning, undermining its effectiveness. Our pilot studies reveal that LLMs exhibit functional specialization, where layers and units differentially encode general-critical capabilities, suggesting that parameter expansion and optimization should be function-aware. We then propose ADEPT, Adaptive Expansion and Dynamic Decoupled Tuning for continual pretraining, a two-stage framework for domain-adaptive CPT. ADEPT first performs General-Competence Guided Selective Layer Expansion, duplicating layers least critical for the general domain to increase representational capacity while minimizing interference with general knowledge. It then applies Adaptive Unit-Wise Decoupled Tuning, disentangling parameter units within expanded layers according to their general-domain importance and assigning asymmetric learning rates to balance knowledge injection and retention. Experiments on mathematical and medical domains show that ADEPT outperforms full-parameter CPT by up to 5.76% on the general benchmarks and 5.58% on the target domain benchmarks with only 15% of parameters tuned and less than 50% training time. Ablation studies, theoretical analysis, and extended investigations further demonstrate the necessity of targeted expansion and decoupled optimization, providing new principles for efficient and robust domain-adaptive CPT. Our code is open-sourced at https://github.com/PuppyKnightUniversity/ADEPT

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces **ADEPT** (Adaptive Expansion and Dynamic Decoupled Tuning), a two-stage framework designed to improve continual pretraining (CPT) for large language models (LLMs) when adapting them to specialized domains like mathematics and medicine. The primary challenges it addresses are catastrophic forgetting (losing general knowledge) and the limited capacity of models to absorb new information. The method is based on the key insight that LLMs exhibit functional specialization, where different layers and parameters have varied importance for preserving general knowledge. In its first stage, ADEPT selectively expands the model by duplicating layers least critical to general competence, adding capacity with minimal interference. The second stage performs decoupled tuning on these new layers, applying asymmetric learning rates to protect general-critical parameter units while allowing adaptable ones to absorb domain knowledge. Experiments demonstrate that ADEPT outperforms the baselines in both target and general domain performance, while being more efficient by tuning only small number of parameters in less than half the training time.

### Strengths
The paper demonstrates a fair degree of originality by effectively combining existing ideas in a novel manner. The significance of the work is clear, as it addresses an catastrophic forgetting problem in LLM area. The clarity of the writing is commendable, with proper use of English grammar and well-designed visualizations that aid in understanding the proposed methods.

- **Novelty Through Methodological Rigor and Transparency**: While the individual components—such as parameter importance analysis, layer expansion, and adaptive tuning—are not entirely new, the paper’s novelty stems from the rigorous process used to select and combine them. The authors build a compelling case for their approach by meticulously justifying their unique combination of existing techniques. This is best illustrated in the appendix, where a systematic comparison of four different methods for probing layer importance is presented. By transparently demonstrating that their chosen "masking out" strategy delivers superior performance, the authors provide a clear, evidence-based rationale for their design choice, which greatly enhances the reader's understanding and trust in the proposed method.

- **Significance and Strong Motivation**: The work addresses the critical and highly relevant problem of catastrophic forgetting during domain-adaptive continual pretraining, a key challenge for deploying specialized LLMs. The authors establish a compelling motivation not just by stating the problem, but by conducting a pilot study (Section 2) to provide an empirical foundation for their core hypothesis of "functional specialization". This initial analysis, which demonstrates that different layers and parameter units contribute unequally to preserving general knowledge, serves as a powerful justification for the entire ADEPT framework.

- **Exceptional Clarity and Presentation**: The paper is written with outstanding clarity, making the complex methodology accessible. The logical flow from the problem statement to the empirical validation is seamless. This is further enhanced by well-designed visualizations; for instance, the main methodology diagram (Figure 3) provides an intuitive step-by-step overview of the ADEPT process , while the analytical figures on activation distributions (Figure 4) and token shifts (Figure 5) offer insightful, visual proof of the method's effectiveness.

### Weaknesses
While the paper is methodologically sound and the results are compelling, its primary weakness lies in the limited scope of its experimental validation, which could be expanded to further solidify the claims of generality.

* **Scope of Domain Adaptation**: The authors make a strong case for ADEPT's effectiveness by testing on two distinct and challenging domains: Mathematics (emphasizing reasoning) and Medicine (emphasizing factual knowledge). This is a commendable choice. However, the claim of the framework's general applicability would be significantly bolstered by including experiments in other complex domains. For instance, code generation represents a crucial application area for LLMs that combines logical structure, strict syntax, and algorithmic reasoning. Demonstrating ADEPT's success in adapting a model to a specific programming language or codebase would provide more robust evidence of its versatility.

### Questions
Thank you for this well-written and insightful paper. The proposed ADEPT framework is an elegant and effective solution to a significant problem in continual learning for LLMs. The empirical results are strong and the analysis is thorough. To further strengthen the work and clarify some points for the reader, I have the following questions and suggestions:

1. **On the Generality of the Framework**: The choice of Mathematics and Medicine as test domains is excellent, as they represent distinct challenges (reasoning vs. factual knowledge). However, to fully substantiate the claim of the framework's general applicability, further evidence would be beneficial.
    - Question: Have the authors considered evaluating ADEPT on a domain with fundamentally different characteristics, such as code generation? Success in a domain governed by strict syntax and algorithmic logic would provide powerful evidence for the method's versatility.

2. **On the Dynamics of Decoupled Tuning**: The methodology for Stage 2 mentions that unit importance scores are periodically recomputed to keep the learning rates adaptive throughout training.
    - Question: Could the authors please specify the update interval for these importance scores used in the experiments (e.g., every 500 steps, as noted in the appendix )? Further, was any sensitivity analysis performed on this interval? It would be interesting to know if performance is robust to less frequent updates, which could further improve efficiency.

3. **Minor Proofreading and Formatting Suggestions**: I noticed a few minor typos that could be easily addressed. Could the authors perform a final proofread to catch minor errors? A few examples I noted include:

    - Page 2, Figure 1: The label "Target Domian Extension" appears to have a typo. 

    - Page 3, Section 2.2: In the first paragraph, the corpus is described as something that "servers as the probing ground". The correct grammar would be "serves as the probing ground". 

    - Page 29, Appendix E: In the table 9, the method name "Importance Cumulatation" is misspelled. It should likely be "Importance Cumulation" or "Importance Accumulation".

### Soundness
3

### Presentation
4

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
The paper proposes ADEPT, a two-stage framework for domain-adaptive continual pretraining (CPT) of LLMs: (1) General-competence guided selective layer expansion—duplicate the least general-critical layers (identified via layer-ablation loss deltas) with function-preserving init; (2) Adaptive unit-wise decoupled tuning—partition each expanded layer into functional units and assign inverse-importance learning rates to protect general-critical units while adapting flexible ones. On math and medical corpora, ADEPT reports gains over full-parameter CPT and other baselines on both domain metrics and general benchmarks.

### Strengths
1. Clear, principled motivation from functional specialization observations (early layers more general-critical; heterogeneous unit importance), which directly informs where to expand and how to tune.
2. Sound expansion design via function-preserving identity copy (zeroing output projections) to avoid initial interference, consistent with Net2Net/FPI ideas.
3. Simple, implementable LR rule, periodically refreshed, giving a practical recipe for decoupled protection vs. adaptation.
4. Strong empirical results across Qwen3 (1.7B/4B/8B) and Llama3-8B on GSM8K/ARC and MedQA/MMCU/CMB; ADEPT often improves both domain and general benchmarks vs. full-CPT, replay, LLaMA-Pro, and LoRA/TaSL
5. Ablations and analyses show both stages matter; uniform expansion underperforms selective expansion; token-shift analysis suggests focused domain injection.

### Weaknesses
1. Importance identification and Fig. 2: 

(a) From the training-domain perspective, it seems more appropriate to identify domain-specific important layers, not just those important to the general domain. Would it further boost performance if expansion happens on important layers for the target training domain? Could you report per-domain importance profiles and their overlap with the “general-critical” set?

(b) This raises a potential domain conflict at expanded layers: a layer deemed “not important” for the general domain might be crucial for math, yet gets expanded and trained on medical data. How often does this conflict occur, and how does ADEPT mitigate it?

(c) MLP dominance. Fig. 2 suggests MLPs dominate general-knowledge importance. Is this driven by parameter scale or by an intrinsic representational role of MLPs? 

2. It makes sense that zero-initializing the expanded layer’s output projection, and it promotes stability, but if many MLP output projections are marked important, does zeroing them hinder adaptation? To ensure that forward computation remains
unchanged, making other modules (e.g., mlp gate/up projection) should also work.
 - Clarification: #264, does the ffn out projection mean down projection?

3. In Table 2, why does uniform expansion produce inferior results in the medical domain? considering it enables more adaptation capacity

### Questions
see weakness

### Soundness
4

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
3

### Summary
This paper proposes ADEPT which is a 2-stage framework for continual pretraining (CPT) of LLMs that aims to inject domain knowledge while avoiding catastrophic forgetting of general abilities. Stage1 selectively expands only those transformer layers that are least important for general competence. These layers are identified via a probing procedure that masks each layer and measures loss increase by duplicating them with function-preserving initialization. Stage2 decouples units such as attention and MLP, inside the expanded layers and assigns asymmetric learning rates inversely proportional to each unit's importance to the general domain, updating only the duplicates. Across Qwen3 models with sizes 1.7B, 4B, 8B and Llama3-8B, ADEPT reports higher target-domain accuracy and better retention on MMLU and CMMLU compare to the full CPT, LoRA, replay, and LLaMA-Pro, while tuning 15% of parameters and cutting training time more than 50%.

### Strengths
- The idea exploits functional specialization in LLMs where some layers and units are general-critical, while others are easier places to stuff domain knowledge. ADEPT explicitly measures that and routes updates accordingly.

- The method is simple with a mask-and-probe score to pick layers, a function-preserving copy to stay stable at step zero and a first-order importance signal to scale learning rates.

- On math and medical adaptation across several models, ADEPT generally boosts target-domain scores while avoiding the general-domain retention that is often seen after continual pretraining.

- The selective duplication step uses a straightforward probing signal and a function-preserving initialization to keep behavior unchanged at the start. The decoupled tuning step uses a first-order importance estimate to steer learning rates. It is kind of practical guardrail that you can integrate into a production CPT pipeline without rewriting everything.

### Weaknesses
- Layer masking for general ability and periodic unit-importance recomputation are not free. The paper shows strong results, but there is no crisp computation of how much extra compute the probing adds as models scale, or how sensitive it is to the quality or size of the probing corpus. 

- The authors fix the number of expanded layers, which works but looks hardcoded. An auto-selection strategy would make it easier and boost the performance.

- The evaluation is largely multiple-choice accuracy. That is fine for comparability, but I would love to see at least some open-ended reasoning or robustness/safety checks.

- Medical gains for Llama3 rely on 8-layer expansion given Chinese weakness. That may suggest that the performance may hinge on relatively heavy expansion in challenging cross-lingual settings. How does ADEPT fare on code, law, finance, or multilingual transfer beyond Chinese and English?

### Questions
- Can you quantify the total compute overhead of importance probing and periodic unit re-weighting, and show how it scales from 2B to 8B or beyond?

- Can you autotune the number of expanded layers under a fixed parameter/time budget, instead of fixing by hand?

- Paper says: "Experiments on mathematical and medical benchmarks show that ADEPT outperforms full-parameter
CPT by up to 5.76% on the general domain". I am confused by this, aren't general-domain benchmarks MMLU and CMMLU? But it says experiments on mathematical and medical benchmarks. What is that improvement?

### Soundness
3

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
The paper proposes Adaptive Expansion and Dynamic Decoupled Tuning for domain-adaptive CPT (ADEPT). The authors utilize a two-stage training process:  1) General-competence-guided selective layer expansion, and 2) Adaptive unit-wise decoupled tuning. 

In the first stage, the authors identify which layers are important for general knowledge and which are for domain knowledge by using curated probing data. In the second stage, they modularize the computation units and identify which units contribute to domain knowledge, updating these units with adaptive learning rates determined by gradient values. The authors use Llama3.1-8B and Qwen3 as base models and perform CPT for the math and medical domains. They experiment with model sizes ranging from 1.7B to 8B parameters and show that ADEPT outperforms the baselines.

### Strengths
- The paper is well written and easy to understand.  
- The proposed method is generally well-motivated.  
- The experiments are performed on various model sizes and show that the proposed method outperforms the baselines.

### Weaknesses
- The authors should conduct more experiments to justify their conclusions. Only MMLU and CMMLU are included to evaluate general knowledge, and only three domain-specific benchmarks are used. There are numerous benchmarks for both general and domain knowledge. To demonstrate that the proposed method is truly effective, the authors should include a more diverse and orthogonal set of benchmarks.

### Questions
- Does Table 6 in Section B.8 represent the end-to-end time from probing to training?  
- Is the proposed method applicable to SFT and RL?  
- How did you prepare $D_{\text{probe}} $?  
- What is "Vanilla"? Is it the base model without CPT?  
- Why is Vanilla better than PT-Full and Replay in the domain benchmarks for the 1.7B and 4B models? If training is properly executed, shouldn't they at least be better than the base model on the domain benchmarks?

### Soundness
3

### Presentation
3

### Contribution
2
