# Accelerating Large Language Model Inference via Speculative Decoding with Progressive Tree Drafting

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 6

## Abstract
The draft-then-verify decoding paradigm, introduced by speculative decoding methods, has demonstrated remarkable performance in alleviating the memory-bound bottleneck and accelerating the inference speed of Large Language Models (LLMs) while maintaining the quality of generated content. Recent studies show that the intrinsic robustness of LLMs can be exploited in a training-free and architecture-agnostic manner, suggesting that auxiliary models or structural modifications are not strictly necessary for draft generation. However, existing methods fail to fully leverage this robustness, leading to substantial redundant and repeated computations. Building on this insight, we propose Progressive Tree Drafting (PTD), a new inference acceleration strategy that further extends this line of work. PTD organizes the drafting process into a progressively updated tree structure, where controlled perturbations are injected to guide generation and a stepwise pruning mechanism enabling the model to produce coherent yet diverse drafts at manageable computational cost. By efficiently coordinating the drafting and verification stages, PTD achieves up to 2$\times$ decoding speedup across different open-source models and benchmarks. Our code is available at https://anonymous.4open.science/r/PTD-D354.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
1

### Summary
This paper proposes Progressive Tree Drafting (PTD), a new speculative decoding method that does not rely on an additial draft models and integrates tree-style drafting and verification process. At each drafting step, given a randomized initial tree, the model generates draft tokens after the given tree and a stepping step is conducted. When the draft tree reaches a certain depth, a pruning step is conducted to reduce the number of nodes in the tree. PTD introduces the on-tree drafting, pruning, and tree-verification to single-model speculative decoding, achieving a significant speedup ratio. Empirical results on MT-Bench compared with baselines (Speculative Decoding, Lookahead Decoding, and Self-Draft) fully show the efficiency of PTD.

### Strengths
1. Single-model speculative decoding is good scenario to focus on, since methods such as EAGLE cannot always be applied in some scenarios where no draft models can be deployed.

2. Integrating tree-verification to single-model speculative decoding is a reasonable and efficient design. Unlike previous works, such design can reduce the redundancy of the drafting process, thereby increasing the decoding speed.

### Weaknesses
1. The presentation of this paper is quite confusing. Most of the proposed methods are densely packed into Sections 3.2 and 3.3, filled with unclear formulas and redundant definitions. The pseudocodes in Appendices A and B are also difficult to follow. The authors should consider reorganizing these sections by removing redundant concepts, adding illustrative figures, and presenting the proposed methods in a clearer and more structured manner.

2. Since this paper focuses on single-model speculative decoding, the authors need to explicitly explain why this setting is emphasized and under what circumstances a draft model cannot be used. The current discussion in the related work section does not make this point clear, leaving readers confused about why the EAGLE series are not suitable for all speculative decoding scenarios. For example, this can be clarified from a GPU-memory perspective: a draft model requires additional GPU memory, and Transformer-based draft models further consume memory through their KV-cache, which becomes a serious limitation for long-context inputs.

3. The proposed method should be compared against stronger baselines such as REST [1] and PLD [2], which also operate without a draft model. If the scope of the paper is limited to single-model speculative decoding, it is unclear why SpeDe (the vanilla speculative decoding method) is still treated as a baseline.
---

[1] REST: Retrieval-Based Speculative Decoding

[2] https://github.com/apoorvumang/prompt-lookup-decoding

### Questions
1. How do the authors position PLD within the speculative decoding community? What is its most defining characteristic: is it a training-free approach, a single-model (no draft model) design, or a faster speculative decoding method? Since the paper is not clearly written, I am unable to determine this with confidence, and therefore cannot fully assess the soundness of the experimental design. The authors should clearly highlight PLD’s key feature in the experimental section by selecting appropriate baselines and reporting metrics that correspond to that feature.

2. The method section is also difficult to follow, which prevents me from forming a confident evaluation, despite my familiarity with speculative decoding methods. Therefore, I have assigned the lowest confidence rating to my review. I strongly recommend the authors provide further clarification and discussion of the method’s logic and assumptions. However, if the above issues remain unresolved, I suggest that the AC consider lowering the weight of my decision in the final evaluation.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the memory-bound bottleneck of autoregressive inference in Large Language Models (LLMs). It focuses on the "draft-then-verify" paradigm of speculative decoding, specifically on training-free methods that use the target LLM itself to generate drafts. The authors identify a key limitation in prior work like Self-Draft: the generation of redundant and highly similar draft branches, which leads to wasted computation. To solve this, the paper proposes Progressive Tree Drafting (PTD), a novel training-free and model-agnostic inference strategy. Instead of generating simple linear branches, PTD organizes the drafting process into a tree structure that is progressively expanded and pruned. This approach uses controlled perturbations and a custom tree-based attention mask to guide the LLM in generating a diverse set of candidate drafts simultaneously. The tree structure allows for efficient prefix sharing, while a stepwise pruning mechanism controls the computational cost. Experiments show that PTD achieves significant throughput improvements (up to 2x) compared to autoregressive decoding and outperforms other training-free baselines like LADE and Self-Draft across various benchmarks.

### Strengths
1. The paper clearly articulates a practical bottleneck in multi-branch linear drafting, where insufficient draft diversity produces redundant candidates and wastes computation.
2. The paper proposes a progressive tree–based strategy that adaptively prunes tree width and depth to improve draft diversity.

### Weaknesses
1. Although PTD starts from random branching, dynamic depth and width control with known tree-pruning schemes is already mature in SD with integrated tree verification (e.g., EAGLE-2, SWIFT, Spec-LLaVA, SpecVLM), so the novelty beyond applying these pruning schemes to multi-branch drafting is unclear. The paper should explicitly argue why existing tree-pruning methods are insufficient for multi-branch drafting, how PTD differs or extends them, and include focused comparisons in Related Work and Experiments.
  - EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees
  - SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration
  - Spec-LLaVA: Accelerating Vision-Language Models with Dynamic Tree-Based Speculative Decoding
  - SpecVLM: Enhancing Speculative Decoding of Video LLMs via Verifier-Guided Token Pruning
  - ProPD: Dynamic Token Tree Pruning and Generation for LLM Parallel Decoding
  - Faster Speculative Decoding via Effective Draft Decoder with Pruned Candidate Tree
  - OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure
  

2. The current presentation appears to equate “diversity” with pruning of duplicate candidates. This risks an overclaim in the introduction: while true diversity could raise acceptance rates, the method section primarily describes deduplication—a compute-saving step that reduces drafting cost and, at best, maintains the same acceptance rate.

3. The evaluated models are somewhat dated, and there is a lack of comparison with the more superior methods recognized by communities such as Medusa and EAGLE series. In addition, Fig. 8 suggests limited speedups on Qwen models; analyze why PTD underperforms there. The paper does not study how progressive width and depth are determined/tuned, nor the sensitivity of PTD to these choices—important under the self-drafting paradigm.

4. The paper’s writing and organization could be further improved, for example (including but not limited to) the following:

  * In Fig. 1(a), the introduction of perturbations is too abrupt. This prerequisite should be explained with concrete examples—what kinds of perturbations are used and how they produce multiple branches.
  * In Fig. 1(b), the workload of the preliminary study is not well described. I am very curious how duplicate branches or tokens behave across different tasks, and whether there are more promising insights to enable a more flexible, domain-/context-aware dynamic tree design instead.
  * Also in Fig. 1(b), please define precisely what constitutes a “duplicate step” (as I understand, generating the same token at the current step) versus a “duplicate branch” (the historical decoding tokens are also identical), and how similarity is measured.
  * Fig. 2(c) makes the PTD pipeline hard to understand—I don’t know what the inputs and outputs are, nor where to start reading the diagram.

### Questions
Please refer to the weaknesses.

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
3

### Summary
This paper proposes an inference acceleration method named Progressive Tree Drafting (PTD), a training-free and model-agnostic approach to speculative decoding. The core motivation is to address the computational redundancy in existing self-drafting methods like Self-Draft, which arises from the high similarity among draft branches. PTD organizes the draft generation process by maintaining a dynamically expanding and pruned tree structure, aiming to generate draft sequences that are both diverse and coherent through structured perturbations.

### Strengths
1. Strong Novelty: The core idea of a "Progressive Drafting Tree" is novel. Structuring the draft generation process into a controllable tree, which uses prefix sharing and pruning to balance draft diversity, coherence, and computational cost, is a valuable contribution.
2. Clear Motivation: The paper clearly identifies a key bottleneck in existing methods (computational redundancy) by analyzing the branch similarity of Self-Draft (Figure 1b), making the proposed solution highly targeted. Furthermore, the experimental evaluation covers a range of mainstream open-source models and diverse tasks (dialogue, math, code), demonstrating the method's generalizability.

### Weaknesses
1. Rough Writing and Presentation: The paper's overall writing and structure appear unpolished. Table captions are brief and lack critical information; for instance, the caption for Table 1 does not clarify which baseline (AR) the "Imp." (Improvement) is relative to. More seriously, the baseline results for Speculative Decoding (SpeDe) on several models (the Qwen series) are marked with a backslash (`\`) without any explanation, which undermines the rigor and completeness of the experiments.
2. Questionable Output Consistency: In theory, speculative decoding with rejection sampling should be lossless, meaning its output distribution is identical to that of the original autoregressive model. However, the ROUGE/BLEU scores in Appendix Table 2 are far from 100, suggesting that PTD alters the model's output in sampling mode. The authors must explain the source of this discrepancy.
3. Vague Methodological Details: The description of key details is vague. For instance, the initialization of the draft tree with "randomly initialized perturbation tokens" in Section 3.2 is not clearly explained.

### Questions
The current version suffers from shortcomings in its writing, presentation, and experimental rigor. I encourage the authors to add more details and further polish this novel method.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Progressive Tree Drafting (PTD), a training-free speculative decoding framework for LLM inference acceleration. Building on the insight that LLMs exhibit semantic robustness under controlled perturbations, PTD extends prior approaches like Self-Draft by organizing draft generation into a progressively expanded tree structure with prefix-sharing, branch pruning, and branch-wise perturbation. The method generates multiple coherent draft sequences and verifies them in parallel without requiring auxiliary draft models or architectural changes. Experiments across multiple LLaMA, Qwen, and CodeLLaMA models show up to 2× throughput improvement, outperforming LADE and Self-Draft, particularly on GSM-8k and MBPP, with competitive generation quality.

### Strengths
1. Method is well-motivated with illustrative diagrams (tree expansion, mask structure) and formal algorithm descriptions.

2. The experimental evaluation covers general QA, math reasoning, and coding tasks, with consistent speedups. The ablation studies on tree depth, branch width, and sampling strategies clearly demonstrate the characteristics of the proposed method.

3. The results provide stronger performance than both LADE and Self-Draft, highlighting real gains over recent state-of-the-art baselines.

### Weaknesses
1. Limited comparison to tree-based speculative frameworks (e.g., SpecInfer, EAGLE-2 dynamic draft trees).

2. The method still introduces non-trivial verification overhead, and overhead trends at larger scales (>32B parameters, >4K tokens) are not fully explored.

3. The choice of specific hyperparameters (e.g., max children=4, depth=6) appears arbitrary without an analysis of their impact across different model scales or task types.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3
