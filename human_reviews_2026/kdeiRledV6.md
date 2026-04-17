# Overthinking Reduction with Decoupled Rewards and Curriculum Data Scheduling

- Decision: Accept (Oral)
- Scores: 10, 2, 6, 8

## Abstract
While large reasoning models trained with critic-free reinforcement learning and verifiable rewards (RLVR) represent the state-of-the-art, their practical utility is hampered by ``overthinking'', a critical issue where models generate excessively long reasoning paths without any performance benefit. Existing solutions that penalize length often fail, inducing performance degradation due to a fundamental misalignment between trajectory-level rewards and token-level optimization. In this work, we introduce a novel framework, DECS, built on our theoretical discovery of two previously unaddressed flaws in current length rewards: (1) the erroneous penalization of essential exploratory tokens and (2) the inadvertent rewarding of partial redundancy. Our framework's innovations include (i) a first-of-its-kind decoupled token-level reward mechanism that surgically distinguishes and penalizes redundant tokens, and (ii) a novel curriculum batch scheduling strategy to master the efficiency-efficacy equilibrium. Experimental results show DECS can achieve a dramatic reduction in reasoning tokens by over 50\% across seven benchmarks while simultaneously maintaining or even improving performance. It demonstrates conclusively that substantial gains in reasoning efficiency can be achieved without compromising a model's underlying reasoning power. Code is available at \url{https://github.com/pixas/DECS}.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
2

### Summary
Handling RLVR model’s overthinking behavior by separating tokens according to their role and penalizing redundant tokens, and scheduling batch with curriculum.  
–
Prior approaches incorporating length penalties (e.g., Eq. 4) broadcast trajectory-level rewards to all tokens via GRPO's advantage estimation (Eq. 2). When all responses in a group are correct but vary in length, shorter sequences receive positive advantages while longer ones receive negative advantages uniformly across every token. Consequently, high-entropy tokens like "Wait" or "However"—which initiate productive exploration—are suppressed simply because they appear in longer trajectories, even when they are part of the necessary reasoning process (Lemma 2, Theorem 1). DECS addresses this by decoupling rewards at the token level: it assigns maximum reward (r₊) to all NRP tokens including high-entropy exploratory tokens, while inversely penalizing only the redundant tokens generated after the NRP (Eq. 9). This surgical approach maintains exploratory capacity (preserving Pass@K performance, Fig. 3c) while achieving superior compression (>50% token reduction, Table 1).

### Strengths
- The idea is backed by theoretical claims, and they are logically correct. The two flaws revealed by Lemma 2 and Theorem 2 (suppression of high-entropy tokens + failure to penalize redundant tokens) are addressed by DECS through curriculum scheduling (Theorem 1) and decoupled rewards.
- In addition, the paper make each claim irrefutable through quantitative verification at every level - from theory to core analysis (Fig 1, 3), then validate broadly with 7 benchmarks x 2 models x 5+ baselines, with robustness checks (Tables 5-7)
- The paper organized their contents very nicely, well distributed content between main text and appendix. 
- Good generalization to out-of-domain tasks (LiveCodeBench, GPQA-D)

### Weaknesses
Limited accessibility for non-expert readers: While the paper presents rigorous theoretical analysis and strong empirical results, the presentation could be more accessible. Figures 1-2, which are central to understanding the core contributions, are dense with notation that is not immediately intuitive without careful reading of Section 3. For a paper introducing novel concepts like "Necessary Reasoning Prefix" and "decoupled token-level rewards," more pedagogical support would strengthen impact.

Suggestions for improvement:
- Add a concrete, annotated example in Appendix showing token-by-token reward assignment for a simple problem (e.g., "What is 2+3?"). Compare how (a) base GRPO, (b) length penalty, and (c) DECS assign advantages to specific tokens including high-entropy ones.
- Figure 2, while comprehensive, relies heavily on abstract labels ("NRP", "Redundancy", "Correct Thinking Process") that may not be 
immediately intuitive to readers unfamiliar with the concepts. Consider replacing these placeholders with concrete text snippets from an actual model response (e.g., showing "Wait, let me verify... 2+3=5" as NRP vs "Let me triple-check... yes, 5" as redundancy) within the same figure space. This would make the core contribution—that DECS protects exploratory tokens while pruning redundancy—immediately graspable without requiring back-reference to Section 3.
- Include an intuitive explanation of why high-entropy tokens matter, possibly with a probability distribution visualization showing how entropy differs between "5" (low H) vs "Wait" (high H).

### Questions
* Q1: The paper presents strong technical contributions, but accessibility could be improved with concrete worked examples (e.g., token-by-token walkthrough of the "2+3" problem showing advantages under baseline vs DECS). Given the paper is already dense, where do you envision such pedagogical content fitting best?
* Q2: Table 1 shows 7B has less overthinking than 1.5B initially. Does this trend continue? Would larger models need DECS?  (Scalability to 30B+ models unverified)

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper addresses the "overthinking" problem in large reasoning models trained with reinforcement learning, where models generate excessively long reasoning paths without performance benefits. The authors propose DECS (Decoupled rewards and Curriculum data Scheduling) to reduce the reasoning length while not harming the performance.

### Strengths
- The studied problem is important;
- The writing is easy to follow;

### Weaknesses
- The proposed reward and curriculum designs are quite artificial. In particular, the decoupled reward depends on pre-training an effective NRP detection model, which introduces practical challenges given that the definition of NRP is ambiguous, let alone the difficulty of reliably detecting it with another model.
- The method is not straightforward to implement, as it requires pre-training a separate detection model whose accuracy cannot be guaranteed or easily verified.
- The curriculum design component lacks novelty and, as such, does not constitute a substantial contribution to the paper.

### Questions
See the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DECS, a novel framework designed to mitigate "overthinking", the generation of excessively long reasoning paths, in large reasoning models (LRMs) trained with reinforcement learning. The authors first provide a theoretical analysis identifying two key flaws in current length-penalty methods: the erroneous penalization of essential exploratory tokens and the inadvertent rewarding of redundant tokens. To address these, DECS incorporates three main components: (1) a lightweight judge model to detect the Necessary Reasoning Prefix (NRP), (2) a decoupled token-level reward that specifically penalizes tokens generated after the NRP, and (3) a scheduling strategy to balance efficiency and performance during training. The authors conduct extensive experiments, demonstrating that DECS can reduce the number of reasoning tokens while maintaining or even improving performance.

### Strengths
1. The paper provides a clear theoretical analysis of why existing length-reward mechanisms are suboptimal.
2. The DECS offers a carefully designed approach grounded in theoretical reasoning.
3. The authors evaluate their method across diverse benchmarks, use two different model scales, and compare against multiple relevant and strong baselines.
4. DECS achieves a reduction in token count while simultaneously improving performance.

### Weaknesses
1. The framework's reliance on a separate, fine-tuned NRP detector adds complexity to the training pipeline. A deeper analysis of its robustness would strengthen the paper.
2. The decoupled reward is used to differentiate between "Leading Redundant tokens" and other redundant tokens, but this does not appear to be reflected in Eq.9.

### Questions
1. I am intrigued by the parallels between the "erroneous penalization" issue and concepts like relative overgeneralization in MARL; a brief comment on this could enrich the paper's context within the broader RL literature.
2. Could the authors elaborate on the reward design shown in Fig.2 and Eq.9?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper targets the “overthinking” phenomenon on LRMs. Based on the fundamental misalignment between trajectory-level rewards and token-level optimization. The authors propose a DeCS framework based on theoretical discoveries that are erroneous penalization of essential exploratory tokens and  inadvertently rewarding partial redundancy. The DeCS framework includes (1) decoupled token-level reward mechanism and (2) curriculum batch scheduling strategy. Through comprehensive experiments the authors demonstrate effectiveness on their proposed framework

### Strengths
**\[S1\] Solid Theoretical Foundation**  
The proposed framework is developed through strong theoretical foundation that enhance its novelty and provide convincing justification for its effectiveness. Also the paper's clear definition of concepts like NRP will likely provide significant value to the field by establishing precise terminology

**\[S2\] Comprehensive Experimentation and Analysis**  
The paper presents a satisfying experimental structure organized around research questions, including thorough comparisons with baseline methods and detailed ablation studies. This approach effectively addresses potential questions about the proposed framework while convincingly demonstrating its effectiveness.

### Weaknesses
**\[W1\] Readability Issues**  
While the logical flow of the paper is solid, it was difficult to understand due to the extensive use of abbreviations and mathematical notations. Additionally, the token length comparison in Table 1 would have been more effectively conveyed through visualization. 

**\[W2\] Limited Model Generalization**  
The experiments are confined exclusively to the DeepSeek model family. The paper would be strengthened by extending experimentation to model families with different exploration capabilities to demonstrate broader generalization of the proposed framework.

**\[W3\] Domain Dependency of the NRP Classification Model.**   
While the NRP classifier trained on OpenR1-Math-220K demonstrates strong reliability through human validation within the mathematical domain, the paper lacks analysis of its performance in non-mathematical domains (e.g., coding, science, etc.). It will be great for the authors to additionally verify whether the NRP classification remains robust and consistent across these domains.

### Questions
**\[Q1\]** In the RQ2 experiment, I understand that the exploration potential maintains similar with the base model. I'm curious about how exploration compares between DeCS  and GRPO-trained models. What differences or similarities exist in the exploration behavior between DeCS and models trained with GRPO?

**\[Q2\]** Table 1 indicates that the proposed DeCS framework demonstrates lower average performance metrics compared to GRPO-trained models. Furthermore, the manuscript predominantly focuses on comparisons with baseline methods and analyses of individual DeCS components, while direct comparative analysis with GRPO appears somewhat limited. Given these observations, are there any theoretical reasons why DeCS shows a lower performance compared to GRPO?

### Soundness
4

### Presentation
3

### Contribution
4
