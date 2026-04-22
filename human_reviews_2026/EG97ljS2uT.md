# Stealthy Jailbreaking Attacks via Hyperbolic Hamiltonian Dynamics and Möbius Fusion

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 2, 8

## Abstract
Recent studies on jailbreaking attacks have shown the vulnerability of large language models (LLMs) to malicious questions. 
Existing jailbreaking attack methods often rely on disfluent or incoherent prompts, which limit their success and make them easy to detect. We introduce SJA, a structured jailbreak attack that overcomes these weaknesses through two key ideas. First, inspired by the logic of Spilsbury puzzle, SJA decomposes a harmful query into a sequence of harmless sub-questions and reconstructs the original answer by combining the sub-question responses. Second, by leveraging the theory of Hamiltonian dynamics on hyperbolic space, we propose a hyperbolic Hamiltonian dynamics-based sub-question generation framework that effectively captures the structural and temporal dependencies. We provide a theoretical analysis of how each sub-question evolves along the trajectory and show that the hyperbolic Hamiltonian system effectively captures the underlying semantic structure. Finally, we propose a hyperbolic narrative fusion mechanism built on fractional embedding and Möbius fusion. This mechanism integrates coherent narratives into sub-questions while preserving geometric consistency and improving stealth performance.
We theoretically validate that the combination of the generated harmless sub-questions, guided by the stealthy narrative, can effectively preserve the contextual semantics of the original harmful question.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a method called Stealthy Jailbreaking Attack. The method decompose the original malicious query into several harmless sub-questions together to achieve attack in multi-round of intereactions.The harmful query is first embedded into hyperbolic space 
and sub-question is represented as an embedding evolving along a Hamiltonian trajectory to ensure they are semantically aligned. The fused hyperbolic embeddings are mapped back to Euclidean space when decoding by LLM.

### Strengths
1. The paper is well written and the figures are well designed.
2. The paper structure is well organized.

### Weaknesses
I have concerns about both the theoretical (many claims lack of support)and experimental aspects of the paper.
1. The authors claim to each subquestion’s semantic representation with its momentum to form a state pair and to use a Hamiltonian system. However, the Hamiltonian system in this paper appears to be only a formal analogy, there is no guarantee that the semantic evolution can actually be described or integrated under a symplectic structure. How do the authors justify that the language representation space satisfies the assumptions of a Hamiltonian system?

2. The author in Theorem 2 proposes a mapping from Euclidean space to hyperbolic space and claims it can preserves direction and local semantics. However, I believe here is a just  heuristic operation. The author need to provide a proof of Riemannian geometric consistency, and error bounds to support this claims.

3. The Möbius-addition in narrative fusion provides neither curvature-control conditions nor a convergence analysis, so the claim of “preserving geometric consistency” is unspported.

4.  My more serious concern is in experiment.  Firstly, the experiment dataset is small, only 50*3. I am not requiring more experiment, but a larger dataset will make the experiment more convincing. Such as HEX-Phi[1]

5.  The experiment setting seems problematic. The paper use default parameter settings of each baseline methods. However the [2] work already show that the sampling parameters can significantly influence the result of attack. In other words, the baseline methods results are misaligned. And also for SJA, the author use parameters searching, this is unfair comparsion.

6. The ablation study of the paper is not well-designed. The paper claims that Hamiltonian dynamics can improve semantic coherence.  The authors should compare it against stochastic interpolation methods and provide quantitative metrics such as s-bert[3] to demonstrate the improvement.

7. Strong relevant methods such as cresendo[4] not included in the experiment 

In my view, the major issues of this paper lie in its novelty and motivation. Although the supplementary material mentions related prior work, the authors do not provide a systematic analysis of the disadvantage of existing approaches,do not clearly explain how their method differs from previous multi-turn jailbreak(semantic decomposition) and the motivation of introducing hamiltionian system. The claimed innovation also suffers from both theoretical assumption and experimental setting. It is over-packaged relative to its actual contribution.

[1]https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI

[2] Huang, Yangsibo, Samyak Gupta, Mengzhou Xia, Kai Li, and Danqi Chen. "Catastrophic jailbreak of open-source llms via exploiting generation

[3]Reimers, Nils, and Iryna Gurevych. "Sentence-bert: Sentence embeddings using siamese bert-networks." arXiv preprint arXiv:1908.10084 (2019).

[4]Russinovich, Mark, Ahmed Salem, and Ronen Eldan. "Great, now write an article about that: The crescendo {Multi-Turn}{LLM} jailbreak attack." 34th USENIX Security Symposium (USENIX Security 25). 2025.

### Questions
See weakness.

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
5

### Summary
This paper focuses on how to overcome the disfluent or incoherent features of existing jailbreaking attack methods and implement stealthy Jailbreak Attacks to LLMs. It proposes a method called SJA to decompose a harmful query into a sequence of harmless sub-questions and reconstruct the original answer by combining the sub-question responses, leveraging the theory of Hamiltonian dynamics on Hyperbolic space. Besides, it applies Mobius fusion to integrate coherent narratives into sub-questions while preserving geometric consistency and improving stealth performance.

### Strengths
1. Innovative Application of Hyperbolic Geometry: The paper's primary strength is its well-motivated and effective use of hyperbolic geometry. The authors correctly recognize that the semantic dependencies within a decomposed query form a hierarchical structure. By leveraging hyperbolic space—which is inherently suited for low-distortion embedding of such structures—they establish a robust geometric foundation for their method. This is a sophisticated application of geometric principles to a complex NLP problem.  
2. Principled Formulation of Hamiltonian Dynamics. This paper introduces a highly novel and principled method for sub-question generation by drawing a compelling analogy to Hamiltonian dynamics. The mapping of sub-question embeddings to generalized coordinates (q) and their target semantic directions to generalized momenta (p) is both clever and rigorously defined. This allows the generation process to be framed as a trajectory optimization problem, ensuring the resulting sequence is not only coherent but evolves along a smooth, logical path.

### Weaknesses
1. While the paper provides a rigorous mathematical and empirical demonstration of its method's effectiveness, it could be strengthened by a more intuitive explanation for why this specific optimization framework successfully bypasses the model's safety alignment. The connection between the mathematical properties of the sub-questions (e.g., forming a smooth trajectory) and the resulting behavioral shift in the LLM (from refusal to compliance) is not fully elucidated from a cognitive or AI psychology perspective. Without it, readers may default to the assumption that SJA is simply another form of multi-turn attack, with the underlying intuition being to 'numb' the model's safety responses over several turns.

2. While the paper demonstrates impressive results across a wide range of current models, the discussion on the method's generalizability to future architectures and its long-term viability (timeliness) is somewhat limited. The landscape of LLMs is evolving rapidly, models like Claude 3, Qwen3, GPT-5 featuring significantly more advanced safety alignment, and even those with architectures like Mixture-of-Experts (MoE) becoming standard. The paper does not explicitly discuss how SJA might perform against these newer paradigms. For instance, it is an open question whether the token routing mechanisms in MoE models could be leveraged to build more robust defenses (e.g., via specialized 'safety experts'). Similarly, future models may possess enhanced high-level intent recognition that could detect the malicious trajectory of SJA's sub-questions, even if each step is individually benign. A forward-looking discussion on the potential robustness of SJA against these next-generation models and defenses would have strengthened the paper's claims of being a general-purpose attack framework.

### Questions
1. I am particularly curious about the implementation details of two critical steps in the SJA framework.
Could you elaborate on the specific mechanism used for the HyperbolicEmbed function? Specifically, how are the text-based semantic directions $v_i$and the initial query $q$ first converted into Euclidean vectors before being mapped into the hyperbolic space? Was a specific layer or output from the Llama2-7b-hf model used for this initial vectorization?
Similarly, regarding the final decoding step (Step 7), how is the fused embedding $\tilde{q}_{i}
$ from the hyperbolic tangent space incorporated into the Llama2-7b-hf decoder to generate the natural language sub-question? Is it injected as a soft prompt, added to the input embeddings, or used in another capacity to steer the generation?

2. Sensitivity to the Choice of the Decomposition Model: As rightly pointed out in the limitations section, the initial decomposition of the harmful query into semantic directions is a critical bottleneck for the entire SJA pipeline. The paper exclusively uses Llama2-7b-hf for this task. I am curious about the sensitivity of the overall attack performance to the choice of this decomposition model. Have the authors experimented with using other models (e.g., more powerful ones, or even the target models themselves in a black-box query) for the decomposition step? It would be valuable to understand how the quality and nature of the initial directions, as produced by different models, impact the final success rate of the SJA framework.

3.  Comparison with Adaptive Multi-Turn Attack Strategies: The execution of the SJA attack, which involves sequentially submitting sub-questions in a single conversation, shares a similar interactive format with the multi-turn attack methods mentioned in the related work (e.g., those that are adaptive and persona-based). I would be very interested to see a direct experimental comparison between SJA and representative SOTA methods from this category [c1-2]. Such a comparison would help to quantitatively assess the benefits of SJA's structured, offline-optimized approach versus the more dynamic, online-adaptive strategies.

[c1] A Mousetrap: Fooling Large Reasoning Models for Jailbreak with Chain of Iterative Chaos.  
[c2] Derail Yourself: Multi-turn LLM Jailbreak Attack through Self-discovered Clues

4. I suggest the authors conduct experiments on Harmbench or other latest jailbreak benchmarks.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a multi-turn jailbreak method for large language models, which breaks a harmful question into a series of innocuous sub-questions, collects their harmless answers, and recombines them to recover the original intent. It uses a mathematical framework, drawing on hyperbolic geometry and Hamiltonian-style trajectories to generate sub-questions that keep structural and temporal relationships, and a narrative-fusion step using fractional embeddings and Möbius-style combination to make those sub-questions read coherently. Experimental results demonstrate the effectiveness of their method.

### Strengths
The experiments on single-turn jailbreak attacks are relatively thorough.

### Weaknesses
1.	There are several multi-turn attack approaches whose ideas are very similar to the method proposed here, but the paper does not compare with or analyze them. Works such as [1] and [2] decompose a harmful objective into a series of harmless queries and then progressively recover the original intent through semantics. I only listed a subset of related papers — the authors should discuss these works more fully in the Related Work and Experiments sections.

2.	More importantly, I have concerns about the rationality of their method. When generating multi-turn queries they start from a relatively generic prompt and add scene descriptions, but this can introduce significant semantic drift. Take Figure 1 as an example: the queries their method generates steer the model to output content about chemical experiments, which is far less harmful than actually building a bomb.

3.	There is a large gap between their theoretical claims and their implementation, which further weakens the contribution of the work.


References

1.	Great, now write an article about that: The crescendo multi-turn LLM jailbreak attack.
2.	Derail yourself: Multi-turn LLM jailbreak attack through self-discovered clues.

### Questions
See the limitations.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes SJA, a novel framework for generating stealthier and more effective jailbreak attacks on LLMs. Unlike prior methods that rely on incoherent or easily detectable prompts, SJA decomposes a harmful query into a series of semantically aligned and logically ordered sub-questions inspired by the Spilsbury puzzle. Using hyperbolic Hamiltonian dynamics, it models the structural and temporal dependencies among sub-questions, while a Moebius fusion mechanism integrates a shared narrative to maintain coherence and geometric consistency. Theoretical analysis shows that SJA preserves the contextual semantics of the original harmful intent, and empirical results demonstrate higher attack success rates and stronger evasion performance across multiple models and datasets.

### Strengths
The paper introduces a new attack paradigm on hyperbolic space, decomposing a harmful query into a sequence of semantically coherent, logically ordered, and individually harmless sub-questions, then reconstructing the original answer. The use of hyperbolic Hamiltonian dynamics on hyperbolic space to model the semantic trajectory of sub-questions is technically original and well-motivated. The framework explicitly captures both structural and temporal dependencies needed to reliably reconstruct the original intent.

### Weaknesses
* The proposed attack comprises seven steps, but the ablation studies do not demonstrate whether each component is necessary. An ablation that removes or replaces e.g. the Hamiltonian dynamics and other key modules would clarify each component’s contribution.

* It remains unclear to what extent the proposed approach outperforms directly prompting LLMs to generate sub-questions based on semantic directions.

### Questions
* Compared to perturbations in Euclidean space, is there empirical evidence demonstrating the advantages of performing attacks in hyperbolic space?

* It is still unclear how final subquestions are generated given the embeddings in the last step. Are embeddings provided as part of input prompts, as shown in the Appendix? Why it is a good strategy?

### Soundness
3

### Presentation
2

### Contribution
3
