# Steering MoE LLMs via Expert (De)Activation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Mixture-of-Experts (MoE) in Large Language Models (LLMs) routes each token through a subset of specialized Feed-Forward Networks (FFN), known as experts. We present SteerMoE, a framework to steer MoE models by detecting and controlling behavior-associated experts. We detect key experts by comparing how often they activate between paired inputs that demonstrate opposite behaviors (e.g., safe vs. unsafe). By selectively activating or deactivating such experts during inference, we control behaviors like faithfulness and safety without fine-tuning. Across 11 benchmarks and 6 LLMs, our steering raises safety by up to +20% and faithfulness by +27%. Alternatively, unsafe steering drops safety by -41% alone, and -100% when combined with existing jailbreak methods, bypassing all safety guardrails. Overall, SteerMoE offers a lightweight, effective, and widely applicable test-time control, while revealing unique vulnerabilities in MoE LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes an inference time MoE-based method to steer the output of the LLM towards user’s desired behavior. The SteerMoE method assumes two things: white-box access to the model, so you can read and adjust the router logits or their log softmax outputs at the target layer(s) during inference, and the availability of paired data to identify which experts in the MoE model are most associated with different behaviors (e.g., safe vs. unsafe responses). Experiments on faithfulness and safety benchamrks show that SteerMoE generally outperforms the baselines.

### Strengths
-	The paper is easy to follow, and clearly motivated.

-	The SteerMoE method is simple (a good example of Occam’s Razor in action) yet effective 

-	The SteerMoE method is efficient, as it requires no training and operates solely during inference.

-	The experiments are thoughtfully designed and cover a range of tasks, providing interesting insights. I really liked the jailbreak results in Table 2, where the latest models like GPT-OSS-120B show a drop in safety from 100% to 0%, highlighting how even simple methods can significantly compromise safety in some of the most advanced open-source models.

### Weaknesses
- It would be good to include a detailed algorithm outlining the method presented in Section 3 (including inputs, outputs, hyperparameters, steps, etc.). This would help clarify the approach and make it easier to understand and reproduce.

- Standard deviations or error bars are missing (e.g., in Figures 2 and 3, Table 2). It would be helpful to include error bars to account for randomness. Including error bars would improve the reliability of the results.

- The code is not provided. While the method seems simple and easy to implement, the absence of code raises concerns about reproducibility.

- The notation for $A$ and $N$ could be improved. It is unclear whether these variables are defined per layer or are averages across all MOE layers $\ell$.  If $A$ and $N$ are layer-specific, it would be helpful to add the layer index $\ell$ as a subscript or superscript.

- The method assumes the availability of paired data which might not be available in practice.

- The paper would benefit from a discussion of its limitations. Acknowledging potential weaknesses or assumptions would provide a more balanced perspective. 


Minor:
- typo in line 126: "4/32 and 4/128 in" should be "4/32 and 4/128 experts in".

- use same writing convention between different models between lines 125-134. Either use "... activates 4/32..." or use "... activates k=2 of E=32...".

- Line 139-140: missing references for ARXIV and PUBMED.

### Questions
-	Line 167-168: “…making it unsuitable in other settings”. What other settings? Could you explain a bit?
-	The method relies on the assumption that paired data is available. Could this assumption limit the applicability of the method in scenarios where paired data is not accessible? It would be helpful to identify these situations and discuss them in the paper.
-	Can the method be extended to scenarios with black-box access? If so, what are the key challenges and considerations to take into account?

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
3

### Summary
This paper proposes a novel way to steer Mixture of Experts LLMs at test time. In particular, given a set of paired examples containing the positive and negative behaviors towards a certain attribute, the proposed method in this paper activates and deactivates experts based on their behaviors. Experiments are carried out on two tasks (RAG and Safety) under different models and datasets showing positive results.

### Strengths
The main strengths of this paper are:

1) Simplicity and clarity: The proposed method is simple, clear, and effective

2) The paper is well-written. The related work section is informative and helpful.

3) The experiments in this work are thorough.

### Weaknesses
Despite the paper's strengths, there are few weaknesses and questions that need to be addressed:

1) The baseline SteerMoE compared against in this paper is the naive use of all experts. Why not comparing the proposed method to other MoE selection strategies from the literature?

2) I extremely like the experiments in section 4.2 - both the safe and unsafe steering vectors are acting as expected. Why not including similar experimental results in Section 4.1 (steering against faithfulness)?

3) The argument in page 9 regarding using the developed method as an attack is not convincing. The setting is unrealistic as the user has usually no control over which experts are employed (this is done at the API side), unlike the jailbreak and prompt injection attacks.

4) It is surprising that 3/4 models in Figure 3 achieve almost perfect score on most tested benchmarks. This might beg for including experiments under stronger attacks and more harmful datasets.

### Questions
Please refer to the weaknesses section. Overall, I am positive about this work. If my concerns are addressed (especially 1 and 4), I would be happy to raise my score.

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
**Core Problem and Approach:** SteerMoE addresses the challenge of controlling Mixture-of-Experts (MoE) large language model behavior at inference time without requiring retraining or weight modification. The framework identifies behavior-associated experts by computing risk difference scores—comparing expert activation rates between paired inputs demonstrating opposite behaviors (e.g., safe vs. unsafe prompts, document-grounded vs. parametric responses). During inference, the method selectively activates or deactivates these identified experts by adjusting router logits, enabling lightweight behavioral steering. This weight-preserving approach treats the MoE router as an interpretable control mechanism rather than merely a computational distribution tool, allowing real-time modulation of model behaviors like faithfulness and safety.

**Experimental Results and Critical Findings:** Across 11 benchmarks and 6 LLMs, SteerMoE demonstrates significant improvements: up to +27% in faithfulness (measured on FaithEval, CF-TriviaQA, and MQuake datasets using SQuAD for expert detection) and +20% in safety (evaluated on AdvBench, MaliciousInstruct, and TDC2023 using BeaverTails for detection). However, the research reveals a critical vulnerability: unsafe steering reduces safety by -41% independently, and when combined with existing jailbreak methods like AIM, achieves -100% safety on GPT-OSS-120B, completely bypassing all guardrails. This exposes a form of "alignment faking" where safety training concentrates in specific expert subsets while neglecting alternate routing paths. The findings indicate that behavior-critical experts cluster in middle layers, and that current alignment strategies must extend beyond surface-level tuning to ensure all expert routing paths maintain robustness and safety.

### Strengths
- Operates entirely at inference without modifying weights or requiring retraining. Uses simple risk difference scoring to identify behavior-linked experts and adjusts routing via logit manipulation—lightweight, fast, and applicable across 6 different MoE architectures with consistent effectiveness.
- Demonstrates both benefits (+27% faithfulness, +20% safety across 11 benchmarks) and critical vulnerabilities (unsafe steering achieves -100% safety on GPT-OSS-120B when combined with jailbreaks), providing the community with actionable insights for both alignment improvement and security hardening.
- Reveals that behavior-critical experts cluster in middle layers with clear token-level correspondence (safe experts fire on safe tokens). This creates practical applications beyond steering: low-cost hallucination detection, input attribution, and exposes "alignment faking" where safety concentrates in sparse expert subsets while unsafe routing paths remain exploitable.

### Weaknesses
SteerMoE detects experts on narrow datasets (SQuAD, BeaverTails) and evaluates on similar distributions, leaving unexplored whether steering transfers to qualitatively different attacks (multilingual, encoded, social engineering), domain shifts (scientific papers, multi-document reasoning), or adaptive adversaries who optimize prompts to evade detected experts.

### Questions
- Can you provide transfer results on qualitatively different settings, such as MultiJail?

[1] Deng Y, Zhang W, Pan S J, et al. Multilingual jailbreak challenges in large language models[J]. arXiv preprint arXiv:2310.06474, 2023.
[2]

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
This paper introduces SteerMoE, a framework for steering Mixture-of-Experts (MoE) LLMs by detecting and controlling behavior-associated experts. The method compares expert activation rates between paired inputs with contrasting behaviors using risk difference, then selectively activates/deactivates experts at inference time. Experiments on faithfulness (+27%) and safety (+20% safe, -100% when jailbroken) across 6 models and 11 benchmarks show strong results while revealing concerning vulnerabilities in MoE alignment.

### Strengths
Novel and timely contribution: Treating MoE routing as a controllable interface is creative and under-explored. The weight-preserving, training-free approach is also practical.

Empirical results: Authors show some improvements on faithfulness and safety benchmarks across multiple state-of-the-art MoE models (GPT-OSS, Qwen3, Mixtral, etc.).

Important security findings: The demonstration that alignment is concentrated in specific routing paths, enabling catastrophic safety bypasses (-100% on GPT-OSS-120B), is a considerable contribution if this is well executed. It is indeed very relevant for future MoE safety research.

Comprehensive evaluation: Testing across 6 models and 11 benchmarks with appropriate control tasks shows thoroughness.

Interpretability insights: The Figures A.4-A.6 provide valuable analysis showing behavior-relevant experts cluster in middle layers with interpretable token-level patterns.

### Weaknesses
1. Experimental Issues

My main concern with this paper is the lack of a strong experimental evaluation.

 There is an absence of naive and baseline solutions. The selection of experts should  be compared against random selection, the bottom K experts, etc. We need more ablations.

Furthermore, there is no comparison to other steering methods (representation engineering, activation patching, prompting strategies).

2. Methodological Concerns

i) Detection phase: For this phase, safety detection analyzes tokens after "Assistant:" in prompted examples, but actual unsafe generation patterns may differ during free generation. I’m slightly concerned about the difference between the generation and the prompted examples. 

ii) Hyperparameter selection (Table A.1): The fact that experts modified varies wildly (0-500 activated, 0-480 deactivated) with no principled selection method, raises some concerns about generability and practical utility. 

3. Limited Theoretical Motivation

i) Risk difference justification: While Appendix A.1.1 provides intuition, there's no formal analysis of why RD is optimal or under what conditions the method works/fails. Have authors tried different scores?

ii) Steering mechanism: The log-softmax + ε adjustment (Eq. 6-8) appears ad-hoc. Why is this the right way to modify routing? Why not choosing mean values, baseline values, etc.? An ablation study would help here.


**Minor issues**

Critical details missing:  size of detection datasets used, exact procedure for computing risk differences, how are the numbers in Table A.1 determined? The computational cost is lacking and many "under review" references seem to be incomplete.

### Questions
Could authors provide samples of the dataset?

### Soundness
2

### Presentation
3

### Contribution
2
