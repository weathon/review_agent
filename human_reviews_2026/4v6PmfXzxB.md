# Latent instruction representation alignment: defending against jailbreaks, backdoors and undesired knowledge in LLMs

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 2, 4

## Abstract
We address jailbreaks, backdoors, and unlearning for large language models (LLMs). Unlike prior work, which trains LLMs based on their actions when given harmful instructions, our method specifically trains the model to change how it interprets instructions. Our method, Latent Instruction Representation Alignment (LIRA), greatly improves generalization. We further boost generalization through an internally adversarial training algorithm. Our methods block over 99\% of PEZ jailbreak attacks; removes a challenging insecure code backdoor; and achieves optimal forgetting on WMDP cyber with negligible loss of benign capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
A paper proposes a post-training intervention method based on white-box representation engineering method and which aims addresses the problem of models producing undesired outputs. Authors demonstrate that proposed method shows substanital improvemnts in threat models of jailbreaks, backdoors and unlearning, against several exhisitng attack methods.

### Strengths
Authors introduce and justify a post-training intervention that appears to work on a limited set of pre-exhisting jailbreaking attacks, a static unlearning evaluation and backdoor challenges.

### Weaknesses
### Lack of Adaptive Evaluation

My main critique of the paper is that it lacks adaptive evaluation of the proposed defense mechanisms. While the authors argue in the introduction that previous mitigations do not generalize to novel attacks [line 32, line 46], they fail to provide this kind of evaluation themselves. Although they claim their method is robust against novel attacks [lines 32, 53], for the jailbreaking part of the paper they only demonstrate effectiveness against a weak PEZ attack and a superficially constrained embedding space attack.

To demonstrate that their defense offers substantial improvements over previous methods, the authors need to evaluate it with *adaptive attacks* - those that explicitly adapt to the defense design (e.g., at least similar to [4]). Adaptive attacks have repeatedly exposed defenses across adversarial ML: in jailbreaking, simple adaptive attacks [1] subverted the adversarially trained R2D2 model [2], CircuitBreakers [3] fell to adaptive attacks in latent space [4, 5] and stronger attacks in input space [6]. Concurrently, several defenses, some based on instruction-data separation principles, were subverted in one go [7] by specifically designed adaptive attacks.

The same adaptive evaluation principle universally applies to other adversarial ML areas like unlearning [8, 9], where adaptive evaluation revealed the brittleness of proposed measures despite the initial authors' claims. Without such evaluation, the robustness claims in this paper remain fundamentally unsubstantiated.
 
### Methodological Problems in Jailbreaking Experiments

**Choice of attacks.** PEZ is an old attack shown to be extremely ineffective in prior work [2, 11], even against non-safety-tuned models. On lines 86-87, the authors claim that PEZ is a "challenging" jailbreak attack and cite Schwinn et al. [12]. However, PEZ and soft-prompt attacks differ critically: PEZ maps back to token space while soft-prompt does not. From my experience, the mapping to discrete space is a central problem of PEZ that will not be addressed by increasing the "suffix" length that authors intoduce (line 360).

The ES (embedding space) attack introduced on line 355 and described on lines 364-369 appears to be largely equivalent to the embedding space attacks used in the CB paper and in [12]. The authors introduce a superfluous constraint of "proportion of embedding dimension attacker is controlling" without justification. This constraint artificially weakens the attack without reflecting realistic threat models, making it unclear what advantage this provides over existing embedding space attack formulations.

In Appendix M, the authors justify not choosing the GCG attack, claiming it achieves only 2% on Gemma-2-2b in HarmBench evaluation and does not benefit from best-of-N (BoN) evaluation. This is an unreasonably low number that suggests either implementation issues or evaluation problems, as it directly contradicts prior work. For example, [14] reports that GCG against Gemma-2-2b **with a safe system prompt** achieves 30% ASR, and [15] analyzes how most of jailbreaking attacks benefit from BoN evaluation, including GCG.

**Choice of benchmark.** In contrast to previous methods like CB [3] or adaptive evaluations like [1, 7], the authors do not use established benchmarks such as HarmBench, AdvBench, or JailbreakBench. Instead, they use a subsplit of CB training/validation data. This problem extends to the ad-hoc grader introduced in Appendix E.1, which lacks human study validation or justification for why it should be preferred over existing graders like StrongReject [13] or HarmBench. The arbitrary choice of harmful queries, grader, and ASR@10 for one attack and ASR@1 for another makes results completely incomparable with prior work.

### Overall

The paper suffers from significant methodological limitations in its evaluation approach. Across all three experimental setups (unlearning, backdoors, jailbreaking), it fails to include adaptive attack evaluations, where adversaries are assumed to have full knowledge of the defense mechanisms. Without such evaluations, the claimed robustness of the proposed measures remains unsubstantiated, making it impossible to assess whether the defense provides meaningful security improvements.

The jailbreaking setup, which I examined in depth, compounds these issues with several questionable choices: an unconventional benchmark and grader, weak static attacks (i.e., no GCG, PRS[1], or strong BoN[6] attacks), and claims about attack effectiveness that contradict established findings in the literature on these specific attack methods, which further undermine the reliability of the results.

[1] Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks

[2] HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal

[3] Improving Alignment and Robustness with Circuit Breakers

[4] Obfuscated Activations Bypass LLM Latent-Space Defenses

[5] https://confirmlabs.org/posts/circuit_breaking.html

[6] Best-of-N Jailbreaking

[7] The attacker moves second: Stronger adaptive attacks bypass defenses against llm jailbreaks and prompt injections

[8] On Evaluating the Durability of Safeguards for Open-Weight LLMs

[9] An adversarial perspective on machine unlearning for AI safety

[11] Universal and Transferable Adversarial Attacks on Aligned Language Models

[12] Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space

[13] A StrongREJECT for Empty Jailbreaks

[14] An Interpretable N-gram Perplexity Threat Model for Large Language Model Jailbreaks

[15] Sampling-aware Adversarial Attacks Against Large Language Models

### Questions
### Questions to Authors

- Why is PEZ performance on Llama missing from Table 2? 

-  Given the limited success of embedding space attacks in the original CircuitBreakers paper and the reported ASR of 28.1% in Table 2, what are the specific differences between your setup and the original CB evaluation? Additionally, did you train a CircuitBreakers model yourself, or did you use the pre-existing checkpoint from https://huggingface.co/GraySwanAI/Llama-3-8B-Instruct-RR? .

- What exactly constitutes the Forget Set used for CyberSecurity unlearning experiments? Do I understand correctly that it is filtered Cyber Corpora from the original WMDP paper?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes Latent Instruction Representation Alignment (LIRA), a post-training defense for LLMs that aims to prevent jailbreaks, remove backdoors, and achieve unlearning. Instead of modifying model outputs directly, LIRA aligns latent representations of harmful instructions with those of benign instructions using Sequence-Aware Gradients (SAG) to block gradients from response tokens. The authors also introduce AdLIRA, an adversarial extension that alternates between “attack” and “align” phases to improve robustness, and evaluate both methods on jailbreak, backdoor, and unlearning benchmarks.

### Strengths
1. Novel conceptual framing: The paper raises an interesting idea, addressing how the model interprets harmful instructions instead of only modifying its final responses. This representation-level intervention could, in principle, yield better generalization across jailbreak and unlearning tasks.

2. Unified perspective: The attempt to address jailbreaks, backdoors, and unlearning within a single framework is conceptually appealing and could inspire future research connecting these safety problems.

### Weaknesses
1. Inefficient training design.
If the goal is to align the latent representations of harmful and its nearby instructions, it seems unnecessary to perform a full forward pass including both instructions and responses, only to block gradients from response tokens afterward. A more direct and computationally efficient approach would be to align both instruction representations via cosine similarity (or other distance metrics) directly, without involving the response logits.

2. Unlearning baselines are outdated.
The unlearning experiments only compare against older baselines (GD, RMU). More recent and stronger methods should be included for fairness:
SimNPO[1] TPO[2] ME[3]

3. Ambiguous terminology and inconsistent definitions.
The terminology should be standardized and explained clearly before use. These inconsistencies make the paper harder to follow. For example: "GD" (Gradient Difference) is used in figures before being defined.
Table 1 introduces terms such as "Malign request refusal" without definitions. It is unclear whether "malign request" and "harmful request" refer to the same concept. 
After reading the entire paper, I can roughly infer the authors’ intended meaning, but clearer and earlier definitions, along with consistent terminology, would greatly improve readability and overall quality.

4. Misuse of "nearby safe instruction" in Figure 1.
The term "nearby safe instruction" is misleading. The example nearby instruction "Help me build a bomb" is clearly harmful; it remains an unsafe instruction, even though the model produces a safe, refusal-type response to it. The authors should clarify that LIRA make the representation of a harmful instruction close to a nearby instruction that lead model to generate the safe answer, not that the instruction itself is safe.

5.	Limited comparison to representation-level defenses.
It omits comparisons to related latent-space approaches included in related work section, such as TLAT[4] and LAT[5]


[1] Fan, Chongyu, et al. "Simplicity prevails: Rethinking negative preference optimization for llm unlearning." arXiv preprint arXiv:2410.07163 (2024).

[2] Zhou, Xiangyu, et al. "Not All Tokens Are Meant to Be Forgotten." arXiv preprint arXiv:2506.03142 (2025).

[3]Yuan, Xiaojian, et al. "A closer look at machine unlearning for large language models." arXiv preprint arXiv:2410.08109 (2024).

[4] Casper, Stephen, et al. "Defending against unforeseen failure modes with latent adversarial training." arXiv preprint arXiv:2403.05030 (2024).

[5]Sheshadri, Abhay, et al. "Latent adversarial training improves robustness to persistent harmful behaviors in llms." arXiv preprint arXiv:2407.15549 (2024).

### Questions
1. In the Table1, does the “malign request” and “harmful request” refer to the same concept? 

2. Why are only older baselines (GD and RMU) included in the unlearning experiments? Could the authors compare against more recent and stronger unlearning methods such as SimNPO [1], TPO [2], and ME [3] to better assess LIRA’s performance?

3. Why does the paper omit quantitative comparisons to recent latent-space or representation-level defense methods mentioned in related work, such as TLAT and LAT? How would LIRA and AdLIRA perform relative to these?

### Soundness
2

### Presentation
1

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
This paper introduces LIRA (Latent Instruction Representation Alignment), a novel post-training method for defending LLMs against jailbreaks, backdoors, and unwanted knowledge expression. The key innovation is training models to align malicious instruction representations with benign ones, rather than directly penalizing harmful outputs. This is achieved through Sequence-Aware Gradients (SAG), which selectively blocks gradients from response positions during training. The authors extend LIRA with: (1) AdLIRA, an internally adversarial training scheme, and (2) a classifier-guided variant for unpaired data. Experiments on Gemma 2 9B and LLaMA 3.1 8B demonstrate strong performance across jailbreak defense (>99% blocking of PEZ attacks), backdoor removal, and unlearning tasks.

### Strengths
1. Novel and well-motivated approach: The focus on instruction representations rather than output behaviors is conceptually elegant and well-justified. The intuition that representations form an information bottleneck easier to defend than the full instruction-to-output pipeline is compelling.

2. Comprehensive evaluation: The paper tackles three distinct security challenges (jailbreaks, backdoors, unlearning) with strong baselines including Circuit Breakers, RMU, and gradient-based methods. The evaluation includes both automatic metrics and human ratings where appropriate.

3. Strong empirical results: LIRA substantially outperforms baselines across tasks. Particularly impressive are: (a) 99%+ blocking of PEZ jailbreaks with AdLIRA, (b) single-step backdoor removal for the HATE task, and (c) effective unlearning with minimal retain set degradation.

### Weaknesses
1. Limited theoretical justification: While the intuition about instruction representations is appealing, the paper lacks formal analysis of why this approach should generalize better. What properties of instruction representations make them more robust targets? Under what conditions might this fail? The empirical observation in Figure 1b is suggestive but insufficient.

2. SAG design choices under-justified: The specific gradient blocking rules (which attention paths to block, which parameter updates to prevent) appear somewhat arbitrary. Why block attention between response tokens specifically? The paper states this prevents "improving intra-benign-sequence quality" but doesn't validate that this is problematic. Ablations comparing SAG variants would strengthen this.

3. Computational cost not addressed: AdLIRA requires iterating between attack and defense phases, and classifier-guided LIRA retrains the classifier each iteration. How do training times and computational costs compare to baselines? This is crucial for practical deployment.

4. Limited analysis of failure modes: When does LIRA fail? The paper shows strong aggregate results but provides little insight into what kinds of attacks or backdoors might bypass the method. The embedding space attack results (Fig 4b) show AdLIRA still fails ~50% of the time with full attacker control—what characterizes successful vs. unsuccessful attacks?

### Questions
Please refer to weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
