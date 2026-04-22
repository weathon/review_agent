# Exploring the Trade-off between Quality and Diversity of Language Models during Reinforcement Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 2, 8

## Abstract
Reinforcement learning (RL) has become the dominant approach for post-training autoregressive language models, but a recurring challenge is that improvements in *quality* often come at the expense of *diversity*, which is a practical concern in exploratory domains such as scientific discovery. Although this trade-off is widely acknowledged, it has lacked a quantitative characterization. In this work, we systematically investigate the quality-diversity dynamics of RL finetuning of language models \hl{primarily} on molecular generation, a domain where diversity is both essential for discovery and quantitatively measurable. 

Across RL checkpoints, we observe that mean quality ($\mathcal{R}$) and diversity ($\mathcal{D}$) trace a smooth trajectory captured by a robust exponential law, $\mathcal{R}=-a\cdot\exp(c\cdot\mathcal{D})+b$, independent of step indexing. Extending prior work on quality-entropy trade-offs, we further show that quality also follows an exponential relation with sampling entropy ($\mathcal{H}$), $\mathcal{R}=-a_0\cdot\exp(c_0\cdot\mathcal{H})+b_0$, with $c_0$ quantifying exploratory progress. An approximately linear link between entropy and diversity explains why the two laws compose, and an information-theoretic illustration clarifies the role of the exponential form. We also conduct ablations on influencing factors, including model scaling, reward shaping, and training setup, validate these findings across multiple generation objectives, and extend the experiments to textual exploratory creativity tasks by large language models. Finally, we demonstrate how the fitted laws provide actionable guidance for RL finetuning of language models on exploratory tasks. Overall, our study moves beyond qualitative accounts of diversity collapse, offering a compact quantitative model, an underlying entropy-based mechanism, and practical tools for exploratory RL with language models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores the relationship between quality and diversity in molecular generation, proposing an exponential law between the two.

### Strengths
The paper employs a meaningful, task-dependent diversity metric to construct and validate the proposed relationship.

Comprehensive experiments are conducted across models of different sizes and under varying reward-shaping mechanisms.

The analysis spans multiple datasets within the same domain, enhancing the robustness of the empirical findings.

### Weaknesses
My primary concern is whether the paper's conclusions are, as claimed, general for language models. While I appreciate the authors’ use of domain-related metrics, the meaning of diversity can vary drastically across tasks. In the theoretical analysis, the paper assumes—without sufficient justification—that molecular diversity and sampling entropy are linearly related. This assumption significantly limits the generalizability of the proposed conclusions to other tasks or diversity metrics.

And some additional unexamined assumptions include:

The assumption of uniform distributions over subsets $U$ and $Q$ is unrealistic for trained language models.

The approximation $|S\cap T|\approx |T|^\eta$
, which lacks theoretical grounding, and the connection between $\eta$ and actual training dynamics remains unclear.

Consequently, the “information-theoretic explanation,” while intuitively appealing, is not mathematically rigorous—it serves more as a heuristic argument than a formal derivation.

From the RL algorithm chosen, the authors employ standard RL methods in molecular generation, which differ from reinforcement learning methods commonly used in current language models (e.g., GRPO, PPO). The treatment of regularization and entropy loss plays a crucial role here. Validating the proposed law across a broader set of reinforcement learning algorithms would strengthen the claim.

Overall, the paper frequently uses “language model” in a general sense, which may overextend the applicability of the proposed exponential law.

### Questions
Can authors give more explanation on why $\eta$ remains constant throughout the entire training process?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper probes the inner connection between generation quality and diversity of language models during reinforcement learning (RL). The authors claimed that the quality and diversity fits an exponential correlation in molecule generation task with the mathematical induction for the tradeoff. In experiments, the results demonstrates the empirical conclusion with a very high confidence (R^2=0.996).

### Strengths
1. The paper presented a simple yet reasonable experience on the tradeoff between quality and diversity with clear math intution. Also, it also probed the connection with entropy. 

2. The experiment design is systematic and comprehensive, including model scale, pretraining data, RL settings, sampling, and redudant removal. The evaluation pipeline seems reasonable and easy-to-implement.

3. The connection is robust across different model scale, reward function, and multiple molecule datasets. 

4. There is valuable discussion about the strategies based on the connection during reinforcement learning finetuning, which is inspiring.

### Weaknesses
1. The external validity is narrow, where the results are only shown for molecular generation, yet the authors claimed at the broader generality.

2. The diversity measurement is limited (solely on mean ECFP/Tanimoto distance). There should be more comprehensive metric based on scaffold and 3D compositions, or motif. 

3. There exists bias for the sampling entropy and the sampling settings (temperature, top-k, etc.), which may directly affect the performance of entropy.

4. There may exist reward hacking for the oracle during RL finetuning. multi-oracle or experimental validation can be added. 

5. The "Algorithm-agnostic" claims are unsupported since only REINVENT algorithm was applied. There should be more algorithms to demonstrate that claim.

6. The theory in section2 mainly depends on strong and untestable assumptions, serving more like an illustration rather than a derivation. 

7. Noveltu vs training data lacks. There should be analysis and results to demonstrate that there was not leakage during training and testing. 

8. Do the R-D and R-H laws still hold across multiple, independently trained JNK3 oracles and additional orthogonal property oracles?

9. There should be more analysis to the sensitivity of the cofficients a,b,c, since they greatly affect the optimal point selection during applications given the law holds ideally.

### Questions
Please refer to the Weakness section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates the quantitative relationship between quality and diversity in reinforcement learning (RL) fine-tuning of language models, an important issue since RL-based optimization (e.g., RLHF) often improves quality at the cost of diversity. The authors empirically find that mean reward (R) and diversity (D) across RL checkpoints follow a consistent exponential law, and discuss why this occurs via an information theoretic argument. Overall, the paper argues that this exponential law captures a fundamental quality-diversity frontier intrinsic to the task, not to model scale or RL algorithm.

### Strengths
1. To a certain extent, the paper provides a novel, quantitative framing of diversity collapse in terms of a measurable, predictable law. It also provides a compact, generalizable model R with clear empirical support.

2. The experiments are performed quite well. Specifically the authors provide well-controlled experiments in molecular generation where quality and diversity are objectively quantifiable. The authors employ replication across objectives (JNK3, GSK3b, QED) and scaling ablations to show robustness. FItted parameters are also reported with their respective confidence intervals.

3. Since the paper links entropy, diversity, and reward through an information-theoretic lens, we gain interpretability beyond curve fitting.

### Weaknesses
1. All experiments are on SMILES-based molecular generation with hand-crafted chemical scores. Molecular generation is a special case with well-behaved reward landscapes. It remains unclear whether other exploratory domains (e.g., creative writing, theorem discovery) yield the same exponential frontier.
2. The authors acknowledge that natural language tasks involve fuzzier definitions of “diversity,” so generalization to text LMs is uncertain.
3. The experiments are performed on tiny models (6–50 M parameters); thus the findings may not hold for billions-scale LLMs where policy entropy and reward landscapes differ drastically.
4. Sampling entropy is treated as a scalar average over tokens; it ignores inter-token dependencies or contextual variation that might distort the R–H relationship.
5. The exponential “law” is descriptive; the precise causal mechanism (entropy contraction → quality improvement) is plausible but not explicitly tested via interventions or ablations (e.g., controlled entropy regularization).

### Questions
1. How general is the exponential law. Does it hold for open-ended text generation or only in chemically structured spaces?
2. How should practitioners interpret c_0? Can it be controlled or estimated apriori?
3. How sensitive is the law to the diversity metric (e.g., Tanimoto vs internal representation–based distance)?
4. Could entropy–diversity linearity break for non-SMILES tokenizations or natural language byte-pair encodings?

### Soundness
2

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
4

### Summary
This work studies quality-diversity tradeoffs in autoregressive transformers during post-training, specifically RL. They use molecular structures as a data domain, which enables them to rigorously assess both quality and diversity. This domain also motivates diversity for the purpose of discovering new molecules. They find that quality-diversity tradeoffs are very well-explained by a simple exponential model, which is motivated by theories of an exponential relationship between entropy and quality and a linear relationship between entropy and diversity. Their model fits are highly consistent across transformer model scales and training objectives, with different models and objectives having different values for free parameters in their exponential model.

Overall I think this paper tackles an important problem with a novel domain and perspective (molecular synthesis), presents interesting empirical results, impressive fits with a novel and well-motivated exponential model, and is generally well-written. Solid accept.

### Strengths
- Interesting empirical phenomena. The consistency of the exponential relationship between quality and diversity is remarkable. 
- The exponential model is well-motivated and the fits are exceptionally good.
- Good choice of domain. Studying molecular structures lets the authors evaluate "semantic" (although they don't refer to it as such) quality and diversity.
- Generally well-written and clearly presented. Despite my having relative little chemistry background, I had no trouble reading relevant parts of this work.

### Weaknesses
- Section 2.3
	- I found this section very terse and took some effort parse, though this could be related to my not having a strong background in information theory. E.g. I had to do some googling to understand the meaning of `≍` and typical set. Readability could be improved by making this more accessible to a general audience, and/or by adding relevant background to the appendix. 
	- More importantly, I'm not sure the authors fully answer the 3 questions they state at the beginning of this section. I found these questions compelling and was asking myself all 3 before I arrived at this section. 1. seems to be explained by a one-line reference to another paper "|T | = exp(H) by information-theoretic arguments (MacKay, 2003)", which leaves me wondering what those arguments are exactly. 2. and 3. seem to be explained, but I found the answers to these questions somewhat unintuitive in comparison to their very straightforward framing. Spelling out the answers in simple language might go a long way here.
- Section 3.3 - I found these suggestions to be weak and I'm not sure if they clearly follow from the authors' results: 
	- 1. seems to highlight a result the authors don't seem to test - that pre-training indeed changes the initial point. And, do the authors have particular suggestions for improving R_max? "by RL finetuning techniques" is a bit vague
	- 2. is this really practical guidance? This point seems to be about the generality of the authors' results
	- 3. Were mixtures or ensembles tested here? If not, then what motivates this suggestion?
- Could use better literature review on quality-diversity tradeoffs in LLMs, e.g. [1, 2]. More of this in the intro could help better set the stage



[1] Kirk, R., Mediratta, I., Nalmpantis, C., Luketina, J., Hambro, E., Grefenstette, E., & Raileanu, R. (2023). Understanding the effects of rlhf on llm generalisation and diversity.

[2] Murthy, S. K., Ullman, T., & Hu, J. (2024). One fish, two fish, but not the whole sea: Alignment reduces language models' conceptual diversity

### Questions
- Some points in early sections could be more clearly connected to later sections, e.g. "Notably, this law is independent of step index or training efficiency." -- this could reference particular experiments related to this (3.1.2, in my understanding)
- "... we do not accumulate molecules sampled across RL steps" I'm not sure what this means, could you please clarify?
- The organization read a little weird for me, where the introduction seemed to do more than what an intro usually does.
- I think the strength of this domain in measuring quality and diversity could be more clearly articulated and emphasized - why is this superior to domains that prior works have used?

### Soundness
3

### Presentation
3

### Contribution
4
