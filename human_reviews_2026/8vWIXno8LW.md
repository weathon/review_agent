# Sparse but Critical: A Token-Level Analysis of Distributional Shifts in RLVR Fine-Tuning of LLMs

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Reinforcement learning with verifiable rewards (RLVR) has significantly improved reasoning in large language models (LLMs), yet the token-level mechanisms underlying these improvements remain unclear. We present a systematic empirical study of RLVR’s distributional effects organized around three main analyses: (1) token-level characterization of distributional shifts between base and RL models, (2) the impact of token-level distributional shifts on reasoning performance through cross-sampling interventions, and (3) fine-grained mechanics of these shifts at the token level. We find that RL fine-tuning induces highly sparse and targeted changes, with only a small fraction of token distributions exhibiting meaningful divergence. We further characterize the structure of these shifts through analyses of token entropy, positional concentration, and reallocation of probability mass. To assess the functional importance of these sparse changes, we conduct cross-sampling experiments that selectively swap token choices between the base and RL models. Inserting only a small fraction of RL-sampled tokens into base generations progressively recovers RL performance gains, while injecting a similarly small number of base token choices into RL-generated responses collapses performance to base levels, isolating a sparse set of token-level decisions directly responsible for RLVR’s improvements. Finally, we explore divergence-weighted variants of the advantage signal as a diagnostic intervention, finding that they can yield improvements over baselines. Together, our results shed light on the distributional changes induced by RLVR and provide a fine-grained, token-level lens for understanding RLVR as a targeted refinement process.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the changes RLVR introduces by analyzing the token-level divergences between pre-trained and RLVR-trained versions of the Qwen2.5-7B model. The authors conclude that changes are concentrated in a sparse set of high divergence tokens, and conduct several experiments to validate and explain this finding: the cross-sampling analysis where high-divergence tokens are sampled from RLVR-trained model and put into the base model, and vice versa; the semantic analysis of changed tokens; the dynamics of changes during training.

### Strengths
S1. The results are novel to my knowledge and are presented in a concise manner; analysis in the paper is well-motivated; paper is dense and filled with results that generally support their claims and conclusions. Main claims of the paper are supported, but the execution of experiments might be significantly strenghtened further (see weaknesses).

S2. Experiments with cross-sampling and advantage reweighting are interesting and clever and seem to be a promising analysis toolkit; however, see W3 and W4.

### Weaknesses
W1. The most serious limitation of the work is that it lacks the treatment of other post-training schemes such as SFT, preference optimization or RLHF, so it is not clear if conclusions made in paper are specific to RLVR and not to fine-tuning in general.

W2. It is well-known that Qwen family shows different properties when training with RLVR on mathematics [1], and the paper does not study other families and tasks. Although these results are recent and the authors might be not aware about them when preparing the manuscript, this limits the generalisability of the paper findings and should be acknowledged in future work.

W3. All comparisons between GRPO and DAPO are affected by the different datasets the studied models were trained on, and this factor is not isolated. This is acknowledged in a paper.

W4. For experiment with cross-sampling, an important control baseline is missing - for example, the case when a token is sampled from another model randomly, instead of divergence-based rule, or something of that kind. This will quantify the effect of high-divergence replacements, and if control baseline would give lower final accuracy, then the conclusion made in line 268 "the improvements from RL fine-tuning are concentrated in a sparse set of high divergence tokens" would be reliable.

W5. Experiment with advantage reweighting in section 6.2 is underdeveloped, shows minor improvement, and it is unclear if the method allows for better results and stable training; another major shortcoming is that only DAPO baseline is analyzed. Since reweighting algorithm appears to use KL divergence, it is unclear how would this algorithm work if baseline algorithm already have KL divergence penalty (e.g. GRPO). Overall, this section is not very useful for the reader and further research in its current form, but the method seems interesting.

W6. As a minor weakness, although paper has a lot of results presented in a concise manner, it is densely written and it is slightly hard to parse and understand the meaning and interpretation of some results. I think that moving some content from main body to appendix and focusing on more clear presentation would improve the paper, but this is more suited for further iterations of the work.

### Questions
Q1. In figures 3 and 9, in the low JS bin, am I correct that RL entropy (red) almost zero and therefore is not visible? Please correct me if I misunderstood the section 3.3. Perhaps, the scatterplot or 2D histogram with JS / entropy axes would be effective; it also might be made for each sequence separately, and each point would represent separate token in a sequence.

Q2. Have you conducted an experiment where position would be measured not relative to the end of the sequence, but to specific elements of it (e.g. distance from system prompt, from the instruction, model generation start, from the final answer to the question etc.)? If yes, then what have you noticed? This is partially addressed by section 3.4; since the result in figure 10 suggest that single token might operate in low-divergence and high-divergence regimes, the positional information might reveal when some token would have high entropy and high divergence.

Q3. Is it possible to extend percentage of replaced tokens in the figure 5? Would trend be kept or will it be bounded from above by RL accuracy and from below by base model accuracy?

Q4. Results in section 5.1 are shown for high-divergence positions and therefore we expect differences to be noticeable; what changes if we do not restrict the positions and calculate this ranking results for all tokens? I also suggest to make shared x-axis for columns in figure 11 (b) so it would be easier to compare.

Q5. Section 3.2 about the dependency of entropy shift on position in a sequence is not substantive enough: absolute values of JS divergences shown for both methods are very low, and trend is seen only for DAPO. Also, since for DAPO the values are significantly larger than for GRPO (SimpleRL) as in section 3.1, it suggests that entropy shift might be the consequence of nonzero KL divergence regularization term and similar mechanisms to preserve the generation behaviour. Can you speculate if this is the case, or correct me if I'm wrong?

Q6. Have you gathered more results on advantage reweighting experiments? Currently, improvement seems very small. Have you identified for which scenarios and baseline algorithms this method is meaningfully applicable?

Considering cross-sampling experiment, you might be interested in checking out [2].

[1] Spurious Rewards: Rethinking Training Signals in RLVR, Shao et al.
[2] Reasoning with Sampling: Your Base Model is Smarter Than You Think, Karan A., Du Y., 2025.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper examines how token-level distributional properties shift during RLVR fine-tuning and explores the functional significance of these changes.

Empirically, the authors find that RL fine-tuning induces distributional divergence in only a small subset of tokens, mostly occurring early in the fine-tuning process. Notably, the magnitude of this divergence is not fully predicted by token entropy.

To assess the functional role of these divergent RL-sampled tokens, the authors inserted some of them into generations produced by the base model. They observed that this intervention caused the base model to perform as well as the RL fine-tuned model. Conversely, as expected, inserting base model tokens into the RL model’s outputs reduced its performance to base model levels.

Furthermore, the authors demonstrate that modifying GRPO/DAPO to weigh these divergent tokens differently—rather than treating all tokens uniformly—during fine-tuning results in improved downstream performance. This highlights a practical value for these divergent tokens.

### Strengths
I found the article to be generally well written. The motivation behind each analysis was clear, and the findings were explained well. The authors’ analyses were thorough—they not only examined how token distributions change, but also investigated which factors might predict these changes, the functional role of the divergent tokens, and how these insights can be leveraged to improve performance of RL fine-tuning.

### Weaknesses
It remains unclear under which specific sequences of tokens these observations were made, and clarifying this would enhance the paper. Additionally, the motivation for using JS divergence over KL divergence could be explained more thoroughly—currently, it is addressed in only a couple of sentences, but this section could be expanded with a simple illustrative example. Especially since the rest of the paper's relies on this observation. 

To streamline the narrative, consider moving certain experiments, such as the analysis of top-k token overlap, to the appendix, as these largely reinforce points already demonstrated by earlier results. At present, the paper contains too many small results, some of which make similar points using different approaches. This redundancy could be reduced; focusing on a few key analyses, with detailed motivation and deeper exploration of findings, would strengthen the paper.

Doing so, could also free up space for the final analysis, which felt somewhat crammed. I believe this final result was particularly important and deserved more attention than it currently receives.

### Questions
It would be good to verify whether these observations hold not only for the Qwen model family, but also extend to other open-source model families such as Llama or Mistral families.

I am also still unclear about why altering the placement of such a small number of tokens after RL fine-tuning can have such a strong impact on downstream performance. Is this function of the task/context within which these analyses were conducted as that is still unclear to me. Could the authors elaborate on their intuition for why this effect occurs?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates how Reinforcement Learning with Verifiable Rewards (RLVR) fine-tuning affects large language models (LLMs) at the token level. While RLVR is known to significantly improve LLM reasoning performance, the paper demonstrates that these improvements are not distributed evenly across outputs. Instead, RLVR induces sparse and highly targeted changes: only a small fraction of tokens in model responses are significantly altered by RL. Notably, the paper finds that these high-divergence token changes are not restricted to regions of high model uncertainty, but often override even confident base predictions. The authors conduct detailed empirical analyses to characterize these changes, evaluate their functional importance, and propose new RL objective variants to leverage these insights.

### Strengths
- The paper provides a novel token-level analysis of distributional shifts caused by RLVR, revealing sparsity and context-sensitivity in model updates that are not addressed by aggregate or entropy-based approaches in previous work. The authors introduce creative cross-sampling experiments that directly test the functional impact of high-divergence tokens and propose divergence-weighted RL objectives to exploit the observed sparsity.

- The empirical methodology is thorough and rigorous, comprising comprehensive token-level and sequence-level analyses, comparisons across several RLVR algorithms and model scales, and carefully designed interventions. Quantitative findings are supported by clear metrics (JSD, accuracy, percentage of affected tokens) and transparent reporting of results.

- The paper is well-organized and clearly written, with all key concepts (e.g., divergence, entropy, cross-sampling) explicitly defined and illustrated. Figures and tables are used effectively to present the main findings, and the logical structure guides the reader through the motivation, experiments, results, and implications.

- The work offers new insights into the mechanisms of RLVR, showing that performance gains stem from a small set of critical token-level changes. Its findings have practical implications for designing more efficient, interpretable, and targeted RLVR fine-tuning strategies for LLMs.

### Weaknesses
1. **Narrow dataset scope and lack of experimental details.** The experiments focus exclusively on math reasoning (AIME24) with Qwen2.5 models, so it is unclear whether findings about token-level sparsity and cross-sampling generalize to other domains, reasoning tasks, and models. The paper uses a fixed sampling setup (32 samples/problem, top-p=0.7, temperature=1), but omits ablations on these choices. No statistical significance or variance is reported for the divergence-weighted advantage gains in Table 2. JS divergence is calculated on top-p–truncated distributions, which can distort divergence magnitudes, yet the paper provides no discussion of this issue. The experimental setup is described at a high level, but several key implementation details necessary for full reproducibility (e.g., cross-sampling thresholds) are omitted.

2. **Limited causal and interpretability analysis.** While cross-sampling demonstrates the criticality of high-divergence tokens, the paper does not deeply investigate why these tokens emerge, what properties of inputs make them critical, or how specific token changes affect reasoning trajectories and outcomes.

3. **On-policy conditioning bias.** Token divergences are measured along RL trajectories only, which may bias where "critical" tokens appear.

### Questions
1. Can you provide results or analyses on non-math tasks or with other LLM architectures to support generalization claims?
2. How sensitive are your main findings to the sampling parameters (top-p, temperature, number of samples)? Have you run any ablations?
3. Can you report statistical significance or variance for the divergence-weighted objective gains in Table 2? Are these improvements robust across seeds or runs?
4. How does using JS divergence on top-p–truncated distributions affect your results? Can you show that your findings are not artifacts of truncation?
5. What exact divergence thresholds and implementation details were used in cross-sampling experiments? Can you provide all necessary information for reproducibility?
6. Have you analyzed what input features or contexts make high-divergence tokens critical? Any qualitative or quantitative results on the causes?
7. Could measuring divergences on off-policy or mixed trajectories affect the identification of critical tokens? Have you considered or tested for on-policy bias?

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
This paper analyzes how Reinforcement Learning with Verifiable Rewards (RLVR) modifies token-level output distributions during RL-tuning. The authors first examine distributional shifts between base and RL-finetuned models using Jensen-Shannon divergence, finding that RL-induced changes are highly sparse, with only a small fraction of tokens exhibiting significant divergence. Furthermore, through the Cross-Sampling experiments, they confirm that the effect of RL-tuning stems from the reranking of high-divergence tokens. Motivated by these findings, the authors propose a divergence-weighted advantage modification to the RL objective, showing potential improvement against existing RL-tuning.

### Strengths
1. The structure of this paper is clear; the authors first provide a thorough analysis of the token-level distribution shift and highlight the importance of the high-divergence tokens during RL fine-tuning. They then proposed an improved divergence-weighted objective based on their findings, making the paper sound and logical.
2. The analysis of token-level divergence and corresponding ablations is insightful and interesting. Although empirical, they are critical in understanding the operation of RL fine-tuning methods from the token-level aspect.

### Weaknesses
1. It is suggested to proofread the paper, especially the experiments. For example, in Table 2, the configuration details should be presented more clearly.  
2. The proposed method is evaluated in a rather limited setting; for instance, it is tested only on the AIME-2024 dataset. Moreover, the experiments lack comparisons with related works [1].  
3. The improvement achieved by the proposed method appears modest. According to the results in Table 2, the gain in Avg@32 is limited.  

[1] Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning

### Questions
1. What is the performance of the proposed method against other benchmarks?
2. In Table 2, what is the pass@1 performance?
3. What is the difference between emphasizing high-divergence and low-divergence during RL fine-tuning? In Table 2, both of these methods show similar performance gain. Could you explain this?

### Soundness
3

### Presentation
2

### Contribution
2
