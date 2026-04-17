# CoDaPO: Confidence and Difficulty-Adaptive Policy Optimization for Language Models

- Decision: Reject
- Scores: 6, 2, 8, 2, 4

## Abstract
Reinforcement learning (RL) post-training strengthens reasoning in large language models (LLMs), yet the prevailing GRPO algorithm exhibits persistent issues. Using a PRAG lens (Probability, Reward, Advantage, Gradient), we diagnose three mechanisms: (i) _probability inflation_—clipping induces one-way confidence drift with weak KL correction, collapsing entropy; (ii) _advantage contraction_—group normalization dulls update signals as accuracy rises; and (iii) _hierarchical convergence_—easy questions improve quickly while hard ones advance slowly via rare discoveries. We then introduce _CoDaPO_, a confidence- and difficulty–adaptive policy optimization framework that rescales per-trajectory advantages by confidence (curbing overconfidence and drift) and difficulty (sustaining learning on hard questions). Across seven benchmarks, CoDaPO demonstrates improvements on mathematical reasoning benchmarks for small and middle-scale models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work analyzes limitations of current RL post-training methods for LLMs reasoning, especially the GRPO algorithm. Through a PRAG (Probability, Reward, Advantage, Gradient) diagnosis, the authors identify three issues: 1) probability inflation, 2) advantage contraction, and 3) hierarchical convergence. To address these, they propose CoDaPO—a confidence- and difficulty-adaptive policy optimization framework that rescales training advantages based on model confidence and problem difficulty. Experiments across seven benchmarks show that CoDaPO yields consistent performance gains, particularly improving mathematical reasoning in small and medium language models.

### Strengths
1. Provides a deep empirical and theoretical analysis of GRPO’s training dynamics in reasoning-oriented RL post-training for LLMs.
2. Proposes a solid and novel algorithm that effectively addresses the identified issues.
3. Presents comprehensive experiments covering various base models, reasoning benchmarks, training dynamics, and ablation studies.

### Weaknesses
2. Extend the evaluation to domains beyond mathematics to verify generality of the new algorithm. 
2. Include additional analyses on the computational efficiency of the proposed algorithm compared with baseline methods.

### Questions
Please see weakness above

### Soundness
3

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
4

### Summary
This paper does an empirical analysis of GRPO (or more precisely, a variant of it called DAPO), through the lens of model miscalibration during training on MATH, and disentangling across 5 levels of difficulty of problems.

### Strengths
The main strengths of the paper are:
- the empirical analysis of the authors is solid and their findings are reasonable - particularly, I found interesting the beginning of section 3.2 on miscalibration 
- the authors find that applying the same advantage for every token is detrimental for learning, which makes sense given that there are probably few tokens that actually matter for the solution and many filler ones, especially in math proofs (such as "thus", "therefore" etc)

While the analysis is interesting, I believe there are several major weaknesses with the paper which I present below.

### Weaknesses
- To begin, it would be good if the authors would give quantitative analysis of the distributions in figure 2 and 3 (namely, min, max, mean, median and kurtosis) since currently it is quite hard to verify qualitatively the statements about tails in Section 3.3

- My second major issue with the current paper is that it does not appear that CoDaPO improves over the baselines (DAPO, and GRPO) in Table 2. Concretely, the improvements are at most 2%, which in general is not statistically significant for post training due the RL noise. Moreover, as far as I understand, the current comparison does not sweep over hyperparameters, and thus it is even more unclear whether the proposed would have any benefit at all under a well tuned baseline.

As a meta comment, while I think the empirical analysis is quite solid and interesting, the paper would be better framed as a science paper studying GRPO rather than proposing a new method which as it stands does not bring any improvements over existing algorithms. Due to this, I am inclined to reject this work.

### Questions
My questions have been posed in the weaknesses section.

### Soundness
3

### Presentation
4

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper defines and analyzes three existing problems in the existing GRPO mechanism and proposes an optimization algorithm, CoDaPO, targeting to address these problems by embedding the notion of model confidence and problem difficulty into the optimization objective. CoDaPO demonstrates improvement over alternative optimization methods on average in a suite of mathematical reasoning benchmarks on the 1.5B and 7B scale Qwen models. The paper also includes a set of ablation studies on the component of CoDaPO.

### Strengths
- The paper clearly identifies the existing problems in GPRO in formal analyses, highlighting the motivation of their method very well.
- The paper formally shows how the introduction of confidence and difficulty would address these problems, making the design choices of method convincing.
- The paper provides a good amount of illustrations that are helpful in understanding the analysis and the impact of CoDaPO intuitively.

### Weaknesses
- The experiments in the paper are only performed in the math domain for both training and evaluation. Whether the method can be extended to more domains such as science and code remains unclear.
- The issue that the model becomes more confident on both correct and wrong answers as mentioned in Line 198-203 does not seem to be addressed by CoDaPO comparing Figure 2 and 4.
- Although the paper provides a series of ablation studies, how the components in CoDaPO other than confidence and difficulty, namely “Sample-level normalization” and “Discarding KL regularization,” affects the effectiveness of CoDaPO remains unexplored.
- In Table 2, it seems that adding confidence reweighting and difficulty reweighting individually already evidently improves the performance over standard GPRO, while combining them together only brings marginal additional gain. More empirical results or justification might be required to explain this result.

### Questions
By definition of difficulty in this paper, how effective it is in optimization can be dependent on the training dataset. How do the authors think about the potential impact of factors such as the overall difficulty in the training dataset and the diversity of problems in terms of difficulty?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies two main pitfalls of GRPO: (1) overconfidence in GRPO training, where the model's confidence increases as training goes on, and (2) difficulty, where in hard examples, the rare successful trajectories may receive low gradient contribution. This paper proposes to modify the advantage function of GRPO and add two reweighting factors, one based on confidence, and one based on question hardness. Experimental results show improved performance over GRPO.

### Strengths
* The problem of entropy collapse and overconfidence in GRPO is important, as it could prevent exploration.
* Table 5 and algorithm 1 clearly convey the difference of the method compared to other GRPO variants.
* The proposed method shows results on par with Dr. GRPO and DAPO (and sometimes slightly outperforms them).

### Weaknesses
* The main issue is lack of evidence for the proposed methodologies. For the overconfidence, section 3.2 is generally poorly written. The section title is “Why probability increases and entropy collapses”, but it does not really provide a reason except for some informal vocabulary. The paper cites Figure 2 for higher confidence of the model, however, Figure 4 (the CoDaPO, that is supposed to fix this issue) seems to be very similar to Figure 2 in terms of entropy collapse. Furthermore, it seems that the paper claims the KL divergence has some contribution to this matter, but there are not experiments supporting this claim.
* The proposed algorithm proposes to upweight both hard and easy samples. The intention behind easy sample upweighting being the model should be confident for easy samples, but again, according to Figure 4, this is not case, and confidence (easy or hard) seems to be in a similar range. In terms of hard sample upweighting, prior works (e.g., see "priority sampling" in Kimi 1.5) have already proposed upweighting difficult samples (by a factor of 1-\hat{r}), but the paper does not cite or compare with those methods. 
* The writing seems a bit rushed, dataset/model details in the middle of gradient analysis and definitions, as well as incoherent flow (i.e., too many discrete jumps on various details of GRPO).
* The paper puts considerable effort in explaining KL divergence, but later it sets KL to zero citing prior works for this reason. The analysis in Eq (6) is also too informal to be considered as a part of the paper’s contributions, so I suggest the authors remove the KL definition, and explanation from the very beginning to focus on the main message.
* The paper removes clipping mechanism, however, it does not report or ablate how many off-policy steps the algorithm can take before it starts to break. Also, clipping usually becomes more important at larger scales, so it is unclear if the method will break in larger scales or not. 
* The ablations of Table 2 seems to suggest addition of either confidence reweighting and difficulty reweighting are showing a similar performance (0.3% on math500 is very negligible). So, the main source of gains is not very clear.

### Questions
* The definition of $c_i$ is not clear to me (in Eq 12), why this choice, there is some explanation under Eq 12 but it does not convey the reasoning clearly? why not a common entropy loss on the samples? The notation overloading of $i$ in the numerator of the equation makes it harder to understand what the coefficient is doing.
* I do not understand how Eq (4) is derived. From a quick skim, I believe $\rho$ should not be present in the gradient formulation of GRPO. Please correct me if I am missing something. 
* Why is the model confidence between 0 and 1 (in Figures 2 and 4)? According to its definition, it is mean log probability. Is it somehow normalized?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work presents an analysis of issues that arise during GRPO training on Qwen2.5-Math-1.5B via tracking confidence, rewards, advantages, and gradients. They identify phenomena like entropy collapse, advantage contraction, and uneven learning according to problem difficulty. Based on these issues, they propose CoDaPO, which adaptively rescales reinforcement learning updates by measures of confidence and difficulty. They test their method on seven math reasoning benchmarks using Qwen2.5-Math-1.5B/7B models, advocating for bounded, information-aware reweighting to improve RL post-training for LLM reasoning.

### Strengths
- The systematic study in section 3 is done well and easy to follow, which motivates the proposed algorithm. This type of investigation is important for developing a principled understanding of current RL fine-tuning algorithms to motivate new ones.
- The author’s proposed method is accompanied with comprehensive ablations across various hyperparameters, evaluation datasets and metrics.

### Weaknesses
- The main weakness of this work is that the proposed CoDaPO algorithm doesn’t yield significant gains over existing improvements to GRPO (DAPO, Dr. GRPO) which are comparatively much simpler. CoDaPO introduces two new parameters (the confidence and difficulty weights) which require explicit definitions and tuning; in particular it is unclear why the quadratic difficulty term was chosen. It doesn’t seem that the added complexity pays off in terms of relative performance gain.
- There’s a lack of error bars/confidence intervals, and given the 1-2% difference gaps between CoDAPO and the other methods, I suspect that running across multiple seeds will result in all methods performing within standard error of each other.
- Furthermore, it’s unclear to me that CoDAPO addresses the issues that Section 3 initially highlights. From Figure 4, it seems that model confidence still concentrates around 1 and entropy is still collapsing. Generally it seems that to mitigate the shortcomings identified by this work and prior studies (entropy collapse, uneven progress in problems, etc.) promising directions lie instead in approaches like diversity/exploration bonuses or incorporating curricula instead of the measures proposed by the authors, where simpler alternatives seem to yield very similar results.

In light of the above and given that the phenomena identified in Section 3 of this work has been noted already in prior work, my score currently leans toward reject. However, I still think the analysis is done well and I invite the authors to clarify their contributions and answer my questions below.

### Questions
- This may be a silly question, but for the scatter plots in Figure 2 it seems that the aggregate number of points across difficulty levels and incorrect/correct is less for the model at 240 steps than the base model (are there points being cut off by the advantage range)? Are all plots (from base model, steps 60 and 240) calculated across the full validation set?
- How would the authors’ results in Section 3 change if clipping was removed or purely on-policy updates were used (so ρ = 1?) In that case, the asymmetric truncation driving one-way probability drift would disappear—so the model might experience less entropy collapse and more symmetric gradient updates, though potentially at the cost of higher variance or instability from unbounded updates.
- What are the potential repercussions of the proposed method if the model happens to stumble upon a correct but low-confidence (i.e., unlikely) response? Could this dampen exploration or cause the model to under-reinforce genuinely informative but rare discoveries?
- Are there Dr. GRPO and DAPO baseline results for post-training on the 7B model (Table 4)?
- Figures 16 and 17 in the Appendix show pass@k performance for the model finetuned on CoDaPO, what are the trends for the other baselines (GRPO, Dr. GRPO, DAPO in particular)?
- Would it be possible to try CoDAPO on another family of base model, not Qwen (eg. OctoThinker from Llama3.1-8B [1])?

[1] Wang, Zengzhi, et al. "Octothinker: Mid-training incentivizes reinforcement learning scaling." arXiv preprint arXiv:2506.20512 (2025).

### Soundness
2

### Presentation
3

### Contribution
2
