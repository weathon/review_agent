# Provably Mitigating Corruption, Overoptimization, and Verbosity Simultaneously in Offline and Online RLHF/DPO Alignment

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 5, 6, 5

## Abstract
Reinforcement learning from human feedback (RLHF) and direct preference optimization (DPO) are emerging and important techniques to align large language models (LLM) with human preference. However, the quality of RLHF and DPO training is seriously compromised by ***C**orrupted* preference, reward ***O**veroptimization*, and bias towards ***V**erbosity*. To our knowledge, most existing works tackle only one of these important issues, and the few other works require much computation to estimate multiple reward models and lack theoretical guarantee of generalization ability. In this work, we propose RLHF-**COV** and DPO-**COV** algorithms that can simultaneously mitigate these three issues, in both offline and online settings. This ability is theoretically demonstrated by obtaining length-regularized generalization error rates for our DPO-COV algorithms trained on corrupted data, which match the best-known rates for simpler cases with clean data and without length regularization. Moreover, our DPO-COV algorithm is simple to implement without reward estimation, and is proved to be equivalent to our RLHF-COV algorithm, which directly implies the equivalence between the vanilla RLHF and DPO algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies how to fine-tune LLMs to align it with human preference data, aiming to address the change of corrupted preference, reward Overoptimization, and verbosity. Inspired by previous works on these challenges respectively, this work proposes new algorithms named RLHF-COV and DPO-COV that can simultaneously mitigate these three issues simultaneously. Theoretical results prove that under suitable assumptions,  the DPO-COV algorithm enjoys a converging length-regularized generalization error rates trained on corrupted data. 
Experimental results further demonstrate the effectiveness of the proposed algorithms.

### Strengths
**Clarity and Quality:**
1. The paper is clearly written and the theoretical results are sound.

**Originality and Significance:** 
1. The paper is the first to consider the joint solution that mitigates all of data corruption, reward over-optimization, and verbosity while being theoretically supported.
2. New technical methods to handle corrupted data in the RLHF setup.
3. The experiments show the competitiveness of the proposed algorithm in aligning LLMs to some extent.

### Weaknesses
From the reviewer's perspective, despite that the joint handling of these challenges is new in RLHF theory literature , the main idea of the algorithm design is not novel given existing works, i.e., the value-guided method to handle over-optimization/encourage exploration (see [1, 2, 3, 4, 5]) without the new components of handling corrupted data and verbosity, which makes the theoretical contributions of the paper weakened.  Also for this reason, the earlier work [1, 2] of this line in standard RL needs also to be mentioned in the paper for the completeness of literature review.

Another weakness is that the experiments are limited, which does not quite demonstrate the overall effectiveness of the proposed algorithm. E.g., see Question 3 below.

**References:** 

[1] Xie T, Cheng C A, Jiang N, et al. Bellman-consistent pessimism for offline reinforcement learning[J]. Advances in neural information processing systems, 2021, 34: 6683-6694.

[2] Liu Z, Lu M, Xiong W, et al. Maximize to explore: One objective function fusing estimation, planning, and exploration[J]. Advances in Neural Information Processing Systems, 2023, 36.

[3] Liu Z, Lu M, Zhang S, et al. Provably mitigating overoptimization in rlhf: Your sft loss is implicitly an adversarial regularizer[J]. arXiv preprint arXiv:2405.16436, 2024.

[4] Cen S, Mei J, Goshvadi K, et al. Value-Incentivized Preference Optimization: A Unified Approach to Online and Offline RLHF[J]. arXiv preprint arXiv:2405.19320, 2024.

[5] Xie T, Foster D J, Krishnamurthy A, et al. Exploratory Preference Optimization: Harnessing Implicit Q*-Approximation for Sample-Efficient RLHF[J]. arXiv preprint arXiv:2405.21046, 2024.

### Questions
1. In Theorem 1, the choice of the parameter $\eta$ depends on the quantity of partial coverage coefficient $\mathcal{G}_{\mathcal{D}}$, which somehow seems impractical. How does the learner know the exact number of the partial coverage coefficient? Even though in practice the parameter is typically chosen via empirical search, but this theoretical choice still does not make sense to me.
2. Beyond the experiments in the paper, how to understand does the proposed algorithm indeed handle the challenge of corrupted data, reward over-optimization, and verbosity? In other words, is there additional experiment which supports that the superior performance of the new algorithm is exactly brought by the mitigation of these issues? For example for reward overoptimization, can the authors show that compared with the algorithm that does not consider this explicitly, the "reward" is indeed not over-optimized. Similar questions for corruption and verbosity.
3. What about the capabilities of the LLMs trained by DPO-COV in complex reasoning tasks, e.g., math and coding?
4. Typos: inconsistent notation for the partial coverage coefficient, see Section 3.3.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper studies how to deal with corrupted preferences, overoptimization and verbosity in RLHF and DPO. The authors propose RLHF-COV and DPO-COV algorithms that address these issues simultaneously in both offline and online settings. Their approach combines the existing techniques and they provide theoretical guarantees for their method. Experimental results demonstrate the efficacy of their approach compared to methods which only deal with part of the three issues.

### Strengths
The literature review is detailed and the presentation is clear.

### Weaknesses
The novelty of this paper is very limited and not sufficient enough for ICLR.
1. Algorithmically, the proposed algorithms are a simple combination of existing works. The anti-corruption component comes from (Bukharin et al., 2024), the anti-overoptimization component from (Liu et al., 2024c) and the anti-verbosity component comes from (Park et al., 2024). The online version adds (Bukharin et al., 2024) and (Park et al., 2024) on (Xie et al., 2024). I see no obvious novelty in the algorithm design.
2. Theoretically, the contribution is also relatively small. Most of the proof utilizes the techniques in (Bukharin et al., 2024), (Liu et al., 2024c) and (Xie et al., 2024). The authors claim that the novelty lies in the analysis order of the noise terms, which seem not significant to me.

### Questions
In Theorem 1, it should be $G _ {\mathcal{D}}$ rather than $\mathcal{G} _ {\mathcal{D}}$?

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
This paper combines several techniques in addressing preference corruption, model optimization, and preference to verbosity into one objective, and shows the effectiveness of the objective both theoretically and empirically.

### Strengths
The paper is sound in it's theory part. The derivations of the proposed objective, along with the mathematical proofs, are sound. The objective is novel to the best of my knowledge.

### Weaknesses
The objective itself, although reduced from minimax problem to just one minimization problem, has many hyperparameters in it. Authors mentioned that they used grid search over three out of the four hyperparameters in their experiment section. This is concerning due to the uncertainty and cost behind the hyperparameter search. Moreover, comparing all the methods against one much more powerful model (GPT4 in this case) on just one test set (AlpacaEval 2.0) isn't very convincing as the evidence to show the efficacy of the proposed objective. I would suggest the authors to add head-to-head win rates between pairs of the algorithms listed for comparison on more test sets.

### Questions
How fine should we set the grids and how sensitive are the results to the hyperparameters?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper proposed designs to mitigate Corrupted preference, reward Overoptimization, and bias towards Verbosity observed in RLHF and DPO. The authors use noise model to mitigate corruption, pessimistic and optimistic regularization to mitigate overoptimization, and length regularization to mitigate verbosity. They show a generalization error guarantee for the proposed methods, and experiments on Alpaca Eval 2.0 to support the proposed methods

### Strengths
Tackling the corruption, overoptimization, and verbosity issues of LLM preference alignment at the same time is not achieved before, and this work makes a reasonable contribution.

The paper is well written and easy to understand.

### Weaknesses
The experimental results are not satisfactory.

### Questions
Using Llama-3-8B-Instruct it seems the current win-rate can be 20%+ (see Table 1 of https://arxiv.org/pdf/2405.00675). The authors reported ~7% win-rate of using Llama-3-8B. Could you explain the reason here?

### Soundness
3

### Presentation
3

### Contribution
2
