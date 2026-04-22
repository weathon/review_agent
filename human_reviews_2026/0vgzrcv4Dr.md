# Asymmetric Proximal Policy Optimization: mini-critics boost LLM reasoning

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Reinforcement learning (RL) has become a central paradigm for post-training large language models (LLMs) to elicit stronger reasoning. Yet, most recent RL for LLMs (RL4LLM) methods avoid explicit critics, replacing them with average advantage baselines. This shift is largely pragmatic: conventional value functions are computationally expensive to train at LLM scale and often fail under sparse rewards and long reasoning horizons. We revisit this bottleneck from an architectural perspective and introduce Asymmetric Proximal Policy Optimization (**AsyPPO**), a simple and scalable framework that restores the critic’s role while remaining efficient in large-model settings. **AsyPPO** employs a set of lightweight *mini-critics*, each trained on disjoint prompt shards. This design encourages diversity while preserving calibration, reducing value-estimation bias. Beyond robust estimation, **AsyPPO** leverages inter-critic uncertainty to refine the policy update: (i) masking advantages in states where critics agree and gradients add little learning signal, and (ii) filtering high-divergence states from entropy regularization, suppressing spurious exploration. Across multiple reasoning benchmarks, **AsyPPO** consistently improves learning stability and performance over strong baselines, e.g., GRPO, achieving performance gains of $> 6$% on *Qwen3-4b-Base* and about $3$% on *Qwen3-8b-Base* and *Qwen3-14b-Base* over classic PPO. Such results highlight the importance of architectural innovations in critics for scalable, efficient algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces Asymmetric Proximal Policy Optimization (AsyPPO), a lightweight actor–critic method for reinforcement learning fine-tuning of large language models. It employs several small “mini-critics” trained on non-overlapping data to provide ensemble value estimates. The agreement and disagreement between critics are used to stabilize policy updates through masking and filtering. Experiments on multiple reasoning benchmarks show 3–6% accuracy gains over PPO and GRPO with lower memory usage and faster training.

### Strengths
1.This paper is well motivated. 
The paper revisits a central assumption of modern RL4LLM, namely the removal of explicit critics (as in GRPO), and proposes an elegant asymmetric redesign that restores the critic’s role while remaining scalable.

2.This paper is easy to understand. 
The proposed mini-critic ensemble, prompt-level non-overlapping partition, and uncertainty-based masking/filtering are clearly described and intuitively justified.

3. Strong exps results. 
The method consistently improves performance across several benchmarks, while reducing memory footprint by ~20% and maintaining comparable speed to GRPO.

### Weaknesses
1. Limited interpretability of critic signals
The mini-critics are trained without access to final-answer correctness; hence their value estimates do not guarantee semantic or symbolic validity, especially on complex mathematical reasoning tasks. The reported gains may stem from smoother optimization rather than genuinely better reasoning reward estimation.

2. Lack of compatibility study with GRPO framework
Following point 1, as GRPO could provide "correct" outcome reward, it is essential to verify whether AsyPPO can be plugged into GRPO’s group-based rollout and reward pipeline as a complementary.

3. Evaluation scope
All experiments are restricted to the Qwen3 model family and math-style datasets. It remains uncertain whether the observed benefits generalize to more diverse reasoning or open-ended tasks and models.

### Questions
Please response to the points listed in the weaknesses

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
3

### Summary
The paper proposes AsyPPO as a lightweight method for incorporating more than one mini-critics in RL. Instead of a full-size critic, AsyPPO uses two lightweight "mini-critics" trained on disjoint, prompt-level shards to encourage diversity while staying calibrated. Their predictions are averaged for value estimation, and the standard deviation across critics is treated as an uncertainty signal: the method masks advantages for states where critics strongly agree (to avoid low-information updates) and excludes high-disagreement states from the entropy term (to avoid spurious exploration). On math-reasoning benchmarks, AsyPPO reports notable average gains over GRPO, classic PPO, and lower peak memory / faster steps than symmetric PPO, while remaining competitive and more lightweight.

### Strengths
1. Re-introducing critics via small, prompt-sharded ensembles seems (to me) a clear, practical twist that fits LLM training constraints and mitigates over-parameterized critics. The paper formalizes the training and aggregation cleanly.
2. The two signals, agreement-based advantage masking and divergence-based entropy filtering, are well-motivated and implemented directly inside PPO, with ablations supporting the choices.
3. The paper shows lower peak memory and faster training steps vs. symmetric PPO, and consistent accuracy gains vs. GRPO/symmetric PPO under off-policy reuse. The "two critics is the sweet spot" result is helpful guidance.

### Weaknesses
1. Strong critic-free baselines like GRPO are covered, but comparisons to recent critic-centric or token-level reward methods are referenced more than reproduced, making it hard to isolate where wins come from in identical settings.
2. Claims of stability under off-policy (UTD>1) are promising, but it's unclear how performance changes with different UTD, lambda, or reward sparsity beyond the selected setup.
3. Experiment-wise, most results are math-reasoning; model family is Qwen3-only. This raises external validity concerns (other domains, model families like Llama, non-verifiable reward, etc.). I strongly suggest validating the claims on at least some coding benchmarks.
4. The method introduce addition hyperparameters, including the choice and number of critics, value aggregation, advantage masking fraction, and entropy filtering fraction. Without cross-domain experiments, it's challenging to assess how sensitive these hyperparameters are to different domains, tasks, or model families in practice (and if they are easy to tune).

### Questions
1. Are masking and entropy filtering applied strictly token-wise or differently?
2. The paper shows gains with bigger critics. Where's the diminishing return point relative to actor size, and how does that trade off with ensemble count? And any insights on why more critics does not yield meaningful return?
3. Can entropy filtering over-prune rare but valuable states (e.g., creative solution paths), reducing long-tail discovery?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Asymmetric Proximal Policy Optimization (AsyPPO), a framework designed to address the high computational cost and instability of training large critic models in Reinforcement Learning for LLMs (RL4LLM). Instead of the conventional symmetric design where the critic is as large as the actor, AsyPPO utilizes an ensemble of lightweight "mini-critics". To ensure these small critics provide robust and diverse guidance, they are trained on disjoint, non-overlapping data partitions. The framework further leverages the uncertainty between these critics to refine the policy objective: it masks advantages in states where critics agree (low uncertainty) to prevent overfitting to low-information samples, and it filters states where critics diverge (high uncertainty) from the entropy regularization term to promote safer exploration. The authors demonstrate through experiments that AsyPPO improves reasoning performance over baselines like GRPO and achieves significant reductions in computational overhead (memory and time) compared to classic symmetric PPO.

### Strengths
- The paper addresses the significant and timely problem of the computational bottleneck in critic-based RL for post-training LLMs
- The proposed method is intuitive and well-motivated.
- The paper is well-written, easy to follow, and provides a thorough analysis of each of its components, supported by targeted ablation studies (e.g., ensemble strategy, advantage masking, and entropy filtering).
- The method is validated across a variety of LLM reasoning benchmarks.
- The literature review is comprehensive and effectively situates the work within the current RL4LLM landscape.
- The authors include a clear discussion of the work's limitations and potential avenues for future research.

### Weaknesses
- The empirical evaluations could be strengthened. Several learning curves in the figures do not appear to have fully stabilized or converged, showing volatility late in training (e.g., in Figure 4(c), Figure 6(b), and parts of Figure 7). Extending the training steps could provide a clearer validation of the method's stability and final performance.

- The performance gains, while consistent, are not always substantial. For instance, the main results in Figure 7 show an average improvement of "about 3 points" over the GRPO baseline, which may be modest.

- The study's baselines could be more comprehensive. While GRPO is a strong baseline, the paper would benefit from comparison against other advanced RL4LLM algorithms mentioned in the related work, such as DAPO or REINFORCE++, to better contextualize its contributions.

### Questions
- The paper frequently uses the term "UTD" (update-to-data ratio), setting it to 4 in key experiments. The appendix also lists `ppo_epochs` as a hyperparameter, setting it to 1 or 4. Could the authors explicitly clarify the definition of UTD and explain the precise difference between it and the `ppo_epochs` hyperparameter in their training setup?

### Soundness
3

### Presentation
4

### Contribution
3
