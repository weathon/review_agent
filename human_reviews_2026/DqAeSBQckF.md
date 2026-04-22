# BLISS: A Lightweight Bilevel Influence Scoring Method for Data Selection in Language Model Pretraining

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Effective data selection is essential for pretraining large language models (LLMs), enhancing efficiency and improving generalization to downstream tasks. However, existing approaches often require leveraging external pretrained models, making it difficult to disentangle the effects of data selection from those of the external pretrained models. In addition, they often overlook the long-term impact of selected data if the model is trained to convergence, primarily due to the prohibitive cost of full-scale LLM pretraining. In this paper, we introduce BLISS (**B**ileve **L** **I**nfluence **S**coring method for data **S**election): a lightweight data selection method that operates entirely \emph{from scratch}, without relying on any external pretrained oracle models, while explicitly accounting for the long-term impact of selected data. BLISS leverages a small proxy model as a surrogate for the LLM and employs a score model to estimate the long-term influence of training samples if the proxy model is trained to convergence. We formulate data selection as a bilevel optimization problem, where the upper-level objective optimizes the score model to assign importance weights to training samples, ensuring that minimizing the lower-level objective (i.e., training the proxy model over the weighted training loss until convergence) leads to best validation performance. Once optimized, the trained score model predicts influence scores for the dataset, enabling efficient selection of high-quality samples for LLM pretraining. 
We validate BLISS by pretraining 410M/1B/2.8B Pythia and LLaMA-0.5B models on selected subsets of the C4 dataset. Notably, under the 1B model setting, BLISS achieves $1.7\times$ speedup in reaching the same performance as the state-of-the-art method, demonstrating superior performance across multiple downstream tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes BLISS, a data selection method for llm pre-training. Specifically, authors formulate data selection as a bilevel optimization problem, targeting at maximizing llm performance on validation sets. Experiments on C4 dataset and a series of 410M/1B/2.8B llms validate the effectiveness of BLISS.

### Strengths
- The research topic of this paper is timely and important.
- This paper is relatively well-written.

### Weaknesses
1. The paper heavily emphasizes operating "from scratch" without external pretrained models, yet still requires a warm-up phase using randomly selected data (Section 4.3). Additionally, the bilevel optimization lower-level objective includes KL divergence with a specific LLM, which must already be partially trained to provide meaningful output logits. This fundamentally contradicts the core claim of independence from pretrained models.

2. The paper claims to capture "long-term impact" by training the proxy model "to convergence" (Equation 1). However, the proxy model is trained for only 3,000 steps per round and is reset to initial parameters at the beginning of each round (line 4, Algorithm 1). This is not convergence in any meaningful sense, and resetting eliminates any accumulated long-term knowledge. Authors should provide empirical evidence (e.g., loss curves, model performance) demonstrating that 3,000 steps represents convergence, or revise claims about capturing "long-term" vs. "medium-term" impact.

3. Although the paper presents a bilevel optimization formulation but provides no convergence guarantees, no sample complexity analysis, and no theoretical justification for why this formulation should select better data than alternatives. Without theory, it's unclear whether observed improvements are due to the principled framework or implementation details. At least, authors should add theoretical analysis showing: (a) convergence rates of the bilevel optimization algorithm, (b) bounds on the approximation error of using proxy models, (c) conditions under which the selected data provably improves downstream performance.

4. Table 4 shows BLISS uses 8.08×10^19 FLOPs vs. MATES' 8.11×10^19 for 410M models, claiming efficiency. However, this comparison is not reasonable because

- MATES trains per-sample gradients (inherently expensive) while BLISS batches operations differently.
- The comparison doesn't account for memory overhead of maintaining two additional models.
- Wall-clock time comparisons are absent.
- Different parallelization strategies could dramatically affect practical efficiency.

5. Many reported improvements fall within standard error margins. For example, in Table 1 (410M setting):

- SciQ: 68.1±1.5 (BLISS) vs. 65.7±1.5 (MATES), this is potentially overlapping
- ARC-C: 25.1±1.3 (BLISS) vs. 25.0±1.3 (MATES), this is essentially identical
- Average: 45.9±1.4 vs. 45.7±1.4, this is a marginal difference

6. The paper uses Pythia models throughout (proxy, score, and target LLM) but only briefly tests LLaMA (Appendix B) with different training procedures (periodic resets, KL divergence removal). No analysis examines what happens when proxy and target architectures differ (e.g., Pythia proxy for LLaMA target, or vice versa).

7.  More recent data selection methods aren't compared. 
- These are highly relevant methods but are not compared:   DataComp-LM[1],  QuRating[2], Meta-rater[3], QUAD[4].

- Also, it is understandable that works like  [2-3] use external LLMs which costs different FLOPs than training from scratch, making direct cost comparison problematic. However, this paper criticizes methods using external models but doesn't fairly account for when this might be practically cheaper than bilevel optimization.


[1] https://arxiv.org/abs/2406.11794

[2] https://arxiv.org/abs/2402.09739

[3] https://arxiv.org/abs/2504.14194

[4] https://arxiv.org/abs/2409.16986

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
2

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
This paper introduces BLISS (Bilevel Influence Scoring Method for Data Selection), a lightweight data selection approach for language model pretraining, aiming to address the limitations of existing methods.

Existing data selection methods often rely on external pretrained models, making it hard to isolate the effects of data selection from those of external models, while also neglecting the long-term impact of selected data when the model is trained to convergence. BLISS tackles these issues through a novel bilevel optimization framework: it uses a small proxy model as a surrogate for the large language model (LLM) and a score model to estimate the long-term influence of training samples. The upper-level objective optimizes the score model to assign importance weights to samples, ensuring that the lower-level objective (training the proxy model to convergence on weighted training loss) achieves optimal validation performance. After optimization, the score model predicts influence scores for the dataset, enabling efficient selection of high-quality samples.

### Strengths
1. The existing data selection approach (especially Qurating) depend on large external models (e.g., GPT-3.5), which introduces cost, bias, and reproducibility issues. BLISS avoids this entirely.

2. Rather than estimating sample influence based on short-term updates, BLISS uses a proxy model trained to convergence, offering more reliable evaluation of data utility. 

3. The use of lightweight proxy and score models significantly reduces computational cost. The total training FLOPs are lower than baseline methods like MATES. Also, across a range of benchmarks (e.g., SciQ, ARC-E, LogiQA), BLISS consistently outperforms baselines, showing generalization benefits from better data selection.

4. Please explain the motivation for performing data selection to retain a small amount of high-quality data during the pretraining phase, rather than preserving a large amount of clean data (e.g. preserving 70%-80% of the full candidate dataset). For details, please refer to the weaknesses section. Also, please show the training performance when using 10%, 20%, ..., up to 70% of the entire candidate data pool selected by BLISS, as well as when using 70% of clean data obtained through a simple filtering method. This would be very interesting.

### Weaknesses
1.  I highly commend the authors for providing standard errors but based on Tables 1 and 2, it appears that the improvements on average task accuracy (over MATES) are well within these reported errors? The only result that seems to not be is round 3 for 2.8B scale. Also, for the topic of "pre-training", the experiment scale in this paper is not sufficient. How well the BLISS method generalize to larger model is questionable. When the model becomes large enough, do we still need to select a subset of good examples instead of focusing on getting more high quality data? Also, I think the goal of 'pre-training' is to obtain a strong foundation (unaligned) model that can serve well for later-stage post-training or alignment. However, BLISS introduces a manually specified validation set, which may affect the model's generalization to other tasks. The validation set is more appropriately introduced during the SFT (supervised fine-tuning) or continued pretraining stages, rather than during the pretraining stage.

2. Related to the problem setting, in real-world, when will people do 3B-scale (even 7B-scale) model pre-training from scratch? Usually people either start from a pre-trained checkpoint then do post-training/alignment (for a few specific tasks), or do continuous pre-training/tailpatch training (for knowledge enhancement). If we really need a small scale pre-train ckpt, most likely we will distill a large pre-train ckpt to a smaller model. So how useful this problem setting should be further discussed.

3. Another limitation of this study is that the authors do not fully motivate the problem setting. While cleaning noisy data and removing near-duplicate examples are useful for LLM pretraining, I am not fully convinced why you have to select a subset of “quality data” from the full dataset. For example, if we do a coarse-grained filtering of the original dataset to leave around 70%-80% data and use all of them to do LLM pretraining, will the model performance better.

### Questions
1. Can you try combining the validation tasks (e.g. take an average of their performances) for the purposes of targeting?

2. Looking at the MATES paper, it seems you report the same exact numbers for their method. Are you using the exact same training implementation as they did? (Mentioned in NIPS)

3. An assumption for the BLISS approach is that you want to model influence based on training to convergence. However, in practice, most LLMs are pre-trained based upon a finite compute budget rather than true “convergence” (due to the sheer quantities of data involved). Perhaps the authors could comment on this potential mismatch?

### Soundness
3

### Presentation
2

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
This paper introduces BLISS (Bilevel Influence Scoring for Data Selection), a lightweight approach for data selection consisting of two key components: a proxy model and a score model. The proxy model approximates the behavior of a large language model (LLM), while the score model assigns an influence score to each data sample. Experimental results indicate that the proposed method achieves superior performance across multiple downstream tasks.

### Strengths
1. The paper is well-written, clearly organized, and easy to follow.
2. The proposed bilevel influence scoring framework is conceptually novel and interesting.
3. The method is straightforward and intuitive—evaluating each data sample individually before selection.
4. Experimental results convincingly demonstrate the method’s effectiveness across several tasks.

### Weaknesses
1. The paper lacks a comparison with the state-of-the-art method [1], which also estimates the influence of individual data samples on model performance. Unlike BLISS, that approach computes influence scores directly on the original LLM using the method from [2], without relying on a proxy model.
2. The differences between BLISS and [1] are not sufficiently discussed, leaving unclear what advantages the bilevel formulation provides.
3. The motivation for training a proxy model instead of leveraging the original LLM is not well justified.
4. Experiments are conducted only on the C4 dataset, raising concerns about generalizability to other pretraining corpora.
5. The paper lacks an ablation study examining the effect of the number of training rounds or other key hyperparameters.

## Minor Issues
1. Lines 161–163: The functions F(-) and G(-) are mentioned but not introduced.



[1] Pan, Yanzhou, et al. "ALinFiK: Learning to Approximate Linearized Future Influence Kernel for Scalable Third-Parity LLM Data Valuation." Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 2025.

[2] Lin, Huawei, et al. "Token-wise Influential Training Data Retrieval for Large Language Models." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

### Questions
N/A

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
The paper proposes BLISS, a novel bilevel optimization framework for data selection in large language model (LLM) pretraining. Unlike prior works that rely on external pretrained models (e.g., GPT-3.5) to assess data quality, BLISS operates entirely from scratch, using two lightweight models—a proxy model and a score model—to estimate the long-term influence of data samples. The proxy model mimics the target LLM’s behavior via KL divergence, while the score model learns to assign sample importance weights that optimize validation performance when the proxy model is trained to convergence. Experiments on Pythia (410M, 1B, 2.8B) and LLaMA-0.5B models show that BLISS outperforms baselines such as MATES, DSIR, and SemDeDup, achieving up to 1.7× faster convergence and 1.4% average accuracy gains on multiple downstream tasks, with reduced computational cost.

### Strengths
1. The use of a bilevel influence scoring mechanism for LLM data selection is innovative and theoretically grounded. It elegantly combines influence estimation with bilevel optimization.
2. BLISS avoids reliance on pretrained LLMs for scoring, addressing a key limitation in existing methods like QuRating and MATES.
3. Unlike single-step influence approximations, the framework explicitly models how data affects the model trained to convergence.
4. The authors present experiments across multiple model sizes (410M–2.8B) and architectures (Pythia, LLaMA), with consistent gains across nine downstream benchmarks.
5. Despite additional components (proxy and score models), BLISS achieves comparable or better performance with lower FLOPs.

### Weaknesses
1. Although bilevel optimization is central, the paper lacks a deeper convergence or approximation analysis for the surrogate models’ fidelity to full-scale LLMs.
2. While KL divergence is used for alignment, it remains unclear how well the proxy truly represents the LLM across domains. Quantitative metrics of proxy fidelity would strengthen the claims.
3. Training additional models (even small ones) adds complexity. The paper should analyze memory and runtime scalability for billion-scale pretraining setups.
4. The choice of validation data (e.g., LAMBADA) could bias data selection toward specific linguistic patterns. An analysis of robustness to different validation datasets would be useful.
5. The learned scores are treated as black boxes; understanding what kinds of data are preferred (e.g., factual, reasoning-heavy, stylistic) would improve interpretability.
6. Some ablation results (e.g., single-level vs bilevel) show marginal gains; the statistical significance of these improvements should be better substantiated.

### Questions
1. Add a theoretical or empirical analysis of proxy model representativeness.
2. Report runtime and memory scaling for the bilevel loop.
3. Discuss potential extensions to multimodal or instruction-tuning datasets, as mentioned in the conclusion.
4. Improve clarity in algorithmic descriptions—especially the stochastic hypergradient update (Eq. 5) which could benefit from pseudocode-level explanations.

### Soundness
3

### Presentation
3

### Contribution
2
