# Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Direct Preference Optimization (DPO) has emerged as a cornerstone of reinforcement learning from human feedback (RLHF) due to its simplicity and efficiency. However, existing DPO-based methods typically treat all preference pairs equally, overlooking substantial variations in data quality and learning difficulty, which leads to inefficient data utilization and suboptimal performance. To address this limitation, we propose **Uni-DPO**, a unified dynamic preference optimization framework that jointly considers (a) the inherent quality of preference pairs and (b) the model's evolving performance during training. By adaptively reweighting samples based on both factors, Uni-DPO enables more effective use of preference data and achieves superior performance. Extensive experiments across models and benchmarks demonstrate the effectiveness and generalization of Uni-DPO. On textual tasks, Gemma-2-9B-IT fine-tuned with Uni-DPO surpasses the leading LLM, Claude 3 Opus, by 6.7 points on Arena-Hard. On mathematical and multimodal tasks, Uni-DPO consistently outperforms baseline methods across all benchmarks, providing strong empirical evidence of its effectiveness and robustness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper develops what’s called a “unified dynamic preference approach”, UNI-DPO, aiming to improve extend the basic DPO. Some key elements of UNI-DPO include: two weighting factors, w_qual and w_perf, which apply, respectively, to the preference pair (y_win, y_loss) and to the current policy \pi_\theta; the former takes into account data quality, and the latter measures how the policy aligns with the pair. Furthermore, a “calibrated negative log-likelihood loss” (c-NLL) term is added to the training objective, to amplify the policy’s confidence on y_win. These are elaborated in \S3.2, 3.3. and 3.4.

### Strengths
The main strength of the approach is to dynamically adjust each sample’s contribution based on both data quality and the model’s learning performance. This effectively incorporates ideas like advantage-based schemes in RL and thereby improves training efficiency and overall performance.

### Weaknesses
The weight w_qual defined in (5) is a bit simplistic in view of the complexity in defining (or capturing) what is data quality. This is also partially acknowledged by the paper (at the end).

### Questions
Can the author(s) comment on how does the dual weighting scheme in UNI-DPO compare with other approaches to preference alignment and reward optimization, and specifically the weighting factors motivated by importance sampling?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Uni-DPO, which injects two jointly learned adaptive weights into single-pass offline training: 1) a quality weight derived from external score margins that up-weights high-confidence preference pairs, and 2) a performance weight computed from the current policy margin that down-weights already well-fitted samples while emphasizing those that remain hard. Additionally, to counteract DPO’s tendency to suppress the absolute likelihood of preferred responses, the authors devise a calibrated negative log-likelihood term that selectively reinforces difficult yet high-quality positive examples. The resulting objective preserves the simplicity of DPO but enables gradient allocation at a fine-grained level that accounts for both sample difficulty and data quality.

### Strengths
- This paper derives the gradient coefficient of Uni-DPO in closed form, explicitly integrating two modulation factors, including quality weight and performance weight, to provide a principled explanation for online sample re-weighting.
- Uni-DPO preserves the simplicity of single-stage offline training without introducing additional reward models or iterative sampling overhead. With only two learnable weights and a calibrated loss, it can achieve consistent performance improvements across diverse benchmarks.

### Weaknesses
- Although the paper proposes a "unified dynamic weighting paradigm", its core idea essentially combines a quality-aware weight with a performance-aware weight. Such sample reweighting concepts have already been extensively studied in machine learning and RLHF literature, including focal loss, curriculum learning, and advantage reweighting. Therefore, the conceptual novelty of Uni-DPO appears somewhat incremental rather than fundamentally new.

- A major concern lies in the lack of essential baselines. While the paper focuses on DPO-based alignment, it would be important to include comparisons with more recent RL-based alignment methods beyond PPO, since several advanced RL algorithms have recently demonstrated improved stability and performance in preference optimization. Moreover, to my knowledge, there already exist studies on data selection and adaptive weighting for DPO that explicitly aim to exploit heterogeneous data quality during training. Including these approaches as baselines would provide a fairer and more convincing empirical validation.

- Although Uni-DPO claims to "outperform SimPO", its improvement may come from the weighting strategy rather than a new optimization formulation. In contrast, SimPO already removes the reference model and introduces length normalization, while Uni-DPO still relies on the log-probability ratio between the policy and the reference model. This raises questions about whether the claimed superiority originates from a fundamentally different principle or just an additional heuristic weighting scheme.

- Uni-DPO is only evaluated on textual understanding and mathematical reasoning tasks, while multimodal experiments are mentioned in the abstract but not presented in the results section. The lack of evaluations on other important aspects such as dialogue safety, factual consistency, or instruction robustness weakens the "Unified" claim. Additionally, the models used for evaluation (e.g., LLaMA-3-8B, Gemma-2-9B) appear to be relatively early versions; including results on more recent model releases would strengthen the credibility and contemporaneity of the paper.

### Questions
Some suggestions here.

- The right-hand side of Figure 3 does not clearly or intuitively convey the core idea of the proposed method. Its meaning only becomes apparent after reading the entire paper and integrating multiple sections. It is recommended to redesign or annotate the figure to make the conceptual distinction between DPO and Uni-DPO more self-explanatory and visually interpretable.

- It would be helpful to include training curve comparisons (e.g., loss or evaluation score versus training steps) to further illustrate the differences in learning efficiency among Uni-DPO, DPO, and SimPO.

### Soundness
2

### Presentation
3

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
The paper proposes Uni-DPO for fine-tuning large language models with human feedback that dynamically reweights preference training pairs based on each pair’s quality and the model’s evolving performance. This addresses the limitation of standard DPO, which treats all feedback equally. By explicitly emphasizing high-quality, challenging preference pairs during training, Uni-DPO enables more effective use of data, which in turn yields better alignment performance across diverse benchmarks.

### Strengths
1.	The paper clearly identifies that standard DPO treats all preference pairs uniformly, which underutilizes high-quality feedback and fails to adapt to varying task difficulty. Uni-DPO tackles this by adaptive sample weighting, allowing the model to focus on informative training examples and thus improving learning efficiency. This insight targets an important problem in RLHF.
2.	The quality aware weight prioritizes pairs with larger expert score margins, while the performance aware weight, inspired by focal loss, shifts focus toward underfitted examples. This addresses observed mismatches between external score margins and the model’s reward margin.
3.	Uni-DPO shows consistent improvements on diverse tasks. Experiments show that Uni-DPO consistently outperforms vanilla DPO and SimPO baselines on multiple language understanding benchmarks, mathematical reasoning datasets, and multimodal tasks.

### Weaknesses
1.	A notable concern is that Uni-DPO requires a “quality score” for each preference pair, often obtained via human annotation or a powerful proxy model like GPT-4. This introduces an extra dependency on external evaluators (in effect, a form of reward signal), partially undermining the simplicity of the reward model-free DPO paradigm. If these quality scores are noisy, biased, or unavailable, it’s unclear how well the method would perform. The authors themselves acknowledge that training data quality is critical and better methods for estimating it are needed. In scenarios without reliable prior scores, the applicability of Uni-DPO could be limited or require additional effort (e.g. training a proxy reward model), which diminishes its practicality.

2.	The contribution of Uni-DPO, though useful, can be seen as a combination of existing ideas in preference optimization rather than an entirely novel invention. The method builds on known techniques: using an auxiliary NLL loss to counter DPO’s bias (as in Pal et al., 2024), applying length normalization to remove sequence-length bias, and adapting focal loss from classification to reweight easy vs hard examples. The quality-based data filtering is conceptually akin to curriculum learning or prioritizing high-confidence preferences, which have precedents in the literature, although not in the exact form used here. While the unification of these components into one framework is valuable, the paper does not derive fundamentally new theory beyond this integration, nor introduce a wholly new training paradigm. It primarily refines DPO with well-motivated heuristics. 

3.	By focusing training on “hard” and high-quality examples, there is a risk, which is not deeply explored in the paper, of overfitting or bias if these weights are mis-calibrated. It would be better to introduce a calibrated performance weight to avoid instability, indicating the approach required careful handling to train reliably. It would be good to know how consistently training converges under Uni-DPO versus standard DPO. For instance, do some runs diverge or get stuck if the thresholds are set poorly?

4.	While c-NLL is proposed to counteract DPO’s preference suppression bias, the paper does not explore its impact on diversity or other alignment metrics. It also does not evaluate fairness or bias trade offs introduced by weighting high quality data more heavily.

### Questions
1. Could Uni-DPO be combined with reward model–based RLHF methods (e.g., PPO) to further improve alignment? How would the dual perspective weighting interact with explicit reward signals?
2. How sensitive is Uni-DPO to the source and accuracy of the quality scores $w_{qual}$? For example, if one uses a weaker judge model or noisy human preference ratings to derive the score margin, does Uni-DPO still outperform vanilla DPO by a large margin?

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
The paper proposes Uni-DPO, a “dual-perspective” extension of Direct Preference Optimization that dynamically reweights each preference pair by (i) a quality-aware weight $w_{\text{qual}}$ estimated from external expert scores and (ii) a performance-based weight $w_{\text{perf}}$ that emphasizes pairs the model currently fails to fit; it also adds a calibrated NLL term for difficult, high-quality positives. The stated goal is to correct DPO’s uniform treatment of pairs and better utilize data during preference learning for text, math, and multimodal LLMs. Empirically, the authors report consistent gains over SFT, DPO, and SimPO on instruction-following (AlpacaEval2, Arena-Hard, IFEval, SedarEval), math (GSM8K, MATH, MinervaMath, etc.), and multimodal suites, and claim Gemma-2-9B-it+Uni-DPO surpasses Claude 3 Opus on Arena-Hard.

### Strengths
- Clear motivation and formulation: The paper explicitly identifies DPO's uniform-pair limiattions and proposes a dual weighting + c-NLL objective.
- Empirical gains across domains: Reported improvements over SFT/DPO/SimPO on text, meth and multimodal data depicts the strength of Uni-DPO
- Ablations: The paper shows that removing each of the component degrades performance, empirically supoprting the necessity of each component.

### Weaknesses
- Lack of transparency and reproducibility of the scalar quality score $w_{\text{qual}}$.
    - The core of Uni-DPO's contribution, $w_{\text{qual}}$, depends on external scalar scores produced by "expert evaluators" (e.g., GPT-4). However, the exact procedure is not disclosed. Since $w_{\text{qual}}$ is an important component of Uni-DPO, the procedure of obtaining this score should be thoroughly described in the paper.
- Insufficient engagement with existing sample-wise weighting preference optimization (PO) literature.
    - There are no related work section regarding the prior methods that apply per-sample or instance-wise weighting in PO[a,b,c,d]. This weakens the novelty and makes it unclear whether if Uni-DPO has any overlaps between the related works.
- Potential unfairness from applying Length Normalization (LN) only to Uni-DPO.
    - The paper notes that LN is used when computing rewards or scores for Uni-DPO, but not for baseline method such as DPO in Table 1. As LN can boost the performance regardless of the underlying loss, the fairness of the baseline is questionable.
- External scorer dependence and fairness
    - The $w_{\text{qual}}$ depends on strong proprietary LLM scorers, which hinders the fairness against baselines trained with weaker/cheaper labels. How is using these external strong models justified?
- If these weaknesses are taken care of, I am willing to raise my score.

[a] Reward Difference Optimization For Sample Reweighting In Offline RLHF, ACL, 2024

[b] Enhancing RLHF with Weighted Preference Optimization, EMNLP, 2024

[c] MWPO: Enhancing LLMs Performance through
Multi-Weight Preference Strength and Length Optimization, ACL, 2025

[d] Relative Preference Optimization: Enhancing LLM Alignment through Contrasting Responses across Identical and Diverse Prompts, ArXiv, 2025

### Questions
- Described in the Weakness section.

### Soundness
2

### Presentation
4

### Contribution
3
