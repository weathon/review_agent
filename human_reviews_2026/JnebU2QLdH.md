# Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 8, 2, 4

## Abstract
We investigate the role of learning rate scheduling in the large-scale pre-training of large language models, focusing on its influence on downstream performance after supervised fine-tuning (SFT).
Decay-based learning rate schedulers are widely used to minimize pre-training loss.
However, despite their widespread use, how these schedulers affect performance after SFT remains underexplored.
In this paper, we examine Warmup-Stable-Only (WSO), which maintains a constant learning rate after warmup without any decay.
Through experiments with 1B and 8B parameter models, we show that WSO consistently outperforms decay-based schedulers in terms of performance after SFT, even though decay-based schedulers may exhibit better performance after pre-training.
The result also holds across different regimes with mid-training and over-training.
Loss landscape analysis further reveals that decay-based schedulers lead models into sharper minima, whereas WSO preserves flatter minima that support adaptability.
These findings indicate that applying LR decay to improve pre-training metrics may compromise downstream adaptability.
Our work also provides practical guidance for training and model release strategies, highlighting that pre-training models with WSO enhances their adaptability for downstream tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates learning rate scheduling in large language model pre-training, specifically examining how different schedulers affect downstream performance after supervised fine-tuning (SFT). The authors propose Warmup-Stable-Only (WSO), which maintains a constant learning rate after warmup without decay, and demonstrate through experiments on 1B and 8B parameter models that WSO consistently outperforms decay-based schedulers (WSD, Cosine, Linear) on post-SFT tasks, despite achieving worse pre-training metrics. The authors attribute this to WSO preserving flatter loss landscape minima that support better adaptability.

### Strengths
1. The authors rigorously demonstrate that WSO outperforms traditional decays across multiple scales (1B and 8B), stages (pre-, mid-, and post-training), and regimes (standard, mid-training, over-training). The consistency of results is convincing — the performance inversion between pre-training and SFT is robust.
2. Loss landscape analysis connects WSO’s success to flatter minima — a strong explanatory narrative consistent with sharpness-aware generalization theory (Foret et al., 2021; Wen et al., 2025). Figure 3’s curvature dynamics clearly support this interpretation.
3. The exposition is systematic, well-cited, and transparent. Appendices include hyperparameters, datasets, and evaluation details.

### Weaknesses
1. The explanation of “flatter minima = better adaptability” is qualitative. The paper would benefit from formalizing how curvature interacts with SFT gradient flow (e.g., via a transferability Jacobian or Hessian spectrum analysis across tasks). Without this, the claim remains an empirical observation.
2. SFT evaluation focuses on AlpacaEval, TruthfulQA, and MMLU. These are instruction-following benchmarks but do not fully probe reasoning or alignment generalization. 
3. WSO maintains a higher effective learning rate longer — potentially increasing training instability or wasted compute in late phases. The authors should quantify total compute efficiency (e.g., perplexity vs wall-clock time) to assess tradeoffs.

### Questions
1. You attribute WSO’s superior SFT performance to flatter minima (lower sharpness). Could you quantify how much this flatness contributes to downstream adaptability? For example, is there a measurable correlation coefficient between sharpness values and SFT task scores?
2. Have you examined whether the flatter minima correspond to wider basins of equivalent loss or simply slower convergence zones? This distinction matters for transfer dynamics.
3. How sensitive are your conclusions to the warmup length? Since WSO keeps the LR constant after warmup, a longer warmup could emulate partial decay.
4. The study is based on Llama-like architectures. Do you expect the same effect for mixture-of-experts (MoE) or sparse transformer setups where parameter utilization patterns differ?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates the impact of learning rate (LR) scheduling during LLM pre-training on downstream supervised fine-tuning (SFT) performance. The authors challenge the standard practice of using decay-based schedulers (like Cosine or WSD) which are optimized for pre-training loss. The paper introduces "Warmup-Stable-Only" (WSO), a simple scheduler that maintains a constant LR after warmup without any decay. Through comprehensive experiments on 1B and 8B models, the authors demonstrate a consistent "inversion": while decay-based schedulers achieve better pre-training metrics, models trained with WSO consistently achieve superior performance after SFT. This finding is shown to be robust across standard pre-training, mid-training, and over-training regimes. The paper provides a mechanistic explanation, analyzing the loss landscape and showing that WSO guides models to flatter minima, which enhances adaptability, whereas decay-based schedulers converge to sharper minima that may compromise downstream performance.

### Strengths
The paper is exceptionally clear, well-written, and easy to follow.

The central conclusion—that pre-training without LR decay enhances SFT performance—is simple, impactful, and supported by extensive evidence. The experiments are comprehensive, covering multiple model scales (1B and 8B), different training pipelines (two-stage and three-stage with mid-training), and modern training regimes (over-training).

This work has significant practical implications for the industry. The WSO scheduler is simple to implement and could provide real economic benefits by producing base models that are more adaptable and performant for downstream tasks.

The mechanistic explanation provided via loss landscape sharpness is insightful. The analysis linking the constant LR of WSO to flatter minima, and in turn, to better adaptability, offers a compelling hypothesis for *why* WSO outperforms decay-based methods in the post-SFT stage.

### Weaknesses
The primary weakness, though minor, is that the investigation of downstream performance is limited to SFT. The paper does not explore other critical post-training stages, such as preference tuning (e.g., DPO) or reinforcement learning-based alignment. It remains an open question whether the significant benefits of WSO pre-training persist or behave differently in these other alignment scenarios. I don't think this would be an issue as the title also constrains the scope to SFT.

### Questions
The paper compellingly argues that WSO leads to flatter minima (lower sharpness) and that WSO models perform better on SFT. The link is made by showing these two facts separately. To make the justification more persuasive, have the authors considered plotting a direct correlation between the measured sharpness of the pre-trained checkpoints and their final SFT benchmark scores? This would provide a more direct piece of evidence that sharpness is indeed the key indicator for downstream adaptability as hypothesized.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigate the role of learning rate scheduling in the large-scale pre-training of large language models, focusing on its influence on downstream performance after supervised fine-tuning (SFT). Specifically, this paper proposes Warmup-Stable-Only (WSO) learning rate schedule for pertaining, which is found to achieve better downstream tasks. Some experiments and intuitive understandings are presented.

### Strengths
1. This paper presents WSO, a very simple and intuitive LR schedule to improve SFT on downstream performance.
2. This paper provides a practical reference for LLM pre-training community to design LR schedule from a global training perspective.

### Weaknesses
The primary concern with this paper is that the proposed approach—while effective—has been extensively discussed, implemented, and validated in prior work, without introducing significant novelty. Furthermore, the absence of references to these existing studies raises questions about the thoroughness of the literature review.

1. In the original WSD paper [(https://arxiv.org/pdf/2404.06395)](https://arxiv.org/pdf/2404.06395), the authors already demonstrated the benefits of switching to high-quality datasets (including SFT data) during the learning rate decay phase, yielding intuitive and positive outcomes.

2. The paper "Scaling Law with Learning Rate Annealing" [(https://arxiv.org/pdf/2408.11029)](https://arxiv.org/pdf/2408.11029) introduces a scaling law describing loss dynamics in relation to learning rates, of which the current work appears to be a specific instance.

3. The paper "Learning Dynamics in Continual Pre-Training for Large Language Models" [(https://arxiv.org/pdf/2505.07796)](https://arxiv.org/pdf/2505.07796) provides comprehensive analyses, and the findings here seem to represent only a minor subset of their paper. Notably, their Finding 3 states: "*PT models with higher loss potential consistently achieve lower D_cpt validation losses. Hence, we advocate that when releasing open-source models, it is beneficial to release a high loss potential version to facilitate downstream tasks.*"

   I strongly encourage authors to read this paper.

4. The paper "A Learning Rate Path Switching Training Paradigm for Version Updates of Large Language Models" [(https://arxiv.org/pdf/2410.04103v1)](https://arxiv.org/pdf/2410.04103v1) applies a similar concept to LLM pre-training.

In essence, the core idea (**Let LR decay happen in the most important stage**) has already been well-established in the field. This paper just translates this idea into a superficial learning rate schedule.

Additionally, the paper lacks rigorous theoretical analysis, and the evaluation is insufficient. For example, Table 2 reports only loss variations and average SFT performance. To strengthen the claims, the authors should address deeper questions such as:

1. How do the results vary if the pre-training duration is extended or shortened?
2. Are certain SFT tasks more or less affected by the proposed schedule? If so, what underlying reasons might explain this?
3. Given that WSO outperforms WSD, why not slightly increase the learning rate during pre-training to further boost downstream SFT performance, even at the expense of higher pre-training loss?
4. Have the authors considered quantifying this process more formally, such as by deriving a scaling law?

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The manuscript presents extensive experimentation devoted to understanding learning-rate scheduling in LLM training. This work provides empirical insights that directly suggest how learning rate scheduling should be selected during pre-training to better support downstream model adaptability. The study recommends adopting Warmup-Stable-Only (WSO) as an alternative learning-rate strategy and releasing WSO-trained models to encourage wider use and adaptability in future LLM development.

### Strengths
1. The paper examines the common practice of using learning-rate decay in LLM pre-training. The paper provides empirical evidence that keeping a constant learning rate after warmup improves performance. This approach, called the Warmup-Stable-Only (WSO) scheduler, outperforms conventional decay-based schedulers in supervised fine-tuning. Their finding highlights practical effectiveness for optimizing the entire LLM training pipeline.
2. The paper demonstrates the inversion effect between pre-training and supervised fine-tuning performance across a wide range of settings. Decay-based learning rate schedulers consistently achieve stronger pre-training metrics, whereas the WSO configuration achieves superior results after SFT. This phenomenon is validated across 1B and 8B model scales, in multi-stage training pipelines.
3. The paper challenges the standard assumption that stronger pre-training performance leads to a better final model. It shows that decay-based learning rate schedules achieve superior pre-training metrics, yet consistently result in worse performance after supervised fine-tuning. Their evidence suggests a need to rethink optimization goals in LLM development. They also emphasize prioritizing downstream adaptability over pre-training loss.

### Weaknesses
1. The experiments are restricted to 1B and 8B parameters, which are relatively small compared to state-of-the-art deployed LLMs (often 30B~70B+). The absence of results at larger scales limits confidence in whether the observed advantages of WSO would extend to all situations.
2. The study evaluates WSO against only three decay-based schedulers (Cosine, Linear, and Warmup-Stable-Decay). Other commonly used or recently explored learning rate strategies, such as polynomial decay or cyclic policies, are not explored. This limited comparison makes it unclear whether WSO’s benefits extend to other learning-rate policies. The paper would benefit from a brief stability assessment that checks whether WSO remains reliable under different environments.
3. The experiments tune SFT hyperparameters separately for each pre-trained model. However, they always use selective learning-rate policies during SFT. This choice gives WSO an inherent advantage in downstream evaluation. The learning-rate policy should instead be maintained consistently across all training phases to enable a fair comparison. The work also lacks theoretical significance, relying mainly on empirical observations.
4. The paper primarily evaluates instruction-following and general reasoning tasks, without testing multilingual ability, coding, or robustness under distribution shift. This narrow benchmark scope limits confidence in how widely WSO’s performance would translate to real-world deployment scenarios.
5. Important related studies are missing from the references, such as [1].

[1] Jin, Hongpeng, et al. "Rethinking learning rate tuning in the era of large language models." 2023 IEEE 5th International Conference on Cognitive Machine Intelligence (CogMI). IEEE, 2023.

### Questions
Please check the detailed comments for weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
