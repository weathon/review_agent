# Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Supervised fine-tuning (SFT) is the standard approach for post-training large language models (LLMs), yet it often shows limited generalization. We trace this limitation to its default training objective: negative log likelihood (NLL).  While NLL is classically optimal when training from scratch, post-training operates in a different paradigm and could violate its optimality assumptions, where models already encode task-relevant priors and supervision can be long and noisy. To this end, we study a general family of probability-based objectives and characterize their effectiveness under different conditions. Through comprehensive experiments and extensive ablation studies across 8 model backbones,  27 benchmarks, and 7 domains, we uncover a critical dimension that governs objective behavior: the *model-capability continuum*. Near the *model-strong* end, prior-leaning objectives that downweight low-probability tokens (*e.g.,* $-p$, $-p^{10}$, thresholded variants) consistently outperform NLL; toward the *model-weak* end, NLL dominates; in between, no single objective prevails. Our theoretical analysis further elucidates how objectives trade places across the continuum, providing a principled foundation for adapting objectives to model capability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper examines the effects of the negative log likelihood loss on various domains. The findings reveal that there is a relationship between the type of loss used and the model's priors on the task. Based on this, the paper proposes a model capability continuum as a way to formalize the spectrum of models' priors and their relationship to various probability-based learning priors. More specifically,y they find that models with weak priors on task tend to benefit more from NLL loss as compared to models with strong priors. This end of the spectrum benefits more from down-weighting low probability tokens. At the center, they find that no one objective function has a clear advantage.

### Strengths
- The motivation is stated clearly in how SFT is applied to LLM alignment compared to classification training
- Research questions are stated clearly 
- The paper lays out the theoretical background of the loss functions used for SFT and a generalized version of it

Method and Experiment:
- Paper shows extensive experiments on the Model Strong and Model Moderate settings with a number of benchmarks
- Ablation studies: The paper does extensive ablation on the high, low, and mid probability tokens' fine-tuning using various values of alpha

### Weaknesses
Related work Depth:
	
- The paper has not examined existing literature exploring alternatives to CE loss. [1, 2]
	
- It would be great to get a comparison to this work and a more comprehensive literature review of the existing landscape of alternative loss functions

Method and Experiment:
- The experiments done on mode weak are not extensive. The choice of benchmark for model weak is much more restrictive compared to the other 2 settings

1. Entropic Distribution Matching in Supervised Fine‑tuning of LLMs: Less Over‑fitting and Better Diversity
2. Computer Vision Losses for Large Language Model Fine‑Tuning

### Questions
- Can the author discuss more about the connection with RL? This setting is similar to a policy with a strong prior setting in RL.
- I would like to know if the results of model-weak still hold when using some other domain like coding, science, multi-lingual, etc (anyone could work).
- For the model strong setting, it would be intresting to see results on a dataset that emphasizes knowledge memorization where the model has strong priors (Wikipedia, etc.).
- Following on the previous question, does the proposed continuum still stand when we make a distinction of datasets that are reasoning/skills vs pure knowledge memorization? In other words is NLL loss sub-optimal choice for each class of the dataset when there strong prior and vice versa

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper revisits the standard practice of supervised fine-tuning (SFT) for large language models by questioning the negative log-likelihood (NLL) training objective. Authors propose a family of probability-based objectives that generalize NLL (which is the limit as $α→0$). They find alternative objectives (example $-p$ or $-(p^{10})/10$, which downweight low probability tokens) can outperform NLL on certain tasks. The key contribution is identifying a "model capability continuum": when the base model is strong (already has high prior knowledge on the task), prior-leaning objectives (that trust the model's prior) yield better generalization than NLL. When the base model is weak, the NLL objective is better for learning from scratch. In intermediate capability settings, no single objective is consistently better. The authors run comprehensive experiments across 7 models, 14 benchmarks, and 3 domains, demonstrating up to 16% accuracy gains with prior-leaning losses on strong models, whereas NLL remains best on weaker models. A theoretical analysis is provided to explain the performance of objectives. The work provides a practical guidance to choose objectives based on current model capability to improve generalization.

### Strengths
* **Novel Perspective:** The paper offers a new viewpoint by questioning the default use of NLL for fine-tuning large pre-trained models. It introduces the concept of a model capability continuum, which is a clear way to understand how a model's prior knowledge should influence training strategy.

* **Thorough Empirical Validation:** The experimental evaluation is very comprehensive. The authors conduct tests on 7 different LLM backbones (of varying sizes and domains) and 14 benchmarks covering diverse tasks (math problem solving, medical question answering, logic puzzles, etc.). This breadth gives good credibility to the findings, the continuum pattern (prior-leaning losses excel with strong models, NLL excels with weak models) is consistently observed, not just a one off result. Significant performance gains (sometimes doubling accuracy) are achieved in few settings using the new proposed objectives.

* **Theoretical Insight:** Beyond empirical results the paper provides a theoretical analysis that supports its claims. The authors derive conditions under which one objective will outperform another, and show that these conditions flip between the "model strong" and "model weak" ends of the spectrum. This adds a lot of weight to the work it's not just "we tried this new loss and it worked" but why it works is partly explained through a formal lens.

* **Clarity and Context:** The paper is well written and not hard to follow. It motivates the problem clearly (highlighting how long chain-of-thought supervision and strong pretrained priors violate assumptions of NLL's optimality). It also contextualizes the work in the literature: for example it contrasts its approach with reinforcement learning from human feedback (RLHF) and other recent techniques like PPO-inspired fine-tuning, importance sampling in SFT, and selective data training.

* **Significance:** The findings have notable implications for the community. If NLL is not universally optimal for post-training, this could prompt many researchers and practitioners to reconsider their fine-tuning procedures. The idea that one should "lean on the model’s knowledge when it's strong, and override it when it’s weak" is a valuable guideline.

### Weaknesses
1. **Objective Adaptation in the Intermediate Regime.**
   The paper identifies that no single objective consistently works well in the model-intermediate regime, but does not propose a method to handle this case. This is a practical gap, since many real-world tasks likely fall in this zone.

2. **Deciding Model Capability in Practice.**
   The framework relies on knowing whether a model is "model-strong" or "model-weak" on a task, but the paper does not provide a way to assess this ahead of time. The current categorization is done post hoc.

3. **Forgetting on Prior Tasks.**
   The paper focuses on improving performance on new tasks during fine-tuning but does not study how different objectives affect retention of previously learned capabilities. This matters for applications where continual learning is important and accuracy needs to be high on the entire sequence of tasks being fine-tuned on.

### Questions
1. Did the authors explore or consider adaptive objective schedules during training (example starting with NLL and transitioning to a prior-leaning loss)? If not what challenges do you expect to see in implementing such an approach?

2. How should practitioners determine model capability before fine-tuning? Can simple metrics like zero-shot accuracy or mean token confidence be used reliably to choose the right objective?

3. Did the authors measure or observe any differences in forgetting on prior capabilities when using prior-leaning objectives like $-p$ or $-p^{10}$ compared to NLL? Would you expect more or less forgetting in these cases?

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
4

### Summary
This paper argues that the standard negative log-likelihood (NLL) objective used in supervised fine-tuning (SFT) of large language models is not always the optimal choice during post-training. The authors observe that pretrained models already contain strong prior knowledge, and forcing them to imitate every supervision token can lead to overfitting and poor generalization. They introduce and study a broader family of probability-based training objectives that either emphasize or downweight low-probability tokens. Through experiments across multiple model sizes, datasets, and domains, they identify a “model-capability continuum”: in domains where the base model already has strong priors (e.g., math), objectives that downweight low-probability tokens (such as −p or thresholded −log p) outperform NLL. In domains where the model has weak priors (e.g., unseen puzzles), NLL performs better because it forces learning from unlikely tokens. In intermediate domains (e.g., medical reasoning), no objective clearly dominates. The paper further supports these findings with theoretical analysis showing how gradients and learning dynamics differ across capability regimes.

### Strengths
The paper addresses an important and timely question in LLM post-training by re-examining the default SFT objective, which is usually taken for granted. The experimental evaluation is broad, covering multiple model families, diverse datasets, and capability levels, demonstrating the generality of the results. The conceptual introduction of a “model-capability continuum” provides an intuitive and practical framework for understanding when different objectives should be used. The empirical results are supported by gradient-based theoretical reasoning, which makes the findings more convincing. The paper has clear motivation, thorough ablations, and actionable insights for practitioners who want to improve fine-tuning outcomes.

### Weaknesses
The classification of domains into “model-strong,” “model-intermediate,” and “model-weak” can feel somewhat heuristic and may not be straightforward to estimate for new tasks in practice. The proposed approach still requires manual selection of the objective based on the capability regime, and the paper does not yet provide an automated or adaptive method for doing this. While the theoretical explanation is suggestive, it relies on simplified assumptions and does not fully capture the complexity of real training dynamics. In some intermediate settings, the differences between objectives are small, which may limit the practical impact in many real-world SFT use cases. The paper also evaluates improvements mainly on reasoning-heavy tasks, so it is less clear how broadly the results generalize to conversational or stylistic alignment tasks.

### Questions
The paper discusses a continuum from model-weak to model-strong domains, but the operationalization of this continuum is not fully specified. How should a practitioner determine where a new task sits on this continuum before training? Is there a quantitative diagnostic metric that can be computed prior to fine-tuning, rather than one derived from already trained or partially trained models?

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
LLMs are usually post trained using SFT, where the model is taught to reproduce a reference answer token by token using NLL loss. The authors argue that once a model has been pretrained, NLL is no longer universally optimal because the model already encodes strong priors and SFT supervision can be noisy or irrelevant. The paper introduces a general family of objective functions that work under different conditions (MS, MI, MW). They show that through this formulation, they improved performance across 14 benchmarks.

### Strengths
* This paper introduces a general family of probability based objectives, it broadens the space of loss functions and connects NLL and accuracy as special cases.
* The proposed idea of model capability continuum is neat, though the way to measure a models MS, MI and MW could be improved.

### Weaknesses
* Experimental results focus on narrow domains (math, medical and puzzles) It would be good have results on some other general benchmarks (wild bench, arena hard, IF-eval, some code and agentic evals)
* The continuum proposed by the paper relies on the mean predicted probability and pretraining coverage as proxies for prior strength. LLMs are often miscalibrated. Using a single scalar to rank tasks may overlook nuanced factors such as variance, entropy or distributional mismatch.
* The paper does not study whether thresholding harms knowledge retention, fairness, or calibration.
* The authors claim that RL‑inspired methods such as implicit reward learning, importance sampling and PPO‑style clipping are special cases of their prior leaning objectives. This should be backed with some empirical comparisons.

### Questions
1. do you anticipate the same continuum behavior will hold for much larger LLMs (> 30B)? Could larger and more capable models potentially benefit even more from prior leaning objectives, or might new challenges (like optimization instability or diminished gains) arise at that scale?
2. Have you considered using UQ methods as a more principled metric for assessing model capabilities. Such methods might better capture the epistemic vs aleatoric uncertainty and could help automate the classification of MS, MI and MW
3. The experiments use a fixed threshold and show that training on the top 10 % of tokens yields strong improvements. How sensitive are the results to this choice?
4. RL‑based methods such as RLHF, DPO, RPO and one‑token rollout also downweight low reward or low probability tokens by sampling. Could you provide a comparison between your probability based objectives and these RL approaches
5. Does downweighting low‑probability tokens have any adverserial effects, like does it affect calibration or fairness?

### Soundness
2

### Presentation
2

### Contribution
2
