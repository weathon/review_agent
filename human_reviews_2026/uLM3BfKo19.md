# Quagmires in SFT-RL Post-Training: When High SFT Scores Mislead and What to Use Instead

- Avg Score: 5.67
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6, 6, 4

## Abstract
In post-training for reasoning Large Language Models (LLMs), the current state of practice trains LLMs in two independent stages: Supervised Fine-Tuning (SFT) and Reinforcement Learning with Verifiable Rewards (RLVR, shortened as "RL" below). In this work, we challenge whether high SFT scores translate to improved performance after RL. We provide extensive counter-examples where this is not true. We find high SFT scores can be biased toward simpler or more homogeneous data and are not reliably predictive of subsequent RL gains or scaled-up post-training effectiveness. In some cases, RL training on models with improved SFT performance could lead to substantially worse outcome compared to RL on the base model without SFT. We study alternative metrics and identify generalization loss on held-out reasoning examples and Pass@large k performance to provide strong proxies for the RL outcome. We trained hundreds of models up to 12B-parameter with SFT and RLVR via GRPO and ran extensive evaluations on 7 math benchmarks with up to 256 repetitions, spending $>$1M GPU hours. Experiments include models from Llama3, Mistral-Nemo, Qwen3 and multiple state-of-the-art SFT/RL datasets. Compared to directly predicting from pre-RL performance, prediction based on generalization loss and Pass@large k achieves substantial higher precision, improving $R^2$ coefficient and Spearman's rank correlation coefficient by up to 0.5 (2x). This provides strong utility for broad use cases. For example, in most experiments, we find SFT training on unique examples for a one epoch underperforms training on half examples for two epochs, either after SFT or SFT-then-RL; With the same SFT budget, training only on short examples may lead to better SFT performance, though, it often leads to worse outcome after RL compared to training on examples with varying lengths. This work develops an enhanced evaluation tool for math reasoning tasks and is open-sourced.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper challenges the point in LLM post-training that a higher Supervised Fine-Tuning (SFT) score guarantees better final performance after Reinforcement Learning (RL). The authors identify and provide extensive evidence for the *SFT metric trap*, where optimizing for SFT performance can be misleading and suboptimal for the final RL outcome. As a solution, the paper proposes and validates two more reliable predictive metrics (i.e., generalization loss and Pass@large k accuracy), which offer higher predictive power for RL success, thereby providing a more principled and cost-effective methodology for optimizing the SFT-RL pipeline.

### Strengths
- The paper addresses a critical problem in real-world LLM development, making the findings impactful for practitioners.
- The empirical evaluation is rigorous and extensive, conducted on multiple models and benchmarks, which lends credibility to the claims.
- The proposed metrics are not only predictive but also actionable, offering a framework for practitioners to make better decisions at both the dataset and instance levels.

### Weaknesses
- My main concerns lie in the generalizability of the findings. The study is exclusively focused on mathematical reasoning tasks, where rewards are often token-level and verifiable. It is unclear if these predictors would hold for other domains like complex instruction following, or agentic tasks, where rewards are more holistic and harder to define. The strong performance of *generalization loss* and *Pass@large k* might be an artifact of the structured nature of math problems. For tasks that value creativity or conversational nuance over discrete correctness, a moderate degree of SFT overfitting might even be beneficial for stylistic alignment, potentially creating an entirely different dynamic with the subsequent RL stage. The paper lacks a discussion on this crucial limitation.
- The paper establishes correlations but provides limited insight into the underlying causal mechanisms explaining *why* these metrics are better predictors.
- The high computational cost of evaluating Pass@large k is a significant practical drawback that may limit its adoption, especially for larger models.
- The anomalous result of the Qwen3 model is a missed opportunity for a deeper analysis that could have provided more profound insights into model-specific behaviors.

### Questions
- How should practitioners weigh the trade-offs between generalization loss and Pass@large k when they provide conflicting signals for different SFT model candidates?
- Given that RL is a dynamic process, is it possible that an SFT model deemed suboptimal by these static metrics could unlock its potential later in RL, perhaps with different RL hyperparameters like a higher exploration temperature?
- How do the two analyzed factors (SFT data selection for instance-level and training paradigm for dataset-level) interact? For instance, would training on shortest examples for more epochs amplify the negative effects observed?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper demonstrates that the performance of a model after SFT is often not predictive of its performance after RLVR. They consider **dataset-level ablations**, which vary the number of training examples and epochs used in training, and **instance-level ablations**, which ablate the distributional makeup of the data. They show that in both settings, the SFT model's performance does not correlate with the final RL model's performance. For example, training for many epochs on the same data leads to better SFT performance but worse outcomes after RL; the same is true for training on short examples. 

The authors then propose two new metrics for SFT models, namely validation loss (which measures generalization) and pass@k performance (for large k), and demonstrate that it is substantially more indicative of final performance. In the dataset-level ablations, they find that both validation loss and pass@k are strong predictors, achieving $\geq$ 0.90 Spearman correlation (30% improvement over SFT pass@1 performance). For instance-level ablations, pass@k is similarly a much stronger predictor (validation loss is deemed "not applicable").

### Strengths
1. This paper studies an important broader question: what is the role of SFT prior to subsequent RL training? While SFT followed by RL is still the de facto post-training pipeline, there remains substantial confusion about what metrics the SFT team should optimize for in order to produce a good starting point for RL.
2. The paper provides strong empirical evidence (and compelling intuition) for their findings throughout the paper.
3. The paper provides two new metrics for evaluating SFT models, validation loss and pass@k, which are more effective indicators of final model performance than the traditional pass@1 performance of SFT models.

### Weaknesses
1. The findings are not entirely novel or surprising. For instance, [Weight Ensembling Improves Reasoning in Language Models](https://arxiv.org/abs/2504.10478) also show that during SFT, the Pass@1 rate improves while Pass@k deteriorates, and that Pass@k of an SFT model is a much stronger indicator of its performance after RL. Pass@k is also the main metric proposed by the authors of this paper for predicting final model performance. Citation & comparison with this work is definitely required. More broadly, it is well-known that SFT and RL optimize for different (and sometimes conflicting) objectives, and that strong SFT models ≠ strong starting points for RL.
2. Many aspects of the writing and presentation can be cleaned up. See below:

- The main finding from Figure 1 is difficult to parse. Within each box (green/orange/red), it looks like higher green bar (post-SFT performance) *does* lead to higher black bar (post-RL performance). However, I think the point you're trying to highlight is that *across the boxes,* this is not the case --- training on random SFT examples gets the highest RL performance, but training on the shortest SFT examples gets the highest post-SFT performance. Would it be better to just include the bars that actually matter, instead of all of these? 
- In §3.1, the experimental setup says that you "vary the training configuration, such as the number of unique samples/training epochs/learning rate," but not what was ACTUALLY varied.
- Consistently throughout the paper, *all* the x-axis bar labels start with the name of the dataset, e.g., "Nemotron 25k2ep", "Nemotron 25k3ep","Nemotron 25k4ep", etc. Authors should omit the redundant information and instead specify exactly the information that changes, e.g., "2ep" "3ep", "4ep." Also, all the $x$-axis labels should be clearer, and not require the reader to infer that "25k2ep" means 2 epochs on a dataset of 25k examples.
- More small notes on writing that did not factor into my evaluation but may improve readability.
    - L14: "In this work, we challenge whether high SFT scores translate to improved performance after RL." RL still improves performance, right? Instead, you're saying that the ranking of SFT models isn't predictive of the ranking of those models after RL.
    - L15: "We provide extensive counter-examples where this is not true." It is more natural to say "counter-examples to [claim]." Counter-examples, by definition, are instances where X is not true.
    - RLVR citation switches from Deepseekv3 to Tülu3 in the middle of the paper. Tülu3 came first and proposed RLVR!
    - L64: "industrial practice" $\rightarrow$ "industry practice"
    - L77: "on the contrary" is not the right transition to use here because the following evidence supports the same finding. Maybe "on the other hand" is better?
    - L291: "During the investigation above, we identified a counterintuitive pattern in which post-SFT performance improves stably when training for more epochs whereas the overtrained models show decreased potentials during the subsequent RL." $\rightarrow$ "In our previous experiments, we identified a counterintuitive pattern where training for more epochs consistently improves SFT performance but deteriorates potential for subsequent RL. "

### Questions
1. It is a bit odd that validation loss was deemed not applicable for instance-level predictions (§5.3). It is still possible to evaluate its performance, right? It does not take away from the story at all to say that pass@k is a strong predictor in both dataset-level and instance-level settings, but validation loss works well only in the former.
2. Would it be accurate to refine the claim "the model with the strongest post-SFT performance is not necessarily the one with the strongest post-RL performance" to add "*when evaluated on the same data*"? That makes the claim stronger + more interesting.

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
The paper challenges the previous conventional practice where the model that achieves the best performance after SFT is assumed to be the best candidate for subsequent RL training. Specifically, the paper shows that the correlation between SFT scores is not predictive for RL scores by training hundreds of models under various settings. To solve this problem, the paper proposes alternative metrics including generalization loss on held-out sets and Pass@larger k and show that they demonstrate much higher correlation.

### Strengths
- The paper targets at a very practical assumption that the better performance after SFT always leads to better performance after RL and the consequent difficulty to find the cause for a bad end performance especially when two different teams are in charge of these two training stages.
- This work spends a large amount of computational resources to make and verify their claim.
- The experiments include evaluations on various math benchmarks, and the correlation gains are significant.

### Weaknesses
- The paper only evaluates on benchmarks in the math domain, and whether the method can be generalized to other domains like coding and science remains unclear.
- For dataset-level scenarios, most of the experiments focus on altering the training epochs and the number of unique examples, which seems to be a less practical tradeoff to consider. Training more epochs could lead to potential overfitting is a well-known concern, so even in the scenarios where the number of high-quality examples are limited, training for many epochs might not be a conventional option.
- For instance-level scenarios, the strategy seems to be selecting the random / shortest / longest examples, varying the number of total examples. This seems to deviate from the actual practice in SFT data selection, and the paper needs some analysis to show how close these simple strategies are to the current data selection methods.

### Questions
It seems that the SFT / RL training datasets are different across models. How are these choices made?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper challenges the existing belief that the pass@1 accuracy of an SFT model is a good metric for determining the model's performance after RL. The authors perform experiments using various LLM architectures on math-focused datasets that show that when modulating training parameters (e.g. epochs) or data parameters (e.g. # unique training data points), the SFT model with the highest pass@1 accuracy is often not the best performing model after RL. Instead, the authors propose validation loss and pass@k accuracy as good metrics for performing this prediction, and perform further experiments to show that this is true.

### Strengths
1. This work suggests metrics which better align with post-RL performance compared to task accuracy
2. Extensive GPU resources are used to perform experiments that support the claim
3. The described methods are easy to apply to existing SFT trained models for ranking and selection

### Weaknesses
1. The SFT-then-RL pipeline was popularized for LLM usage in [1], and used previously in NLP as cited in their paper. Citations should be fixed.
2. (Section 3, line 214), (Section 3.1, line 222), (Section A.1, line 711), (Section B.2, line 900) conflict on which parameters (especially learning rate) were modified during RL training.
3. (Section 3.1, line 250) and (Figure 3) lacks intuition on why post-SFT task performance is expected to have a linear relationship with post-RL performance.
4. (Figure 1), (Figure 4), etc. show that best SFT's post-RL task performance usually only differ from best RL by a few points. Error bars or other standard deviation reporting would be enlightening.
5. Grammatical error in (Section 6, line 483), "...algorithms may worth further explorations..."
6. Considering pass@1, pass@k, generalization loss, and Spearman's rank correlation are central to the paper's claims, a brief exposition on their definitions would be good to include
7. (Section 5.2, line 394) How many models is 50%?


**References**

[1] Ouyang, Long, et al. ‘Training Language Models to Follow Instructions with Human Feedback’. arXiv:2203.02155, arXiv, 4 Mar. 2022. arXiv.org, https://doi.org/10.48550/arXiv.2203.02155.

### Questions
0. See Weaknesses
1. What kind of hyperparameter search was done for each set of experiments?
2. How do the predictions (based on gen loss, pass@large k) hold up when constrained to a single task instead of the average of 7 tasks?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper investigates why high Supervised Fine-Tuning (SFT) scores often fail to translate into good Reinforcement-Learning (RL) post-training results.
Through large-scale experiments across Llama3-8B, Mistral-NeMo-12B and Qwen3-4B on mathematical reasoning tasks, the authors show that excessive SFT can overfit the model’s representation space, reducing its capacity to benefit from RL.
Two new proxy indicators are proposed:
Generalization Loss – captures when validation loss flattens while diversity collapses;
Pass@large k (k = 64) – estimates solution-space width.

### Strengths
1. Demonstrates convincingly that high SFT scores can mislead RL training, challenging a community-wide assumption.
2. Cross-model, multi-dataset evaluation with some interesting statistics (R² and Spearman correlation) makes the results good

### Weaknesses
1. The paper never demonstrates that using Generalization Loss or Pass@64 for early-stopping or checkpoint selection actually improves final RL accuracy.
All reported metrics are correlations (R² / spearman's rank correlation); no absolute post-RL accuracy table is shown.
2. From Fig. 4–5, the best RL model surpasses the best SFT by only ≈ 1~2%, and the actual other early stop strategy might not be able to reach the best RL model, so the real-world benefit appears even smaller.

### Questions
1. Could you report one concrete example where checkpoints chosen by Pass@64 or Generalization Loss actually outperform standard validation-loss selection after full RL?
2. How sensitive are results to the choice of k (32 vs 64)? Would smaller k approximate the same signal?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 6

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper challenges a widely-held assumption in LLM post-training: that high supervised fine-tuning (SFT) scores reliably predict better performance after reinforcement learning (RL). Through experiments on moderate models trained with hundreds of variations and evaluated on 7 math benchmarks, the authors demonstrate numerous counterexamples where this assumption fails. High SFT scores can be biased toward simpler or more homogeneous data and do not reliably predict subsequent RL gains or scaled-up post-training effectiveness. In some cases, RL training on models with improved SFT performance actually produces substantially worse outcomes compared to RL on base models without SFT. To address this predictability gap, the paper identifies two more reliable metrics for predicting post-RL success: generalization loss on held-out reasoning examples and Pass@large k performance.

### Strengths
This paper makes a practically important contribution to LLM post-training by systematically demonstrating that SFT performance is a poor predictor of final RL outcomes. The proposed metrics are intuitive and practical. The clear categorization into dataset-level and instance-level scenarios makes findings applicable to different use cases, and the promised open-source tool adds concrete value.

### Weaknesses
The study is narrowly scoped to mathematical reasoning tasks using GRPO-based RL with verifiable rewards, limiting generalizability to other reasoning domains (coding, science, logic) or alternative RL paradigms (offline RL, DPO, other algorithms). The paper identifies failure modes but provides limited mechanistic insight into why SFT characteristics diverge from RL outcomes. For instance-level prediction, generalization loss is explicitly not applicable due to distributional gaps, reducing its utility for practical data selection scenarios. The proposed linear predictor requires calibration data from actual RL runs, which partially defeats the purpose of avoiding expensive RL experiments. Additionally, the Qwen3 results are presented only qualitatively because models showed minimal responsiveness to SFT variations, suggesting the findings may not universally hold across all architectures.

### Questions
(1) Can the authors provide mechanistic explanations for why overfitting during SFT constrains RL exploration? Is this about reduced diversity in model outputs, behavioral rigidity, or something else? (2) How sensitive are the proposed metrics to hyperparameter choices in SFT (learning rate, batch size, optimization algorithm) beyond epochs and data volume? (3) Since generalization loss requires a validation set from the same SFT distribution while Pass@large k works across distributions, why not simply use Pass@large k for both scenarios, and what are the trade-offs? (4) Can these findings transfer to other domains (code, science reasoning, non-verifiable tasks) and other RL methods (offline RL, preference-based methods), or are they specific to online GRPO with math tasks?

### Soundness
2

### Presentation
3

### Contribution
2
