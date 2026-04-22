# The Anatomy of Alignment: Decomposing Preference Optimization by Steering Sparse Features

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Prevailing alignment methods induce opaque parameter changes, obscuring what models truly learn. To address this, we introduce Feature Steering with Reinforcement Learning (FSRL), a framework that trains a lightweight adapter to steer model behavior by modulating interpretable sparse features. First, we theoretically demonstrate that this mechanism is expressive enough to approximate the behavioral shifts of post-training processes. We then apply FSRL to preference optimization and perform a causal analysis of the learned policy. Our analysis reveals a crucial insight: the model learns to reward stylistic presentation as a proxy for quality, disproportionately relying on features related to style and formatting over those tied to alignment concepts like honesty. By effectively optimizing the preference objective, FSRL serves as a transparent proxy for observing the alignment process. Overall, FSRL offers an interpretable control interface and a practical way to diagnose how preference optimization pressures manifest at the feature level.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes Feature Steering with Reinforcement Learning (FSRL) — a framework that aligns large language models by steering one-layer interpretable sparse features rather than updating dense parameters directly. This allows for more transparent and auditable control over model behavior during preference optimization.

### Strengths
1. The proposed FSRL algorithm empirically optimizes preference alignment tasks using an interpretable basis, providing a possible interpretable framework for downstream task fine-tuning.
---
2. The authors provide causal analysis using zero-out steering on different feature categories and find evidence that preference alignment places most pressure on stylistic presentation rather than alignment-related concepts such as safety, honesty, and helpfulness. 
---
3. Given the assumptions of local linearity and the empirical sparsity of active features, the authors build the theoretical relation between FSRL and the expressive power of LORA.  Specifically, the authors prove that the FSRL steering can be interpreted as an affine map given the local linearity assumption and relate FSRL’s soft-thresholding in sparse features to LoRA’s low-rank decomposition given the empirical sparsity assumption.

### Weaknesses
1. Requires more rigorous analysis of the trade-off between math reasoning degradation and preference alignment. It's unclear whether the preservation of math reasoning capability results from limited learnable capacity or from the authors' proposed methodology.
---
2. Table 3's intervention over feature activations is insightful, but more analysis is needed. What is the intervention loss difference between the unmodified model and the FSRL-trained model? Would the FSRL-trained model rely more on alignment features rather than disproportionately prioritizing style features? If not, what are the implications of observing this disproportionate prioritization of style-related over alignment-related features during preference alignment? The paper lacks empirical evidence supporting the claim that FSRL "provides a controlled framework for auditing alignment pressures".
---
3. The performance of FSRL is not sufficiently convincing (see Table 1). More empirical experiments are needed to demonstrate the benefits of FSRL—beyond interpretability—compared to interpretable model-diffing, e.g., other interpretable steering baselines like SAE-lens direct editing or neuron activation control.

### Questions
My questions are presented in the Weaknesses section.

### Soundness
3

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
3

### Summary
This paper introduces Feature Steering with Reinforcement Learning (FSRL), a framework for interpretable alignment of large language models. With pretrained sparse autoencoder (SAE) and frozen LLM, FSRL trains a lightweight adapter to modulate interpretable SAE features of a frozen LLM, rather than performing full-model fine-tuning. Additionaly, the authors theoretically establish that FSRL's activation-space corrections are equivalent to a restricted class of LoRA updates, inheriting LoRA's expressive power. Empirically, they apply FSRL to preference optimization on the UltraFeedback dataset using the Gemma-2-2B-it model and pretrained SAEs from GemmaScope, and demonstrate a moderate gain in MMLU, TruthfulQA, GSM8K compared to baseline while enabling interpretable interface. Through causal analysis based on their method, they find that style features contribute more than alignment features, suggesting preference optimization treats stylistic presentation as a proxy for quality rather than prioritizing deeper alignment concepts.

### Strengths
< Strength >

- The work addresses an important and timely problem. The opacity of standard alignment methods makes it difficult to diagnose issues like reward hacking. The use of SAE is theoretically justified and empirically shown for its interpretability.
- The theoretical justification connecting FSRL to LoRA updates is well-constructed, providing principled grounding for the approach. The limitation of the justification, the needs for adaptation across all layers, and empirical evidence is also discussed sufficiently.
- The paper is well-written with precise specification of the methodology. Figure 1 effectively illustrates the architecture.
- The causal ablation study provides unique compelling evidence for the disproportionate importance of style features, going beyond correlational analysis and has practical impact

### Weaknesses
< Weakness >

- The claimed advantage on interpretability still on a coarse high-level. Although the main advantage of the method is interpretability, the paper doesn’t have generation samples for steered generation except one in Figure 1. The fact that the policy was highly distributed (mentioned in the Appendix) implies the interpretability based on the proposed method remains on coarse level. As Figure 1 describes the methods as a sample-wise fine-grained steering, the paper should explicitly mention about the coverage in the main paper.
- Similarly, the analysis focuses on relatively simple, coarse-grained concepts of alignment vs. style, and MCC 0.448 for alignment features also raise concern for the substantial noise.
- Further, given the marginal performance gains in Table 1, this limited interpretability is difficult to justify.
- SimPO loss is used as the primary metric throughout (Tables 1 and 3), but this is merely the training objective, not a direct measure of alignment quality. Larger gap on SimPO loss between Base and FSRL (37.19%) in Table 1 despite a marginal gain on downstream performance (14.44% on MMLU and 0.71% on TruthfulQA) also raise the question on the performance. The addition of standard metrics like win-rates evaluations (even with LLM-as-judge) would bridge this gap.
- The degradation of full fine-tuning on mathematical reasoning (GSM8K) is acknowledged as a limitations from SimPO, however the paper doesn’t explain why FSRL shows significantly less degradation than full fine-tuning despite both optimizing the same SimPO objective.
- Is this due to insufficient training? The authors mention checking SimPO loss convergence would help
- DPO, a more popular and GSM8K-friendly baseline, was not evaluated despite acknowledged SimPO limitations

< Minor issues >

- $\boldsymbol{\theta}\in R_{+}^{d_{\text{sae}}}$ in adapter is confusing with $\theta$ in $\pi_\theta$, as those two models are different modules.
- Typo in line 105, “adding adding”

### Questions
<  Questions >

- Can you provide more qualitative generation examples showing how style and alignment feature steering manifests in actual text? While Figure 1 offers one illustration, more diverse qualitative samples would help readers better understand the behavioral changes induced by FSRL and strengthen the interpretability claims.
- In lines 349-351, “It enables researchers to causally link undesirable behaviors to the optimization of specific feature categories, thereby clarifying their root causes.”, could you provide empirical evidence demonstrating that steering specific feature categories can successfully mitigate or induce undesirable behaviors? The current results in Table 1 show that FSRL does not fully match the alignment efficacy of full fine-tuning, which makes this claim somewhat difficult to catch from the presented experiments.
- Would it be possible to supplement the evaluation with more standard alignment metrics such as win-rates or LLM-as-judge assessments? These metrics would provide a more direct measure of alignment quality.
- Given that DPO is more widely adopted in the community and is known to work well with mathematical reasoning tasks, could you provide results using DPO as the optimization objective? This would help clarify whether the findings such as style-over-alignment phenomenon or FSRL better performing on math are specific to SimPO or represents a more general characteristic of preference optimization methods.

### Soundness
3

### Presentation
4

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
This paper introduces Feature Steering with Reinforcement Learning (FSRL), where an adapter is learned from preference data to steer model behavior. In particular, the adapter operates over the activations of a particular layer in the model. The adapter learns to perturb these activations by modulating additive features learned by SAEs that correspond to higher level semantic concepts (e.g., alignment or style). This results in a method that can steer model behavior by only updating the activations for a single layer in a model in a supposedly more interpretable manner.

The authors show that FSRL improves the preference loss of a 2B model on several standard benchmarks, comparing to SimPO. The authors then also show that the proportion of features corresponding the “alignment’ and “style” concepts decreases when training with FSRL, and that preventing adaptations to certain features (either “alignment” or “style”) impacts the model’s resulting loss differently. These results indicate that certain high-level features—in this case “style”—are more important for reducing preference loss than others. 

Finally, the authors also prove that FSRL updates fall within the class of LORA updates, providing theoretical justification for their method.

### Strengths
The author’s theoretical analysis makes an interesting connection between LORA updates and steering model behavior via updating only the activations of the model. Additionally, the paper is well written and easy to follow. The method, FSRL, is clearly explained and well situated in prior work. FSRL is also evaluated on standard benchmarks making it easy to understand comparisons to prior work.

### Weaknesses
I have two major concerns: (1) I am left unsure of the benefit of FSRL over full-fine tuning (2) I am not convinced of the novelty of this work. I additionally have a third, lower priority concern (3) the mechanistic insights could use more explanations or analysis. 

Regarding (1): Table 1 shows that SimPO outperforms FSRL across all benchmarks except for GSM8K, but here FSRL is outperformed by the base model. In terms of performance, it is not clear to me that there is any benefit to using FSRL; I don’t think it is sufficient to argue that FSRL is more performant than SimPO on one benchmark (although worse than the base model on that benchmark) but less performant on all others. Additionally, the authors of SimPO indicate that their method may do worse on GSM8K because of the training data; if you use different training data (e.g., perhaps more tailored to reasoning tasks) do you still see relatively better performance than SimPO? Its also not clear to me if FSRL possesses a benefit in terms of interpretability: the authors note that the FSRL adapter outputs a steering vector that is significantly more dense than the SAE’s features, and the empirical evaluations with regards to interpretability are limited. Finally, it is not clear if FSRL possesses any benefits in terms of computational expense: while it only requires limited adaptations to the models activations for a single layer, finding that layer required extensive search. Therefore, I am left unsure of where FSRL outperforms prior work. 

Regarding (2): This recent work from Bayat et al. (https://arxiv.org/pdf/2503.00177) also focuses on learning to steer a model by leveraging SAEs with targeted updates. Bayat et al. also learns how to steer model behavior from preferences, and evaluates on the same benchmarks as this submission. I think this work would be a fair comparison empirically—-or at least the authors should justify conceptually how their work differs.

Regarding (3): The authors note that, given Table 2, “the adapter learns to decrease the proportional activation of both alignment and style features relative to the baseline.” Why would the adapter learn to decrease the number of alignment features and style features? What type of behavior does this correspond to? My initial thought is that this observation might indicate that the adapter results in different features that are not classified as alignment or style features, possibly because these features are now different from the ones used to train the classifier. These empirical results do not tell me sufficient information about how the model behavior is changing. This analysis, without further explanation or connection to model behavior, feels incomplete.

### Questions
Where is the benefit of FSRL over prior work? 
What is the novelty of FSRL with respect to Bayat et al.?
Why do you observe decreasing proportions of alignment and style features after training with FSRL?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a method (FSRL) for preference alignment with a lightweight adaptor that also enables interpretable steering. The method uses the SimPO objective and an adaptor to transform activations into features to pass into a sparse autoencoder for updating the residual stream. The authors show that FSRL's updates are equivalent to an input-dependent LoRA update, conduct some ablations on the form of the adaptor itself, demonstrate that the method yields performance between that of full finetuning and the original base model, and show that the trained adaptor can be used to provide insights into the alignment process (through observation and intervention on the latent features).

### Strengths
1. The paper is well-written: Figure 1 makes the high-level idea clear and experimental setups are well-explained.
2. The paper performs ablations to motivate the setup.
3. The paper provides an interesting analysis on the contribution of different high-level categories (e.g., alignment, style) to the post-training objective.

### Weaknesses
1. The paper could benefit from more direct comparisons with related work, including other forms of steering as well as other adaptors (such as non-interpretable ones). The latter are not discussed in the related work, and additional experimental comparisons would strengthen the paper in the context of the existing landscape of training with adaptors and interpretable steering. For instance, how does the additional interpretability of this setup compare to vanilla adaptors for training?
2. The paper only runs experiments on a single model. Running experiments on one additional model would strengthen the paper (and potentially offer important practical insights on the application of this method). For instance, can this method with the same SAE be used on a model from a different model family?
3. To be able to more confidently draw conclusions from section 5, it would be helpful to propagate the error from the categorization process into the results of the analysis. Do the claims still hold when this is taken into account?

### Questions
Please see the above for questions!

### Soundness
3

### Presentation
3

### Contribution
3
