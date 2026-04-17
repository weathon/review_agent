# Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
This paper presents *the first study of grokking in practical LLM pretraining*. Specifically, we investigate when an LLM memorizes the training data, when its generalization on downstream tasks starts to improve, and what happens if there is a lag between the two. Unlike existing works studying when a small model generalizes to limited and specified tasks during thousands epochs' training on algorithmic data, we focus on a practical setting for LLMs, i.e., near single-pass pretraining of next-token prediction on a cross-domain, large-scale corpus, and generalization on diverse benchmark tasks covering math/commonsense reasoning, code generation, and domain-specific retrieval. Our study, *for the first time, verifies that grokking still emerges in pretraining mixture-of-experts (MoE) LLMs*, though different local data groups may enter their grokking stages asynchronously due to the heterogeneity of their distributions and attributions to others. To find a mechanistic interpretation of this local grokking, we investigate the dynamics of training data's pathways (i.e., expert choices across layers in MoE). Our primary discovery is that *the pathways evolve from random, non-smooth across layers, instance-specific to more structured and transferable across samples*, despite the converged pretraining loss. This depicts a transition from memorization to generalization. Two novel metrics are developed to quantify these patterns: one computes the pathway similarity between samples, while the other measures the consistency of aggregated experts between subsequent layers for each sample. These training data based metrics induce near zero cost but can faithfully track and monitor the generalization of LLMs on downstream tasks, reducing reliance on costly instruction tuning and benchmark evaluations. We also ground our findings in a theoretical analysis of one-layer MoE, showing that more structured pathways improve the generalization bound.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates whether grokking the delayed transition from memorization to generalization observed in small synthetic settings also occurs during MoE pretraining.
Using the OLMoE model, the authors analyze routing dynamics across checkpoints and introduce two new metrics, Pathway Similarity and Pathway Consistency, which aim to monitor generalization without external validation data.
They find that as generalization improves, inputs increasingly follow similar expert routes, and routing transitions between layers become more coherent. The proposed pathway-based measures strongly correlate with downstream performance, suggesting that they can serve as internal indicators of generalization in MoE models. These findings position routing-based metrics as potential tools for real-time monitoring of generalization in large or resource-constrained training regimes, without the need for held-out validation data.

### Strengths
1. Overall, the paper is clearly written and easy to follow, with well-motivated experiments and clear presentation of figures.

2. It provides an insightful analysis of routing dynamics in MoE model, OLMoE.
It convincingly shows that as generalization improves, inputs increasingly follow similar expert routes, suggesting an internal reorganization of computation during training. This analysis of how MoE models acquire generalization ability could potentially inform better training or routing strategies in future work.

### Weaknesses
1. While the paper presents an interesting and clearly written analysis, its empirical scope and generality remain limited. The entire analysis and validation are restricted to the OLMoE model.
It remains unclear whether the proposed metrics hold under different MoE configurations for example, varying the number of experts, the top-k routing parameter, or the load-balancing loss.
Recent works such as [1,2,3] have explored different routing and balancing objectives to prevent expert collapse or to improve hardware utilization.
Under these regimes, it is uncertain whether the proposed pathway metrics would behave consistently or still correlate with generalization.

2. Questionable “one-epoch” claim.
The authors state that grokking occurs “during one-epoch pretraining.”
However, according to [4], OLMoE was trained for 1.3 epochs (“We shuffle all samples randomly at the beginning of each epoch and train for a total of 5.133T tokens (1.3 epochs following Muennighoff et al. [121]).”).
Furthermore, the DCLM-Baseline dataset used for training reportedly contains ~80 % duplicates ([5]), meaning that the model effectively saw many samples multiple times.

3. Line 324 discusses that shallow layers show a steep decline in after memorization, while deeper layers show smaller reductions or even increases. Similar depth-dependent specialization effects have already been discussed in [6].

4. While the authors argue that their metrics could remove the need for held-out validation, this claim seems optimistic.
Because the proposed metrics are only defined for MoE models and depend on architecture-specific details such as depth, number of experts, and routing their use as a universal proxy for generalization appears limited.
In practice, held-out monitoring may still be necessary for comparability across different settings.

[1] https://arxiv.org/abs/2408.15664

[2] https://arxiv.org/abs/2412.19437

[3] https://arxiv.org/abs/2501.11873

[4] https://arxiv.org/abs/2409.02060

[5] https://huggingface.co/datasets/Zyphra/dclm-dedup

[6] https://arxiv.org/abs/2406.18219

### Questions
1. Given these routing observations, do the authors see potential for improving MoE training efficiency for instance, by designing better routing or load-balancing strategies informed by pathway dynamics?
If so, how might these insights translate into practical optimization or architectural adjustments?

### Soundness
2

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
2

### Summary
This paper investigates whether grokking-delayed generalization after training loss/accuracy has converged-appears in LLM pretraining. Using public checkpoints of a 7B MoE model (OLMoE), the authors (i observed asynchronous (“local”) grokking across domains (math, code, commonsense, domain QA), (ii introduce two routing-pathway metrics for MoE that are cheap to compute during pretraining and (iii) provide a routing-kernel NTK analysis linking decreased “effective dimension” to improved generalization. Empirically, the two pathway metrics are strongly correlated with downstream benchmark gains (after light LoRA instruction tuning), while conventional training/validation losses poorly track those gains. The paper claims these metrics offer near-zero-cost, evaluation-free monitoring of LLM generalization.

### Strengths
1. The author uses realistic setting like One-epoch, heterogeneous web-scale data, public 7B MoE checkpoints, and diverse domains. 
2. The author provides evidence of asynchronous memorization and delayed generalization, including matched training/test groups and domain-dependent lags.
3. The author uses public checkpoints/datasets and explicit data-contamination filtering which has good reproducibility.

### Weaknesses
1. The paper mentions “virtual pathways” for dense models as future work; however, the core claim (test-free generalization monitoring) would be much stronger with experiment on a small-scale dense model.
2. Consider replacing “zero-cost” with “near-zero-cost” or “cheap to compute” to avoid overstatement.

### Questions
1. If you change LoRA rank/targets or remove LoRA, do correlations with pathway metrics persist?
2. Have the authors considered causal interventions--such as randomly dropping some experts or adding entropy penalties--to verify whether enforcing more structured routing actually causes earlier or stronger generalization (as opposed to merely correlating with it)
2. How sensitive are the results to the memorization thresholds, the routing cumulative-weight cutoff, and router temperature/noise during logging?

### Soundness
3

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
This paper examines grokking in practical LLM pretraining. It investigates when memorization occurs, when downstream generalization improves, and whether a lag exists between them in a realistic one-epoch, next-token pre-training setup spanning diverse domains and tasks. 

Focusing on mixture-of-experts (MoE) LLMs, the paper shows that grokking still emerges, with local data groups entering the grokking phase asynchronously due to distributional heterogeneity. 
To explain this “local grokking,” the authors analyze training data pathways (expert selections across layers) and find that these pathways evolve from random, instance-specific patterns to more structured and transferable ones, even after the pre-training loss has converged. 

They introduce two zero-cost metrics, (1) pathway similarity between samples and (2) layer-to-layer consistency of aggregated experts for a sample, that reliably track downstream generalization without instruction tuning or expensive benchmark evaluations. 

A theoretical analysis of a one-layer MoE indicates that more structured routing tightens the generalization bound. 
Empirically, an analysis of a 7B-parameter MoE model confirms that generalization can arise well after memorization.

### Strengths
* Zero-cost, pathway-based indicators derived from MoE routing dynamics  
The paper proposes two metrics, sample-to-sample pathway similarity and across-layer pathway consistency, that can track the rise of downstream generalization without instruction tuning or benchmark evaluations.
LLM evaluation is expensive. Leveraging internal routing information to estimate generalization progress directly during pretraining is highly cost-effective.

* Discovery of “local grokking” under data heterogeneity  
The paper reports asynchronous grokking across local data groups due to distributional heterogeneity and differing attributions to other groups.
Real-world corpora are heterogeneous; focusing on local dynamics captures behaviors that global averages miss, informing data curricula and domain-wise monitoring at scale.

* Theoretical support via a one-layer MoE analysis  
The analysis indicates that more structured pathways improve the generalization bound.
Providing a theoretical link between pathway structure and generalization strengthens the validity and potential generality of the empirical findings.

### Weaknesses
* Limited empirical scope (single model)  
I understand there are no other publicly available MoE checkpoints, but the paper's results and discussion are tailored to the specific OLMoE 7B model.
Ideally, robustness should be assessed across a broad range of choices in optimization, model design, and training data, such as learning rate schedules, model scales, expert capacity and number, and data mixtures.
Otherwise, the paper's results and findings may be model-specific or biased by the model used.
From this perspective, the paper's findings and conclusions may be unreliable or not yet conclusive.

* Correlation versus causation remains underexplored     
Although strong correlations between pathway structure and generalisation are evident, intervention studies (e.g., deliberately altering routing) are limited.
Other factors, such as learning rate, load balancing, and regularisation, could confound this link.
For example, we usually downscale the learning rate during pre-training, so the weight change essentially stabilises in the later part of the training phase.

* [Minor] Although this paper examines only a single MoE model, the title "Grokking in LLM Pretraining?" suggests that it covers all LLM pretraining cases, and I feel there is a large gap between them.

### Questions
* I find it challenging to understand the details of how the training-test paired datasets are constructed as described around lines 234--239, beginning with “Accordingly, we group ...”.
Could the authors please clearly describe the detailed procedure for constructing these datasets?
In addition, please explain how this relates to the notation such as “Pretrain@370k -> Test@855k” and “Pretraining Objective (memorized @ 125k)” in the figures.
I believe that an exact understanding of how these datasets are constructed is crucial, as it directly affects the results and conclusions.




* Reproducibility of investigation and analysis:  
While the results and findings of this paper depend heavily on the training-test paired datasets.
I feel that reproducing these datasets is difficult for readers (other researchers).
Would the authors consider releasing the logs and code used to generate the datasets, as well as the datasets themselves, to improve reliability and to support future follow-up studies?

---
I am open to revisiting the overall assessment following discussions with the authors, particularly in light of the concerns and questions outlined above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the behavior of grokking in LLMs by looking at pathways among experts in MoEs. Using OLMoE checkpoints, the authors report high correlations between the proposed metrics and post‑hoc downstream scores. They define a pathway edit distance and a per sample pathway consistency and observe that these metrics correlate with jumps on downstream benchmark scores after lightweight LoRA instruction tuning. They also provide a NTK‑style bound for a one‑layer MoE with fixed routing, arguing that decreased effective dimension aligns with reduced pathway complexity.

### Strengths
Previous studies of grokking mostly used small models trained on synthetic algorithmic tasks, but this work examines a 7‑billion‑parameter mixture‑of‑experts (MoE) model (OLMoE) and shows that grokking still appears in one‑epoch pretraining.

The observation that that grokking in LLMs are local and asynchronous broadens our understanding of grokking at scale.

The metrics they define; pathway edit distance and a per sample pathway consistency, rely only on pretraining data and internal activations, so they can be computed during training without external validation.

### Weaknesses
The conversion of top‑k experts per layer into comma‑separated strings and computing Levenshtein distance is ad‑hoc, since edit distance is sensitive to sequence length and arbitrary thresholding. This distance can also decrease simply due to stronger load‑balancing or saturated routers.

The bound assumes fixed routing and an NTK regime for a one‑layer MoE, while in practice OLMoE updates routing and experts jointly across many layers for trillions of tokens.

### Questions
In Fig. 4, Please clarify whether distances are computed token‑wise or sequence‑wise, and whether you average per‑token pathways or per‑sequence pathways.

All generalization is measured after LoRA instruction tuning on different datasets. Is it possible that the observed jumps may be influenced by finetuning dynamics, not purely grokking during pretraining?

### Soundness
3

### Presentation
3

### Contribution
3
