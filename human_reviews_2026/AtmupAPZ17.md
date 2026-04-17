# SkipPipe: Partial and Reordered Pipelining Framework for Training LLMs in Heterogeneous Networks

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 6, 4

## Abstract
Data and pipeline parallelism are ubiquitous for training of Large Language Models (LLM) on distributed nodes. The need for cost-effective training has lead recent work to explore efficient communication arrangement for end to end training. Motivated by LLM's resistance to layer skipping and layer reordering, in this paper we explore stage (several consecutive layers) skipping in pipeline training, and challenge the conventional practice of sequential pipeline execution. We derive convergence and throughput constraints (guidelines) for pipelining with skipping and swapping pipeline stages.
Based on these constraints, we propose SkipPipe, the first partial pipeline framework to reduce the end-to-end training time for LLMs with negligible effect on convergence, which we verify analytically and empirically. The core of SkipPipe is a path scheduling algorithm that optimizes the paths for individual microbatches and reduces 
their end-to-end execution time, complying with the given stage skipping ratio.  We extensively evaluate SkipPipe on LLaMa models from 500M to 1.5B parameters on up to 20 nodes, through emulation and deployment prototypes. Our results show that SkipPipe reduces training iteration time by up to 50\% compared to full pipeline.
Additionally, our partial pipeline training also improves resistance to layer omission during inference, experiencing a drop in perplexity of only 2\% when running only 75\% of the model.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
A new pipeline parallel training method for distributed LLM training is introduced, where layers now allowed to be swapped and micro-batches can skip some fraction of layers. This results in better training throughput, which can also transfer to inference settings. 

The authors evaluate simulated and real training settings and show that models which have been trained using the layer skipping method are more robust to layer skipping during inference.

### Strengths
- the proposed method has impressive results, reporting up to 250% speedup over no layer skipping
- only small drops in perplexity compared to no layer skipping are reported in experimental evaluations
- as a byproduct of training models using the SkipPipe algorithm, they can also be sped up during inference using layer skipping

### Weaknesses
W1. The paper propose a method for pipeline parallel training, which is sued for training large models. However, all experiments use quite small models, where such a parallelism method would never be used in practice. Unfortunately, this severely limits the strength of this publication. From a computational standpoint, running larger models should not be that expensive, since you only need to run a small amount of iterations to compute the throughput metrics. We need to be able to have some quantification of how the proposed method scales to larger models which is the setting where it would be applied.

W2. In the same vein, although more expensive to evaluate: we do not know whether the favorable convergence behavior remains as we scale beyond the 1B model range. Also, there are no benchmark evaluations and only perplexity is reported, which is not always indicative of performance on actual tasks.

W3. While the real deployment speedups are impressive (250%!), these stem from a "ultra-distributed" setting across continents. There do not seem to be real deployment evaluations in a setting within a single datacenter.

### Questions
Q1. In Figure 4b, using 25% skipping seems to lead to a significantly smoother validation loss curve than no skipping. Do you have an intuition as to why this might be the case (and is not the case in Fig 4a)?

Q2. The abstract and conclusion state an "up to 50%" speedup but in your experiments your report up to 250%. Is this a typo?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes SkipPipe, a novel framework to randomly skip some stages of PP for LLM training. The core contribution is two-fold: 
- With both theoretical and empirical studies, the authors show that skipping a proportion of LLM layers has negligible impact on the loss convergence.
- Skipping these layers results in end to end training accelerations. Evaluations show up to 50% acceleration with no significant accuracy degrade.

### Strengths
- A novel design on the path finding and scheduling algorithm to avoid collisions.
- Evaluation shows significant training acceleration via both simulation and geo-distributed experiments.
- Evaluation on convergence shows similar training loss curves compared to baseline. Moreover, during inference it shows robustness against layer dropping.

### Weaknesses
Major Concern:
The primary concern is with the methodology of pipeline stage skipping. While there is literature on similar topics, existing work does not fully support the authors’ claim of negligible accuracy loss; in some cases, it suggests the opposite. For example, LayerSkip [1] consistently frames early exits as a trade-off between computation and accuracy. Figure 5 in LayerSkip shows significant accuracy degradation during continual pretraining when early exiting is applied. To address this, LayerSkip proposes self-speculative decoding to correct the accuracy. In line with this, I concur that optimizations like SkipPipe should be treated as a trade-off between compute efficiency and model performance, rather than claiming negligible impact on accuracy. At minimum, additional evidence should be provided to support the claim that LLMs are robust to SkipPipe-style stage skipping, which is not convincingly demonstrated in the current paper.

Other Concerns:
- Limited evaluation and scale: The experiments are conducted on a 0.5B model for pretraining and a 1B model for fine-tuning. It remains unclear how SkipPipe affects convergence and performance on larger models with more layers or stages.
- Performance degradation on downstream tasks: Table 4 shows noticeable accuracy drops across most tasks, which appears to contradict the claim that SkipPipe does not affect accuracy.
- Theoretical analysis limitations: The theoretical discussion focuses on individual layers, whereas SkipPipe skips entire stages composed of consecutive layers. The analysis may therefore not fully capture the impact of the proposed method.

Suggestions:
I recommend the authors reframe SkipPipe as an approach that enables an effective trade-off between accuracy and latency, rather than claiming negligible accuracy loss. It is counterintuitive that skipping layers would fully preserve accuracy. The paper could be strengthened by demonstrating that SkipPipe achieves better efficiency compared to alternative methods with the same goal, such as model pruning, quantization, or direct model size reduction, especially in geo-distributed setups.

Reference:
[1] LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding, ACL 2024. https://aclanthology.org/2024.acl-long.681.pdf

### Questions
- Is the theoretical analysis still correct considering SkipPipe is dropping consecutive layers?
- Is there a significant overhead for the path scheduling algorithm to run?
- There might be some confusion on the evaluation setup, the authors explains the 250% acceleration against DT-FM is caused by improved network compared to DT-FM experiments. Can authors clarify that whether the network and hardware setup are the same for SkipPipe and DT-FM?

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
This paper presents an algorithm to optimize the training of an LLM--here synonym with decoder transformer--over distributed nodes, given the bandwidth and latency between any of them.

Five constraints are identified related for three of them to the tolerance of a decoder transformer to change in the order of evaluation of its layers, and for two to the memory constraints and the necessity to avoid collision. The dispatching of minimatches and stages reduces to a  discrete (and complex) discrete optimization problem.

Experiments demonstrate excellent performance compared to baselines.

### Strengths
The problem addressed is very important, as it could unlock large-scale collaborative training and more generally allows a better use of existing infra. The method is well justified and simple in the statement of the optimization problem, but involve some technicalities for the optimization. The performance seems excellent compared to baselines, i particular the results with a "real" setup of Appendix E are very reassuring. The writing is clear (except the explanation related to TC2).

### Weaknesses
The methods is restricted to a specific, although currently super dominant, class of models. The experiments are unfortunately mostly simulation, and far from large scale, which is quite problematic for such a class of methods. Performance are not given in tangible quantities (e.g. wall clock times), it may be clear to people in that precise field, but it would then be helpful to have even rough estimates.

### Questions
1. I presume speed measurement do not need more than a few percent of a full training, why no experiments with a 8B or bigger? How does this method combines with e.g. FSDP ?

2. Please re-explain TC2, which seems to be a heuristic more than a constraint.

3. There seems to be no substantial degradation in Fig 4 (a), what happens with e.g. 50%, 75%, 90% ?

4. Is DT-FM the best available alternative? How does this compare with a standard cluster?

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
4

### Summary
This work proposes a new pipeline-parallel approach to pre-training large language models, SkipPipe, grounded in the empirical robustness of LLMs to layer skipping and layer reordering. SkipPipe can intelligently schedule data paths at the micro-batch level to boost the efficiency of LLM pre-training. The results show robust performance and time savings compared to the full pipeline approach. The authors also observe robustness to layer omission during inference for models trained with SkipPipe.

### Strengths
- S1. This work tackles a critical and expanding area of ML: efficient LLM pre-training on geo-distributed networks.
- S2. The proposed method, SkipPipe, is based on established observations regarding the robustness of LLM to specific perturbations, i.e., layer skipping and layer re-ordering. Leveraging the known robustness of LLMs to skipped or shuffled layers is a clever idea that underpins SkipPipe’s design.
- S3. Despite being based on established techniques, SkipPipe has the merit of coupling them in a principled way, driven by constraints and scheduling algorithms, producing predictable savings and emergent robustness.

### Weaknesses
- W1. The overall training speedup due to SkipPipe’s scheduling algorithm, beyond existing techniques that also skip layers, appears modest. While SkipPipe achieves up to 50% speedup over a standard pipeline, a large portion of this gain comes simply from skipping layers rather than the new scheduling; the novel contribution yields only an incremental improvement (~10–20% beyond a baseline that already skips layers).
- W2. The experimental evaluation is incomplete in terms of demonstrating the model’s effectiveness. The paper evaluates training speed and shows validation perplexity, but it lacks downstream task evaluations (e.g., in-context learning tasks or other language understanding benchmarks) to show the impact of SkipPipe on actual performance. Including such experiments would strengthen the paper.
- W3. The contribution has limited novelty. Similar ideas of skipping layers or reordering in training have been explored (e.g., layer drop for skipping layers during training, or prior heterogeneous scheduling methods for pipeline arrangement). SkipPipe’s main innovation is integrating these existing ideas (skip training and path scheduling) – a useful engineering improvement, but arguably an incremental advance rather than a fundamentally new concept.
- W4. The paper’s claims about ‘robust inference’ (tolerating dropped layers) are not well contextualized relative to prior work. Prior research (e.g., dropout, layer drop) shows that training with missing parts can improve robustness. The authors should clarify whether this phenomenon is unique to SkipPipe or simply an expected outcome of partial training (and compare to a baseline model trained normally: how much worse would a standard-trained model perform if some layers are skipped at inference?). Currently, the robustness result is presented without discussing such context or baselines, leaving it unclear how novel or significant this finding is
- W5. The authors provided an anonymous code repository link, but the code is currently unavailable (the repo is empty/inaccessible at review time), hindering reproducibility.
- W6. The writing could be clearer in places. Some sections (e.g., the theoretical constraints or algorithm description) were hard to follow due to dense notation and could be better explained. Improving the clarity and presentation (figures, notation) would help readers understand the contributions.

### Questions
- Q1. Can the authors discuss the limitations of assuming a static (B, Lambda) graph and how their method would change if such a graph is dynamic (changing over time)?
- Q2. The evaluation of model quality is very limited. Can the authors add sensible ICL downstream tasks to their evaluation instead of the limitedly insightful validation perplexity on some benchmarks?
- Q3. Most time savings compared to the baseline seem to come from skipping layers. Can the authors quantify how much communication (in bytes) DT-FM-skip and SkipPipe are saving with respect to the baseline DT-FM? How strong is the correlation between such savings in bytes and the savings in time? This comparison could help readers understand SkipPipe’s contributions.

### Soundness
3

### Presentation
2

### Contribution
2
