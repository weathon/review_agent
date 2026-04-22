# Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs

- Avg Score: 5.50
- Decision: Reject
- Scores: 8, 6, 4, 4

## Abstract
Large Language Models increasingly possess capabilities that carry dual-use risks. While data filtering has emerged as a popular pretraining-time mitigation, it faces significant challenges: labeling whether data is harmful is expensive at scale, and given improving sample efficiency with larger models, even small amounts of mislabeled content could give rise to dangerous capabilities.
To address risks associated with mislabeled harmful content, prior work proposed Gradient Routing (Cloud et al., 2024) - a technique that localizes target knowledge into a dedicated subset of model parameters so they can later be removed. We explore an improved variant of Gradient Routing, which we call Selective GradienT Masking (SGTM), with particular focus on evaluating its robustness to label noise. SGTM zero-masks selected gradients such that target domain examples only update their dedicated parameters.
We test SGTM's effectiveness in two applications: removing knowledge of a language from a model trained on a bilingual synthetic dataset, and removing biology knowledge from a model trained on English Wikipedia. In both cases SGTM provides better retain/forget trade-off in the presence of labeling errors compared to both data filtering and a previously proposed instantiation of Gradient Routing.
Unlike shallow unlearning approaches that can be quickly undone through fine-tuning, SGTM exhibits strong robustness to adversarial fine-tuning, requiring 7 times more fine-tuning steps to reach baseline performance on the forget set compared to a traditional unlearning method (RMU).
Our results suggest SGTM provides a promising pretraining-time complement to existing safety mitigations, particularly in settings where label noise is unavoidable.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes Selective GradienT Masking (SGTM) (an improved version of gradient routing) to remove undesirable knowledge while maintaining desirable performance for LLM pretraining. This approach works by performing:

- Partitioning the model parameters into retain and forget sets;
- During the backward pass, selectively masking out retain gradients when the data are sourced from the forget dataset;
- During the forward pass, selectively masking out forget parameters when the data are sourced from the retain dataset;

The authors evaluate their approach on two datasets: a synthetic bilingual TinyStories case wherein the English version is the retain set and the Spanish version is the forget set, and Wikipedia, wherein the biology knowledge is the forget set. Empirical results show that SGTM consistently outperform gradient routing on both setups, and has the potential to close the gap with data filtering. Additionally, models trained with SGTM are more robust to adversarial finetuning.

### Strengths
- The paper is very well-written. I must admit that I'm not an expert in pretraining approaches to unlearning, but the authors have demonstrated expertise in their exposition, overview, and contributions. The literature review also gives a concrete sense of SOTA approaches.
- The proposed SGTM approach is simple and intuitive, making them suitable for large-scale training runs for frontier models. 
- The experiments are well-designed, with a more controlled, stylized setting of bilingual TinyStories as a proof of concept, and a more realistic Wikipedia pretraining as validation. 
- The authors demonstrate that SGTM is more robust to unlabeled forget data Figure 3(b), albeit at the cost of higher retain loss on identically sized models. One can argue that the effective parameter count is lower due to the masking schemes, and the authors conduct targeted scaling law analysis in the appendix to investigate this.
- It is somewhat expected (but interesting!) that SGTM models are more robust to adversarial finetuning

### Weaknesses
- Scale remains a primary concern, as the largest models studied in this paper is 254M. While I don't deduct points from it due to the expense of pretraining, whether this approach can be scaled to frontier systems is an open question, and whether the compute penalty (6% for general knowledge according to the authors) is a worthy trade-off remains to be studied.

### Questions
- I really like the masking idea, but I'm curious if the authors should explore masking in the output space, i.e. by masking out the loss of undesirable tokens so that the model can have a good understanding of the surrounding context, but at inference time they will not generate these undesirable knowledge.

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
3

### Summary
This paper proposes a method for unlearning in a LLM at the pretraining stage, by masking gradients to a parameter set on the backward pass if an example is part of the forget set. They argue this is more robust to label noise than data filtering or the similar Gradient Routing. They support this with empirical evidence.

### Strengths
- clearly written, seems novel - caveat that I am not deeply familiar with the unlearning literature
- experiments seem to support the basic point of improvement the authors suggest for SGTM, and are fairly thorough (ablations with related data categories are cool)
- Fig 1 is great! In general the communication around tradeoffs is well done

### Weaknesses
- it's odd to me that there aren't results shown for Fig 4 for GR - isn't this the main baseline we should be comparing to?
- some contradictory statements around parameter subsets: in Fig 2 caption the authors that the after forget parameters are assigned, “the remaining parameters are designated to the retain data” but then discuss something called "joint" parameters in line 183
- it would be good to give more intuition here - why is SGTM more robust to label noise? it's not a priori obvious to me that it should be, some exploration about the difference to baselines would be helpful
- I'm not sure exactly how this is usually handled in unlearning, but there doesn't seem to be a lot of information about how the SGTM model performs without the parameters masked
- as someone not deeply familiar with the literature, it's not quite clear to me if the only difference between SGTM and GR is activation vs. parameter gradient masking? would be good to state this more clearly
- Leakage: defining this as a percentage is odd to me - it's misleading that the number is constant between 5% and 50% tokens, since that's the equivalent of 4x tokens, which is a lot! I think something that scales with the token equivalence number (eg 707k in Fig 5a) is more sensible

### Questions
- how does GR perform on the real data in Fig 4?
- why does SGTM display more robustness to label noise?
- clarify: main difference between GR and SGTM is activation vs parameter gradient masking?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Undesirable data is often filtered out during pre-training, but filters have false negatives, so some undesirable content remains in the pre-training data. This paper proposes a training technique that encourages such undesirable data to reside in a small subset of parameters that can then be removed.

### Strengths
Empirical results on multiple settings show an improved trade-off between general capabilities and forgetting of undesirable content, compared with filtering. Moreover, it has much better performance against fine-tuning compared with a strong unlearning method, RMU.

The method is quite simple and intuitive. Gradient masking sequesters undesirable knowledge into a small subset of parameters, while parameter masking encourages the rest of the parameters to function well even when those parameters are removed.

The paper is well-written and clear.

### Weaknesses
This paper compares only with filtering and a similar previous work (Gradient Routing), but other methods have also been developed as alternatives to filtering:
* https://arxiv.org/abs/2302.08582 This paper explores several training objectives and finds that a "conditional training" approach works well. It seems that SGTM could directly compare with this approach.
* https://arxiv.org/abs/2505.03052 This paper has a somewhat different motivation, but they can use a more aggressive threshold on the classifier because their intervention is less strict than filtering. It also seems worth discussing and possibly comparing with

$\theta_{\text{retain}}$ is used but is not clearly defined (e.g. Line 182). It's also confusing that the retain parameters are not mentioned in Lines 250-253.

The experimental settings are somewhat toy. For TinyStories, 64M is a very small model. It is also quite synthetic to generate the Spanish data with translation, when multilingual corpora also exist. The noise model is also not realistic, as it is pure iid noise. Finally, the motivation of this experiment is a bit unclear, since in practice one would not want to prohibit the model from learning a second language. The Wikipedia experiments are more realistic in noise, though 254M is still quite a small model.

The experiments are also somewhat narrow. Only these two model sizes are considered, and for Wikipedia only one possible forget set is considered. Perhaps toxic text could be considered as another type of data that is typically filtered but only imperfectly.

Finally, the paper does not clearly explain the methodological difference with the previous version of Gradient Routing from Cloud et al. (2024). This is important for explaining the novelty of this method.

### Questions
Could you explain more what $\theta_{\text{retain}}$ is, how it differs from $\theta_{\text{joint}}$, and how it is used?

What are the main differences between SGTM and Cloud et al.? From looking at that paper, it seems that the difference may be from the selective parameter masking, but I am curious if this is correct and if there are other differences.

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
The paper introduces selective gradient masking, a pretraining time technique to localize and remove specific capabilities from LLMs. The authors evaluates the method on two settings of (1). synthetic bilingual data and (2). wikipedia corpus. Across both, SGTM achieves a better retain/forget tradeoff under label noise compared to the baselines. The authors also show SGTM is more robust to adversarial fine-tuning.

### Strengths
1. **Adversarial robustness**: the detailed discussions on mislabeled content and adversarial fine-tuning are valuable and highly relevant to the community. 
2. **Clear presentations**: the figures and visualizations are informative and well-designed.

### Weaknesses
1. **Insufficient Evidence**: This is my primary concern. The evaluation relies solely on model loss, which may not adequately capture downstream perfromance differences that truly matter. It is unclear to me whether higher loss indeed indicates better forgetting. Including additional evaluations for forgetting and general performance retention would substantially strengthen the paper's empirical support. 

2. **Limited Scale**: As noted in section 6, the experiments use very small model and dataset sizes. It remains uncertain whether the findings would generalize to larger models of real-world training scales.

### Questions
1. How were the data points in Figure 1 obtained? Do they represent Pareto frontiers or averages? 

2. What is $\theta_{joint}$ specifically? How are the $\theta_{joint}$ parameters selected, and how are they different from $\theta_{retain}$ or $\theta_{forget}$?

### Soundness
3

### Presentation
3

### Contribution
2
