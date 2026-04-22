# Bayesian Primitive Distributing for Compositional Zero-shot Learning

- Avg Score: 6.00
- Decision: Reject
- Scores: 4, 6, 10, 4

## Abstract
Compositional zero-shot learning (CZSL) aims to recognize unseen attribute-object combinations by learning primitive concepts (i.e., attribute and object) from seen compositions. Existing CZSL solutions typically harness the power of vision-language models like CLIP via textual prompt tuning and visual adapters. However, they independently learn one deterministic textual prompt for each primitive or compositional labels, ignoring both the inherent semantic diversity within each primitive and the semantic relationships between primitive concepts and their compositions. In this paper, we propose BAYECZSL, a novel Bayesian-induced framework that learns probability distributions over each primitive textual prompt from a Bayesian perspective. Specifically, BAYECZSL models image-specific primitive textual prompts as learnable probability distributions to capture intra-primitive diversity. Building on these primitive distributions, we aggregate learned probability distributions from attribute and object branches to form compositional prompt space via Compositional Distribution Synthesis strategy, thus capturing the semantic relationships between primitive concepts and their compositions. Moreover, Three-path Distribution Enhancement module is introduced to transform initial distributions into expressive ones via invertible mappings.
Finally, these enhanced distributions are sampled to generate diverse textual prompts, achieving more comprehensive coverage of the prompt space and generalizing to unseen compositions. Extensive experiments on multiple CZSL benchmarks demonstrate the superiority
of our BAYECZSL. Code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces BAYECZSL, a novel Bayesian-induced framework that learns distributions over primitive textual prompts. The authors observe that the existing CZSL works use a single deterministic textual prompt for each primitive concept and its composition, which is insufficient to capture variations within the compositions; for example, old in old dog is different from old town. Furthermore, they notice that prior work ignores the rich relational structure between the primitives and compositions. BAYECZSL addresses these limitations through the following steps. First, it learns a probability distribution over primitive concepts to better represent intra-primitive diversity and reduce overfitting. Then, it uses compositional distributional synthesis to aggregate the learned probability distributions into a unified compositional prompt space. Then, to model more complex distributions, it uses a three-path distribution enhancement module to transform the initial prompt and composition distributions into flexible distributions using a sequence of invertible mappings. Finally, it draws multiple Monte Carlo samples from the distributions and mixes them with the original prompt representations to improve generalization. The results on the CZSL benchmarks show that BAYECZSL improves performance over the state-of-the-art methods.

### Strengths
The paper is easy to follow and well-written.

The proposed method is well-motivated. The method outperforms prior work on compositional zero-shot learning datasets.

The ablations in Section 4.3 are quite helpful in understanding the contributions of BAYECZSL.

### Weaknesses
**Method**

- The core components (variational posteriors, Gaussian fusion, normalizing-flow enhancement, and Monte Carlo sampling) are established Bayesian/VI tools. Novelty is rather modest since it mainly adapts these techniques to CZSL prompt distributions.

- The approach uses MLP-based disentanglers to obtain attribute and object features and to parameterize their base posteriors. The compositional posterior is obtained via inverse-variance weighted fusion rather than being learned directly. A simpler alternative is to get a composition posterior from $x^{c}$ (Eq. 2) without CDS and additional disentanglers; an ablation here would be helpful.

- The method is close to CoCoOp [a]. CoCoOp adds additional information about the image, in the form of a meta token, to the text prompts, thereby improving performance. Although this paper is included in the related work, it is not compared in the results section. Including them as baselines or explaining why they are not directly comparable would make the evaluation fairer.

- The framework assumes single attribute-object compositions and does not evaluate multi-attribute or multi-object cases. Suppose there is an object with multiple attributes at test time (e.g., small white cat), the framework could potentially sample vectors from the same enhanced distributions for multiple attributes rather than treating them as separate attributes. This could lead to a performance drop when integrated into the prompt vectors. Including experiments on attribute-attribute-object settings [b] would strengthen the paper.

**Architecture.**

All the experiments are limited to the CLIP ViT-L/14 model. It would be great if the authors could include experiments with additional CLIP models and other models such as BLIP, etc. It is also unclear from the paper if their method will even work with vision-language models that use a decoder instead of the bi-encoder architecture seen in CLIP (Figure 2).

**Impact**

While the paper shows positive, albeit small, improvements over prior methods on the compositional zero-shot learning datasets, my concern is that the paper is too task-specific.
The paper makes strong assumptions about the types of compositions it can handle, i.e., the method can only handle attribute-object compositions. This severely limits the impact of the paper.

**Minor suggestions**

Lines 97-99: It would be great if the authors could explicitly say that they are reporting relative improvement in performance. At first glance, it appeared to be an absolute improvement.

**References**

[a] Conditional Prompt Learning for Vision-Language Models, CVPR 22.

[b] Learning to Compose Soft Prompts for Compositional Zero-Shot Learning, ICLR 23.

### Questions
In addition to the questions listed in the weaknesses section, here are a few more questions. 

**Clarification for Misc. claim**

- Lines 61-63: What does “cross-branch synergies” mean? Could you explain that in simpler words? 

- In lines 60-61, the authors say that prior work ignores the rich relational structure between the primitives and their compositions. Could you clarify what this sentence means and how BAYECZSL understands the relational structure of the concepts? In addition, could you also discuss Appendix D in more detail in the main paper? The plots suggest the model's performance can drop below the best numbers reported in Table 1. I’d like to understand the trade-off between the number of Monte Carlo samples and performance.

**Error bars**

Since you are averaging over $L$ prompts, could you also include the error bars for the method in the results section?

### Soundness
2

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
4

### Summary
The paper proposes BAYECZSL, a Bayesian-induced framework for compositional zero-shot learning (CZSL) that represents each primitive textual prompt (attribute/object) as a probability distribution, rather than a single deterministic vector like most prior work in recent years. WIthin the proposed approach, these distributions (of attributes and objects) are image-conditioned via variational inference, then synthesized into a compositional prompt distribution using variance-inverse Gaussian fusion; distributions are further made expressive by a three-path distribution enhancement module based on invertible mappings (normalizing-flow–style). Sampling from these distributions yields diverse prompts that are mixed with the soft prompts in a three-path CLIP-based architecture (attribute/object/composition). The loss combines branch cross-entropies with a Bayesian regularizer, and inference fuses composition scores with the product of primitive scores. The authors conduct experiments on standard CZSL benchmarks MIT-States, UT-Zappos, and C-GQA show state-of-the-art results in closed-world and open-world settings. The authors also conduct ablations on the three modules and sampling sensitivity.

### Strengths
To the best of my knowledge, the paper’s core idea—modeling attribute/object primitive prompts as image-conditioned distributions, composing them into a compositional distribution via variance-inverse Gaussian fusion is novel for CZSL. 

I think this probabilistic framing is clear and technically sound: the objectives are explicit, the inference story is coherent, and the components (BPD, CDS, TDE) are well-motivated. I think the evaluation setup in this paper is correct and consistent with the CZSL literature (closed-world and open-world settings on MIT-States, UT-Zappos, and C-GQA), and the approach shows strong performance overall, with especially robust gains on UT-Zappos. I like that the ablations are clean and isolate each module’s contribution, and the sensitivity analyses over the sampling count L and the mapping depth N are informative rather than cosmetic. 

Overall, I think this paper is a good work on compositionality in CZSL with a principled probabilistic formulation and solid empirical support.

### Weaknesses
While I have no concerns about the novelty or the methodological soundness of this work, my primary concern is reproducibility. The reported numbers appear to be single-run point estimates without variability, and I do not see evidence of repeated runs or variance statistics. Including results over multiple random seeds (idealy over 5) with mean +/- confidence intervals would materially strengthen the quantitative claims, especially for the main comparisons and ablations. Clearly specifying the random seed policy, sources of stochasticity (e.g., initialization of the flow/TDE, sampling count L, data shuffling), and any early-stopping criteria would also help others reproduce the results faithfully.

### Questions
I am wondering if the authors have thought about how sensitive are results to the diagonal-Gaussian residual assumption in BPD? Have the authors tried things like full-covariance, mixture posteriors, or normalizing flows in place of TDE to shift complexity upstream?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This paper proposes BAYECZSL, a Bayesian-induced CZSL framework that learns probability distributions over each primitive textual prompt from a Bayesian perspective. The method explicitly models prompt uncertainty for attributes and objects, then synthesizes compositional distributions through a principled fusion mechanism, and enhances those distributions with invertible mappings. Experiments on three major CZSL benchmarks (MIT-States, UT-Zappos, C-GQA) demonstrate BAYECZSL outperforms existing CZSL methods in both Closed-World and Open-World settings.

### Strengths
1) This paper proposes very novel ideas to more effectively tackle the core challenge of intra-primitive semantic diversity in compositional zero-shot learning via Bayesian distribution modeling.
2) The key idea of learning probability distributions over each primitive textual prompt, rather than learning a single deterministic prompt as in prior work, is both theoretically grounded and interesting. In addition, this idea is well-aligned with the core challenge.
3) I am also generally impressed  by the the novel use of primitive distributions to construct a compositional prompt space, and the practical use of distribution enhancement strategy to facilitate diverse prompt sampling.
4) The experimental results are convincing, naturally leading the reader to concur with the authors’ perspective. 
5) The paper is exceptionally well-written and a true pleasure to read.

### Weaknesses
1) Its better to explain why the variance-inverse weight Gaussian fusion strategy is used in the Compositional Distribution Synthesis module. 
2) More extensive ablation experiments on more datasets, such as MIT-States or C-GQA, would improve the experiment part.
3) Its better to analyze the impact of the hyper-parameters $\beta_a,\beta_o,\beta_c$.
4) The model introduces computational overhead compared to single-prompt and simple soft-prompt baselines, given multiple sampling, flows, and synthesizing steps. Its better to analyze training/inference cost, memory consumption, or tradeoffs between performance and complexity.

### Questions
How is numerical stability maintained during covariance inversion in the compositional distribution synthesis step? Is regularization ever needed, and does this affect fusion quality?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes BAYECZSL, a Bayesian-induced framework for Compositional Zero-Shot Learning that models attribute and
object prompts as probability distributions rather than deterministic embeddings. The method captures intra-primitive diversity
and semantic uncertainty by learning Bayesian distributions over textual prompts, which are then fused through a Compositional
Distribution Synthesis module to form a compositional prompt space. A Three-path Distribution Enhancement module further
refines these distributions via invertible mappings, enabling more expressive sampling and richer semantic coverage.

### Strengths
1. The paper reformulates CZSL from a Bayesian inference standpoint, introducing the idea of learning probability distributions over primitive textual prompts. This probabilistic view allows the model to explicitly model intraprimitive variability and semantic uncertainty, addressing a key limitation of prior deterministic prompt-based methods.

2. The proposed CDS and TDE modules jointly enable a unified, expressive prompt distribution space. CDS fuses attribute and object distributions to model their semantic relationships, while TDE transforms base distributions into more flexible ones via invertible mappings.

### Weaknesses
1. Although the combination of Bayesian modeling and compositional synthesis is interesting, several prior works have explored distributional or probabilistic prompt spaces. E.g. "Prompt Distribution Learning" and "Prompting Language-Informed Distribution for Compositional Zero-Shot Learning". The contribution may thus be perceived as an evolutionary extension rather than a fundamentally new paradigm.

2. The baselines used for comparison are outdated, primarily consisting of works from 2022 and 2023. It would strengthen the paper to include evaluations against more recent state-of-the-art approaches.

3. Despite being compared with relatively outdated baselines, the reported improvements are not particularly significant.

### Questions
How many prompt tokens are used ?

### Soundness
3

### Presentation
3

### Contribution
2
