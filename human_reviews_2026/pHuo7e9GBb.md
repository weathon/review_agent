# Exposing Hidden Biases in Text-to-Image Models via Automated Prompt Search

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Text-to-image (TTI) diffusion models have achieved remarkable visual quality, yet they have been repeatedly shown to exhibit social biases across sensitive attributes such as gender, race and age. To mitigate these biases, existing approaches frequently depend on curated prompt datasets - either manually constructed or generated with large language models (LLMs) - as part of their training and/or
evaluation procedures. Beside the curation cost, this also risks overlooking unanticipated, less obvious prompts that trigger biased generation, even in models that have undergone debiasing. In this work, we introduce Bias-Guided Prompt Search (BGPS), a framework that automatically generates prompts that aim to maximize the presence of biases in the resulting images. BGPS comprises two components: (1) an LLM instructed to produce attribute-neutral prompts and (2) attribute classifiers acting on the TTI’s internal representations that steer the decoding process of the LLM toward regions of the prompt space that amplify the image attributes
of interest. We conduct extensive experiments on Stable Diffusion 1.5 and a state-of-the-art debiased model and discover an array of subtle and previously undocumented biases that severely deteriorate fairness metrics. Crucially, the discovered prompts are interpretable, i.e they may be entered by a typical user, quantitatively improving the perplexity metric compared to a prominent hard prompt optimization counterpart. Our findings uncover TTI vulnerabilities, while BGPS expands the bias search space and can act as a new evaluation tool for bias mitigation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the issue of social biases in TTI models. It point out that existing bias detection methods are either limited by the scope of manual test sets or generate uninterpretable adversarial text. To address this, the paper proposes an automated framework named Bias-Guided Prompt Search (BGPS). This framework combines an LLM with attribute classifiers that operate on the internal activations of the TTI model. The core mechanism involves using the bias signals from the classifiers to steer the LLM's decoding process, thereby automatically discovering prompts that maximize bias exposure while maintaining natural language fluency. The authors conducted experiments on Stable Diffusion 1.5 and a debiased model. The results show that BGPS can find hidden, undocumented biases and reveal that significant biases persist even in models that have undergone debiasing efforts.

### Strengths
**1. Well-motivated problem and a practical objective**

 The objective of developing an automated framework to audit TTI models for fairness and safety is of high practical importance for the responsible deployment of generative AI.

**2. Generation of interpretable, human-readable prompts**

A primary strength of the proposed method is its ability to generate human-readable and interpretable prompts. The paper correctly identifies the limitations of gradient-based optimization methods, which often produce nonsensical and un-interpretable text. By leveraging a large language model, the BGPS framework successfully generates fluent, natural language prompts.

### Weaknesses
**1. Insufficient Experimental Scope and Lack of Generalizability**

The paper's claims of providing a generalizable framework are unsubstantiated due to a severely limited experimental scope. The evaluation is confined entirely to a single, model, Stable Diffusion 1.5. The failure to test the method on any other diverse, modern (SDXL, Flux), or transformer-based (DALL-E 3, SD3) models means it is impossible to assess whether the framework is truly general-purpose.

**2. Use of Outdated and Unspecified Models**

The core components used in the experiments are outdated and poorly specified, making the work feel disconnected from the current models in 2025. Beyond the use of the old SD 1.5 model, the paper relies on an aging LLM (Mistral-7B) and, critically, fails to specify which version was used. This omission is a significant flaw that severely harms the reproducibility of the research.

**3. Minimal Method Novelty**

The paper's methodological contribution is largely incremental. The proposed framework is a straightforward application of existing principles in guided decoding, with basic prompt engineering for LLMs. As such, the work's novelty from a methodological standpoint is limited.

**4. No New Scientific Insight**

While the method is effective at finding biases, the experiments primarily succeed in rediscovering well-documented and highly intuitive social biases and stereotypes that have been long established in both sociology and prior AI fairness literature. The paper offers little to no new scientific insight into the nature of bias, serving more as a confirmation of known issues.

**5. Poor Presentation Quality of Figures and Captions**

Figures 2 and 3, which are crucial for illustrating the method's outputs, are hindered by poorly formatted captions. The prompt texts are broken across lines in unnatural and confusing ways, which severely impairs readability and forces the reader to decipher the intended meaning. This detracts from the paper's professionalism and weakens the impact of its results.

### Questions
**1.** Given that the current experiments are limited to outdated models, can the authors provide any new results on contemporary SOTA models to substantiate the framework's claimed generalizability?

**2.** Beyond confirming well-documented social biases, did the authors' method uncover any counter-intuitive or genuinely novel insights into the nature of bias in TTI models?

**3.** I strongly encourage the authors to commit to reformatting Figures 2 and 3, as their current layout severely hinders the readability of the paper's core qualitative evidence.

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
5

### Summary
This paper automatically discovers interpretable prompts that maximize bias exposure while remaining natural. It optimizes a joint objective involving LLMS for naturalness and an attribute classifier on the middle block representations of the Stable Diffusion Model. They utilise a beam search algorithm to search through the prompts that maximize the joint objective. All the experiments are performed on SD1.5. Auditing is also performed for one of the existing debiasing methods.

### Strengths
This paper tackles the problem of exposing biases in T2I models using realistic, neutral-sounding prompts, which is an important problem that helps in auditing large-scale T2I models.

### Weaknesses
1. Results are reported primarily on SD-1.5. Since the method depends on UNet architectures, the evaluation should be extended to SD-2.1 and SDXL to assess robustness on newer, stronger models.
2. Given the paper’s focus on residual bias in “debiased” models, the audit should also include recent debiasing methods, e.g., ITIGen, which operates in the prompt space, to strengthen the generality of the claims.
3. The paper uses simple, clean, single-person prompts. In order to test the applicability to real-world settings, it is important to evaluate real-world prompts such as multi-person scenes (define how gender is assigned) or long/ambiguous prompts.
 4. Because an LLM is used as the prompt prior, its own biases can seep into the pipeline. How sensitive is the approach to LLMs used? It is important to include sensitivity to different LLM priors and an ablated/weaker prior to measure this effect.

### Questions
1. Can you provide more qualitative results and corresponding prompts for different races? 
2. What happens when the attributes are very complex in this setup? How much does the performance of the attribute classifier affect the whole pipeline?

### Soundness
2

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
This paper introduces Bias-Guided Prompt Search (BGPS), an automated method that uses LLM and attribute classifiers to discover text prompts for exposing hidden social biases (e.g., gender, race) in text-to-image models. The proposed method effectively uncovers subtle, context-driven biases that standard evaluations miss and produces more natural prompts than gradient-based alternatives.

### Strengths
1. The proposed method can discover subtle and previously undocumented biases, which expands the evaluation space beyond curated datasets.

2. Compared with gradient-based methods, the proposed method can generate more natural text.

### Weaknesses
1. The scope of biased attributes evaluated (gender, race) is limited by the classifiers used, though the method is generalizable.

2. Although the generated prompts look more natural than the prompts generated by gradient-based methods, as shown in figure 1, the generated prompts are still not very natural (not like common prompts written by human)

3. The technical novelty is a little limited. There is neither strong technical insight or theoretical analysis.

### Questions
Please refer to the weakness part.

### Soundness
2

### Presentation
2

### Contribution
2
