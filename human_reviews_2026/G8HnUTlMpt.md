# Harnessing Hyperbolic Geometry for Harmful Prompt Detection and Sanitization

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Vision–Language Models (VLMs) have become essential for tasks such as image synthesis, captioning, and retrieval by aligning textual and visual information in a shared embedding space. Yet, this flexibility also makes them vulnerable to malicious prompts designed to produce unsafe content, raising critical safety concerns. Existing defenses either rely on blacklist filters, which are easily circumvented, or on heavy classifier-based systems, both of which are costly and fragile under embedding-level attacks.
We address these challenges with two complementary components: Hyperbolic Prompt Espial (HyPE) and Hyperbolic Prompt Sanitization (HyPS). HyPE is a lightweight anomaly detector that leverages the structured geometry of hyperbolic space to model benign prompts and detect harmful ones as outliers. HyPS builds on this detection by applying explainable attribution methods to identify and selectively modify harmful words, neutralizing unsafe intent while preserving the original semantics of user prompts.
Through extensive experiments across multiple datasets and adversarial scenarios, we prove that our framework consistently outperforms prior defenses in both detection accuracy and robustness. Together, HyPE and HyPS offer an efficient, interpretable, and resilient approach to safeguarding VLMs against malicious prompt misuse.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces the HyPE and HyPS, a system for malicious prompt detection and sanitization based on constructing prompt embeddings in a hyperbolic space that naturally separates malicious from benign prompts. Sanitization is performed by selectively replacing harmful words in the prompt using an explainable attribution method. The approach is benchmarked against a range of SOTA approaches and stress-tested against some adversarial attacks.

### Strengths
* Notwithstanding my questions below, the hyperbolic approach is well-suited to single-class training, reducing a dependency on labeled malicious examples.
* The method includes a sanitization step, enabling end-to-end workflows where refusal is undesirable.
* Empirical experimental results are relatively thorough and demonstrate strong performance of the method compared to several baselines.

### Weaknesses
* Section 3.1 appears to leave out important details of how the mapping from token sequence to hyperbolic space is learned, instead focusing on how R is tuned. It appears that the experiments leverage a pretrained encoder for this step (HySAC).
* A different encoder is selected for HyPS and this should be justified.

### Questions
Explain in more detail the reasoning behind the choice of a hyperbolic space. You say it "naturally disentangles  hierarchical and compositional relations, making it well-suited for modeling data with latent hierarchical structure". What is the latent hierarchical structure in text-to-image prompts? Is the expectation that the malicious/benign label represents the top level of the hierarchy?  

The underlying assumption appears to be that malicious prompts will be outliers in this learned space- have you stress-tested your approach using OOD benign images?

Minor notation issue in Eq 1: if x is n+1 dimensional then the Lorentzian inner product should only sum up to n (the elements of x are indexed 0 to n, comprising n+1 elements).

For HyPS: How do you make the decision about what words to replace (is it a hard threshold or more adaptive method)?

Suggest blurring the offensive image in Fig 1- readers will trust your claim that it is offensive.

Detection is trained using HySAC embeddings but attribution appears to be applied using CLIP embeddings and I'm curious to better understand the implications. Aside from empirical validation, can you support the claim that the two embedding spaces can be distinct? Is it difficult to implement LIG against the HySAC encoder?

Presentation: Table 1 misrepresents HyPE as the winning method in the Precision column for adv-MMA (should be GuardT2I). There are some ties that should also be marked- eg SneakyPrompt recall and COCO accuracy.  Ideally you would also indicate statistically significant wins. 

For HyPS experiments, do you also validate that the rewritten prompts fall inside R?

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
2

### Summary
This paper proposes HyPE (Hyperbolic Prompt Espial) and HyPS (Hyperbolic Prompt Sanitization) to detect and mitigate harmful prompts for VLMs. HyPE models benign prompt embeddings with a hyperbolic one-class SVDD objective, leveraging hyperbolic geometry to better capture hierarchical semantic structures. HyPS then uses token-level attribution to locate malicious prompt components and sanitizes them while preserving intended meaning. Experiments on harmful prompt benchmarks show improved harmful-prompt detection and higher safety-preserving sanitization quality compared to Euclidean baselines and existing safety filters. Results cover multiple VLMs and attack settings.

### Strengths
1. Novel use of hyperbolic geometry for prompt safety detection; motivation around hierarchical semantics is clearly presented.
2. Covers both detection and sanitization, producing a more practical defense pipeline rather than only binary classification.
3. Experiments across multiple VLMs and datasets show consistent gains over Euclidean one-class and prompt-filtering baselines.

### Weaknesses
1. Evaluation focuses mainly on text-based harmful prompts; applying the method to multimodal triggers or image triggers would strengthen generality claims.
2. The robustness against paraphrased, or style-trigger [1] is unclear. Since harmful intent can be distributed across multiple tokens or advanced style rephrase, it is uncertain whether the method can consistently detect or sanitize such stealthy, paraphrased attacks. Evaluating these attack strategies would strengthen the claim of general robustness.

Ref:
[1] Mind the Style of Text! Adversarial and Backdoor Attacks Based on Text Style Transfer

### Questions
1. Can HyPE generalize to vision-triggered or multimodal jailbreak scenarios, beyond text-only attacks?
2. What is the runtime/memory overhead of hyperbolic SVDD compared to Euclidean baselines?
3. How robust is the method to paraphrased or iterative jailbreak prompts where harmful intent is spread across multiple tokens?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes HyPE (Hyperbolic Prompt Espial) and HyPS (Hyperbolic Prompt Sanitization), two complementary modules for detecting and sanitizing harmful prompts in Vision–Language Models (VLMs). HyPE employs a hyperbolic SVDD anomaly detector trained solely on benign prompts to identify harmful inputs as outliers in Lorentz space, while HyPS applies explainable attribution (Layer Integrated Gradients) to localize harmful words and neutralize them through replacement or removal, preserving semantics. Experiments across six datasets, multiple adversarial attacks, and two downstream tasks (text-to-image generation and image retrieval) demonstrate superior detection accuracy, interpretability, and robustness over prior methods .

### Strengths
- The paper extends classical SVDD to hyperbolic space (Eq. 2) using the Lorentz distance and curvature parameter K. This geometric formulation enhances interpretability and robustness, which are often lacking in prior NSFW classifiers.
- The one-class setup removes the need for harmful data training, improving safety and scalability, an important property since unsafe content may not always be known or accessible to defenders.
- The method is evaluated on six datasets spanning paired and single-class prompts as well as adversarial attack settings. Results show consistent improvements over multiple baselines, demonstrating strong practical effectiveness.

### Weaknesses
- The proposed approach appears to be a direct adaptation of SVDD into Lorentz space, with limited new theoretical insights. For instance, the Eq. (2) largely represents a straightforward geometric reformulation rather than a fundamentally new concept.
- HyPE relies on embeddings from HySAC, which may introduce bias and limit the generality of reported performance.
- All evaluated datasets focus on English NSFW and violent content, with no assessment of multilingual or socio-cultural harmful expressions, potentially constraining generalization.

### Questions
Could the authors provide additional experiments or discussion on multilingual settings and other categories of safety violations to support the broader applicability of their framework?

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
This paper proposes a hyperbolic geometry–based framework to detect and neutralize harmful prompts in VLMs. It introduces two modules: HyPE, a hyperbolic anomaly detector that models benign prompts in hyperbolic space and flags harmful ones as outliers, and HyPS, a post-hoc sanitization system that identifies harmful words and replaces or modifies them using lookups or LLM-based substitution. Experiments across six datasets and two adversarial attacks show that HyPE outperforms prior detectors in accuracy and robustness, while HyPS effectively sanitizes malicious prompts without distorting their meaning.

### Strengths
1. HyPE demonstrates the SOTA detection performance on six datasets and two adversarial scenarios
2. HyPS performs well on two downstream tasks

### Weaknesses
Main concern
1. The contribution is relatively incremental. This paper mainly applies hyperbolic models to harmful prompt detection in VLMs. Apart from the empirical gains shown in the experiments, more analysis of why we need to use the hyperbolic models in this task is needed (e.g., some theoretical insights or some empirical comparison between the SVDD and hyperbolic SVDD, etc).

Other concerns
1. The authors do not consider the adaptive attack scenario, in which the attackers have white-box access to both the detectors and the text encoder.
2. The authors only evaluate the effectiveness of one text encoder, HySAC. More evaluations on other encoders would show the robustness of this approach.

### Questions
1. If the classifier were not based on a hyperbolic model, would HePS still function effectively? Does this module have to depend on hyperbolic modeling?

### Soundness
3

### Presentation
3

### Contribution
2
