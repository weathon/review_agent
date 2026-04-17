# Emotions Where Art Thou: Understanding and Characterizing the Emotional Latent Space of Large Language Models

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
This work investigates how large language models (LLMs) internally represent emotion by analyzing the geometry of their hidden-state space. Using a synthetic dataset of emotionally rewritten sentences, we identify a low-dimensional emotional manifold via singular value decomposition and show that emotional representations are directionally encoded, distributed across layers, and aligned with interpretable dimensions. These structures are stable across depth and generalize to eight real-world emotion datasets spanning five languages. Cross-domain alignment yields low error and strong linear probe performance, indicating a universal emotional subspace. Within this space, internal emotion perception can be steered while preserving semantics using a learned intervention module, with especially strong control for basic emotions across languages. These findings reveal a consistent and manipulable affective geometry in LLMs and offer insight into how they internalize and process emotion.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors analyze consistency of the underlying geometry of emotions in LLMs. They present several studies to support their claims: similarity metrics of synthetic to real samples, dimensionality reduction of the space to discover lower-dimensional structure in a principled manner, and steering representation to study whether the space behaves predictably.

### Strengths
- The authors present many metrics and models, as well as some useful baselines in the appendix to contextualize the numbers
- Many studies are presented to corroborate the claims of the paper instead of building a case from only a single case.
- The authors use existing theoretical work in emotion to ground their findings.

### Weaknesses
- Manuscript is very difficult to read, with methodology blended with experiments, making it very difficult to follow what is currently being presented. For example, what is Table 2 showing (L210)? Same with L230, and all of the subsequent sections (5 and 6). I found myself jumping back and forth constantly, trying to understand what is being compared to what and how. Moreover, some numbers seem to be presented in the text only. I would appreciate clearer methodology in the rebuttal, as some concepts are also not explained at all.
- Tables 1 and 3 contain so much information that it becomes very difficult to figure out what the takeaway from each should be. A better mode of presentation would be preferable, and the full tables can be in the appendix for the interested reader.
- Figure 2 could be improved, perhaps by showing the projection to the Dominance-Valence plane. As it is now, sad and happy seem to have the same embedding (meaning the same valance value, among the other dimensions). As a result, the study this corresponds to I believe would benefit from some quantification to substantiate the claim.
- The fact that from layer 0 to layer 31, the % of neurons remains the same might indicate that the clustering is happening because of word embeddings themselves rather than emotion content per se. This is because we would expect higher-level emotional content to emerge after processing in later layers, except if it based on individual words.

To reiterate, the main weakness of the paper is its lack of clarity, not necessarily a lack of substance or novelty.

### Questions
- Formatting errors: the authors have used the wrong citation format (every citation is used outside of parenthesis, probably showing some lack of care when switching between templates)

### Soundness
3

### Presentation
1

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
The paper presents an investigation on how LLMs represent emotions in their latent spaces. It finds generalizable evidence that emotion is confined to a few critical dimensions that can be interpreted and manipulated. The authors then present a new method for steering controls to improve emotion classification. The authors robustly back up these central claims through several methods and across multiple languages and datasets. I particularly appreciate the multi-lingual analysis to demonstrate that these findings are generalizable. While paper itself provides some interesting and novel findings, but the presentation is somewhat unclear and messy.

### Strengths
This paper presents the first investigation on emotional latent space representations in LLMs, and I believe the techniques used are novel and interesting.
The authors provide analysis on many different perspectives, styles, and languages, which adds to the robustness of their findings.
The new steering method introduced presents a new way to consider how to change emotions: by focusing on changing the underlying emotional subspace rather than focusing on downstream output.

### Weaknesses
It is unclear exactly what Table 2 is measuring, specifically in regards to cosine similarity and MSE. The paper states high cosine similarity between emotions in real datasets and their synthetic counterparts: what synthetic counterpart are we referring to? Does this reference the Reichman et al. synthetic dataset? Table 2 does not mention which method it utilized as well. What does it mean to measure the cosine similarity of an emotion between two datasets, what datum from each dataset is actually being passed in? Same to MSE and the other metrics. I would recommend rewriting this section to be clearer, as in its current state I was unable to understand exactly what is going on.

On that note, it is not clear exactly what the experimental setup in 4.2 is. I would appreciate some more clarity on the exact methods used, number of experiments, etc. In particular, I feel like the distortion metrics should be clarified on what they actually represent, as this information is not present in the paper and should not assumed to be common knowledge.

The presentation of the paper is not very clear; the sections seem rather disjoint and clarifying figures are somewhat lacking throughout. While I believe the content is novel and unique, a rewrite for clarity would improve this paper significantly.

### Questions
Does the Space Alignment method rely on the same Reichman et al. dataset that Centered-SVD does? If so, please state that explicitly.
 Additionally, it would seem to me that the performance of Centered-SVD and/or Space Alignment would highly depend on the quality of this underlying training dataset. I would appreciate some clarity quantifying the quality of this Reichman dataset (or justifying why it was chosen) as this choice seems to be integral to both of these methods.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies how large language models (LLMs) internally represent emotion by analyzing the geometry of their hidden states. Using models like LLaMA 3.1, Olmo-v2, and Ministral across eight multilingual datasets, the authors find a low-dimensional emotional manifold that aligns with psychological dimensions such as valence, dominance, and arousal. These representations are directional, distributed, and consistent across layers and languages. Through SVD and neuron-level analysis (ML-AURA), the paper shows that emotional axes are stable and interpretable. A learned steering module further demonstrates that these internal states can be manipulated predictably without altering semantics, revealing a coherent and controllable affective geometry in LLMs.

### Strengths
- The paper presents a comprehensive, cross-lingual study covering eight datasets in five languages, offering strong evidence for the universality of LLM emotion representations.
- The use of ML-AURA and SVD-based analyses provides a rigorous and interpretable framework for linking internal neuron activity to affective semantics.
- The learned steering module demonstrates practical control of emotion representations while preserving meaning, which is an innovative advance beyond descriptive analyses.

### Weaknesses
- While broad in scope, the work is methodologically complex, and the abundance of metrics (stress, distortion, spectral flatness, etc.) may obscure key takeaways.
- The evaluation relies heavily on synthetic emotion text for subspace construction, which may bias the identified directions.
- Although the paper claims semantic preservation under steering, this is mostly supported by cosine similarity metrics rather than human evaluations.

### Questions
- How does the emotional manifold evolve across training or fine-tuning stages? Does it emerge early or gradually with language exposure?
- Could the authors validate the psychological interpretability of latent axes quantitatively (e.g., correlations with human valence/arousal ratings)?
- Does the steering module modify only internal representations, or can it predictably change generated emotional tone in open-ended text?

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
This paper analyzes how large language models (LLMs) internally represent emotions. Through geometric and probing analyses, the authors identify a low-dimensional “emotional subspace” embedded across model layers, in which affective states are encoded directionally and often linearly decodable. They show this structure generalizes across multiple emotion datasets and five languages, producing a broadly consistent emotional manifold. The authors further develop a learned steering module that can intervene on hidden states to shift the model’s internal emotional perception toward target emotions while largely preserving semantic content; evaluations report strong post-steering classification accuracy for many basic emotions across models and languages. The study combines alignment metrics (cosine similarity, regression error), distortion/stress diagnostics, linear probes, and qualitative rewriting examples to characterize representation geometry, cross-domain robustness, and steerability.

### Strengths
- Identification of a low-dimensional, directionally encoded emotional manifold: The paper demonstrates that emotions in LLMs occupy a low-dimensional subspace that is interpretable and directionally organized across layers, with principal axes (PC1–PC3) showing high rank correlations in many models/layers.
- Cross-dataset and multilingual generalization of emotional structure: Using eight emotion datasets spanning five languages and diverse textual styles, the authors show that the extracted emotional subspace generalizes (low alignment distortion, above-chance linear probe accuracy), supporting the existence of a near-universal affective subspace in multiple LLM families.
- A learned intervention/steering module that controls internal emotional representations: They introduce and evaluate a module that shifts hidden states toward target emotions. Post-steering emotion prediction rates typically rise substantially (often >85% for many emotions), while semantic-similarity loss remains low, indicating control without wholesale semantic degradation. The method is evaluated across model families and languages, with ablations in the appendix.

### Weaknesses
- Geometry vs. local distortion — inconsistent relational preservation: Although global alignment measures (cosine, regression) are often strong, stress and distortion analyses reveal notable local warping of relative geometry in many layers and datasets. Thus the emotional manifold is not uniformly faithful to human emotion-space relations, which complicates interpretation and downstream use.
- Uneven multilingual and dataset robustness: Performance and steerability degrade in lower-resource settings (e.g., some emotions in Hindi/Bhaav), and certain datasets (e.g., Go-Emotions) show high layer-wise distortion. This suggests lexical sparsity, annotation imbalance, or domain mismatch limit universality claims and practical applicability across all languages/styles.
- Potential semantic and safety/ethical concerns with steering: Although semantic-similarity loss is reported low, steering produces surface rewrites that can alter tone, register, or pragmatics (examples show forceful rewrites for anger). The paper does not deeply address possible misuse (manipulating perceived emotion), downstream impacts on user trust, or safeguards for safe deployment. Additionally, steering effectiveness varies across emotions and models; some target emotions remain difficult to induce reliably.

### Questions
SEE WEAKNESS

### Soundness
2

### Presentation
2

### Contribution
2
