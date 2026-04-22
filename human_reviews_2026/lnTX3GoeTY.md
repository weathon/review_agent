# Feature segregation by signed weights in artificial vision systems and biological models

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 6, 4, 6

## Abstract
Signed connectivity is fundamental to neural computation in both brains (excitatory/inhibitory) and machines (positive/negative). Yet the role of signed weights in shaping visual representations in object recognition remains unclear.
Dale's Law, the biological principle that neurons send exclusively excitatory or inhibitory outputs, is typically not enforced in artificial neural networks (ANNs). 
Here, we find that accuracy in ImageNet-trained ANNs correlates with the spontaneous emergence of sign-specific  "Dale-like" segregation in their output layers.
Ablation and feature visualization reveal a functional segregation in ANNs: removing positive inputs primarily disrupts localized, object-related structure, while removing negative inputs alters mainly dispersed background textures. 
This segregation is more pronounced in adversarially robust models, persists with unsupervised learning, and vanishes with non-rectified activation functions.
We validate these observations in the macaque ventral visual cortex (V1, V4, and IT) using encoding models and in vivo feature visualization.
The features recovered by encoding models qualitatively matched those identified in vivo.
Model representations changed more upon positive than negative input ablations.
We analyzed the most Dale-like units across neuron models, positive units showed localized features, while negative units showed larger, more dispersed features.
Consistent with this, experimentally clearing the background around a neuron's preferred feature enhanced its response, likely by reducing inhibitory drive.
Our results suggest that both artificial and biological vision systems segregate features by weight sign: positive weights emphasize object-related features, while negative weights refine context. This highlights a convergent representational strategy in brains and machines, yielding predictions for visual neuroscience.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, the authors study the role of positive and negative weights in artificial neural networks and their analogues with excitatory and inhibitory synapses in biological neural networks.  By performing visualization and ablation experiments in a variety of convolutional neural networks, the authors suggest that positive weight connections primarily encode object-relevant features while negative weight connects play a larger role in encoding background content.  The authors further train linear mappings to predict biological neural activity in monkeys from learned representations in CNNs.  Performing further visualization and ablation experiments with these learned mappings, the authors show the impact of projection weight sign on biological neural firing rate and conclude that both biological and artificial visual systems segregate features based on weight sign.

### Strengths
- The authors motivate their work based on biological principles.  Studying feature segregation in the visual system with image models is an interesting idea and mechanistic understandings that result from such analyses can be important to better understand information encoding in both biological and artificial systems.
- Analyses provided in this paper extend beyond the study of artificial neural units.  The authors sought to corroborate their findings in-vivo.

### Weaknesses
- With regard to sensitivity of output unit weights to positive and negative connections (results section 4.1), why would we expect anything different than the results presented (i.e., that positive output weights are important for encoding object-relevant information and negative weights for non-object information)?  Since these connections directly contribute to an output unit that is trained to have high activation only when a specific object is present (via the classification loss), shouldn't we expect these observed results as our default hypothesis?  And if so, why would this not also explain why we saw a similar, but lesser effect, when studying unsupervised models and no segregation in models trained with tanh units (which could contribute evidence for the presence of an object to an output unit by multiplying a negative activation with a negative weight or positive activation with a positive weight)?
- I am unconvinced by the qualitative analysis of section 4.4.  In early and mid layers, there is no evidence that visualized positive (negative) features are used primarily for object (background) information.  All visualized features in these early layers could likely be activated with various objects (or background textures).  More causal and quantitative evidence is needed to support this claim.
- Analyses in section 4.5 rely on the assumption that the model predicts biological neural activity well on stimuli outside of the training set.  This is claimed to be true (line 411 “images from intact models reliably drove biological neurons to firing rates…indicating out-of-distribution generalization”) but Figure 16, left, would suggest otherwise: outside of the training data, neuron responses appear to be poorly predicted by the model (what are the $r^2$ scores for held out in-distribution predictivity and out-of-distribution predictivity?).  Driving biological neurons to fire more than they do for natural images could simply be a result of the fact that the presented synthetic images are out of the natural image distribution.

### Questions
- For analyses in section 4.5, why were units from monkey visual areas V1 and V2 aligned with the penultimate representations in AlexNet when earlier layers of CNNs tend to be better predictors of activity in the early ventral stream?
- With regard to weakness bullet point 3, what is the variance explained (or pearson correlation) between biological neural activity and model neural activity for extrapolated images?

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
This paper investigates an interesting organizational principle in neural networks: whether positive and negative weights segregate different types of visual information. Through systematic ablations and feature visualizations across multiple ImageNet-trained architectures, the authors demonstrate that positive weights preferentially encode localized, object-like features while negative weights encode more dispersed, texture-like or background features. They show this segregation is enhanced in adversarially robust models, persists with unsupervised pretraining, but critically depends on ReLU activations. The authors extend their investigation to primate ventral stream recordings, fitting linear models from CNN features to neural responses and performing both in silico and in vivo feature visualizations.

### Strengths
## Strengths

* Originality: The systematic investigation of feature segregation by weight sign is a novel angle in interpretability research. While prior work has explored weight pruning and sparsity, the specific hypothesis that signed weights might organize visual information differently (analogous to biological E/I circuits) is creative and underexplored. The connection to adversarial robustness is particularly original.


* Quality: The experimental design is thorough and rigorous. Testing across multiple architectures (AlexNet, VGG16, ResNet50, robust ResNets) with different training regimes (supervised, unsupervised, robust) provides strong evidence that this is a general phenomenon rather than an architectural artifact. The ablation methodology is sound, using cumulative weight removal based on magnitude. The use of two different GANs (DeePSiM and BigGAN) for feature visualization helps ensure results aren't generator-specific. Scaling to 100 ImageNet classes (Fig. 11) and validating with LPIPS demonstrates robustness of findings.


* Clarity: The paper is generally well-written with effective visualizations. Figure 1 provides a clear overview of the phenomenon. The comparison between ReLU and Tanh networks (Fig. 2) elegantly isolates the role of rectification. The progression from output layers to intermediate layers (Section 4.4) is logical. Methods are sufficiently detailed for reproduction.

* Significance: This work has potential impact for both AI interpretability and neuroscience. For AI, it offers a new lens for understanding how networks organize information and suggests that analyzing positive/negative pathways separately could aid interpretability. The connection to adversarial robustness is important - if robust models show stronger segregation, this could inform development of more interpretable models. For neuroscience, while the biological validation is preliminary, the approach of using ablation-based feature visualization to generate predictions for circuit experiments is valuable and could inspire new experimental designs.

The finding that ReLU is necessary for segregation connects nicely to recent work on how activation functions shape representational geometry, adding to our theoretical understanding of deep learning.

### Weaknesses
## Weaknesses

While the core ANN experiments are well-executed and reproducible, several issues limit the soundness of the overall claims:
* **Interpretation ambiguity:** The central interpretation of "object vs. background" segregation is not sufficiently justified. The observed effects could equally reflect:

1. Local vs. global spatial structure
2. High vs. low spatial frequency content
3. Shape vs. texture information
4. Figure vs. ground organization

The YOLOv7 objectness metric provides only weak support, as it reflects one specific computational definition of "object" that may not align with what the networks actually learned. I recommend: (1) testing alternative interpretations using texture/shape metrics (e.g., Geirhos et al. 2019 style analyses), (2) frequency domain analysis of the features, and (3) more direct tests of foreground/background using segmentation masks.

* Biological validation concerns. The neuroscience component has significant limitations:

1. R² = 0.27 is quite low for claiming the models "capture" neural representations
2. 160 images are substantially a small amount (although I believe experiments could be tricky, I'm afraid this could limit the final conclusions) 
3. The leap from positive/negative ANN weights to excitatory/inhibitory neurons oversimplifies Dale's law, which involves distinct cell types, complex dynamics, and circuit-level interactions not present in ANNs
4. The penultimate layer may not be optimal for predicting ventral stream responses

I suggest: (1) comparing against larger image sets to assess whether findings hold, if possible, (2) testing multiple layers to find optimal predictions, (3) being more careful about the analogy to E/I balance, and (4) acknowledging these are computational models of neurons, not direct biological measurements.

Lastly, I believe that the paper excellently demonstrates **what** happens but provides no insight into **why**. What properties of the learning objective, training dynamics, or network architecture cause this segregation to emerge? Adding:

1. Analysis of weight evolution during training
2. Theoretical framework or toy models
3. Predictions about when this would/wouldn't occur would significantly strengthen the contribution.

-----

The paper is generally well-written with clear figures. However, some improvements would help:

* The "Dale's law inspired analysis" in Section 4.4 and Appendix A.3 feels somewhat disconnected from the main narrative. Consider integrating it more smoothly or clarifying its relationship to the classification unit findings.
* Figure 1B-C would benefit from showing more than just the best visualization per condition to convey variability
* The biological methods (Section A.1) could be condensed, with some details moved to supplementary materials
* Some key results (like the 100-class validation) are relegated to appendix; consider moving to main text

### Questions
## Questions for Authors

To summarize the points raised above: 

* Alternative interpretations: Have you tested whether the segregation reflects spatial frequency rather than object/background? Could you show power spectra of features preferred by positive vs. negative weights?

* Training dynamics: When does this segregation emerge during training? Does it correlate with the development of robust features or adversarial robustness?

* Mechanistic predictions: Can you predict which architectures or training procedures would show stronger/weaker segregation based on some principle?

* Causality: The ablation experiments show correlation, but do they demonstrate that positive weights cause object representations? Could there be confounding factors?

### Minor Issues

* Table 2 shows positive/negative weight ratios are very close to 1:1. This is interesting but underexplored. Why this balance?
* The limitations section is quite brief; consider expanding

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
This paper is a commendable piece of interdisciplinary research that bridges mechanistic interpretability in deep learning with fundamental principles of neurobiology. The overall contribution is substantial, and the authors' efforts to validate their computational claims with biological data are notable. However, the work suffers from weaknesses in its presentation, methodological rigor, and conceptual clarity across both the AI and Neuroscience domains.

### Strengths
1) The core contribution is the novel insight that the **signed weights ($+/-$) in ANNs serve distinct, functionally segregated roles**, which moves beyond treating weights merely as mathematical optimization parameters. This work is a crucial step toward understanding the computational significance of weight structure.

*   **Interpretability Breakthrough:** The paper provides a clear, interpretable meaning for the sign of weights—positive weights encode **object features**, and negative weights encode **background/contextual information** (i.e., feature segregation). This is a significant advance in explaining *why* modern deep networks achieve robust visual representations.
*   **A Solid Link to Neuroscience:** The work addresses a fundamental question in both fields by investigating whether a balanced mixture of positive and negative signals is required for representations. This alignment of **computational principles (signed weights)** with **neurobiological function (excitatory/inhibitory balance)** makes it particularly relevant for interdisciplinary venues like ICLR.

2) The methodology for cross-validation is robust, demonstrating a deep commitment to scientific rigor and biological plausibility.

*   **Biological Plausibility:** The use of **novel, complex V4 neuronal recordings from macaque monkeys** is a major strength. This direct, first-hand biological validation significantly elevates the paper's credibility, ensuring the claimed segregation is not a mere artifact of the ANN architecture but a potentially universal principle of visual processing.
*   **Effective Computational Methodology:** The authors employed an appropriate and sophisticated methodology—**customized activation maximization (feature visualization)**—to probe the specific functional preference of both positive-only and negative-only weighted inputs. This technique is well-suited for visualizing the intrinsic features encoded by the network, moving beyond simple classification tasks.
*   **Architecture and Stimuli Relevance:** The choice of **ResNet** architecture is appropriate given its deep usage in vision and common comparison with the visual processing stream, enhancing the generalizability of the findings.

3) The paper's execution required non-trivial effort across multiple domains, reflecting the significant endeavor of the research team. Conducting novel electrophysiology experiments and integrating them with advanced computational analysis is a demanding task that is not commonly accomplished, warranting praise for the authors' hard work.

### Weaknesses
1) The paper suffers from a general lack of clarity and transparency that severely impedes the reviewer's ability to assess its claims and ensures reproducibility.

*   **Insufficient Referencing and Background:** The Introduction lacks essential references to support foundational claims and necessary background knowledge, particularly in the neuroscience domain. This poor referencing undermines the academic rigor required to justify the "fundamental question" being investigated and the paper's overall scholarly context. While some of references are addressed in related works, referencing only 2 papers in 6 paragraphs of introduction needs to be extensively revised considering the majority of the community is not familiar to the neuroscience (e.g., Dale's law, brain's visual stream, brain region etc)
*   **Misleading Presentation of Core Methods:** Despite having detailed procedures in the Appendix, the minimal description in the main text creates unnecessary ambiguity regarding the source of the biological data (new recording vs. open-source) and the specifics of the complex feature visualization. This structure places an undue burden on the reader and reviewers and minimizes the necessary rigor for introducing novel electrophysiology data. **(Recommendation: Integrate a summary of all critical methods into the main body of the paper., even there are repeated section in methods and appendix)**. Moreover, figure 1A seems to describe the concept of method, how exactly monkey recording can reconstruct image while spliting exicatory or inhibitory is really hard to grasp. 


2) Critical technical details are presented with insufficient formal rigor, making the analysis difficult to verify.

*   **Ablation Equation Ambiguity (Line 147):** The equation for ablation (e.g., using $\alpha$) lacks formal clarity. It is unclear if $w$ refers to layer-wise weights or the entire model, and whether the operation respects the sign of the weights or only their magnitude. Ambiguity regarding whether $\alpha$ can exceed 1 suggests a lack of precise definition for the weight's normalization or clipping.
*   **Undefined Notation ($L_{\infty}$):** The term $L_{\infty}$ (Line 138) must be explicitly defined (presumably the $\ell_{\infty}$ norm) as this basic notation should not be assumed or left to the Appendix for clarity in the main body.

3) The core premise of the cross-disciplinary comparison is built on a simplifying assumption that requires a more robust discussion.

*   **Structural Mismatch in Synapse Analogy:** The direct mapping of an ANN's signed weight to a biological synapse is structurally flawed. A standard ANN unit receives both positive and negative weighted inputs, whereas a biological neuron's *outgoing* connection typically adheres to Dale's Principle (single sign). The paper fails to rigorously address this significant **structural mismatch**, treating the *sign of the weight* as an equivalent proxy for the *sign of the synapse*. This omission weakens the fundamental premise that the observed feature segregation is truly analogous to biological circuit function and should be thoroughly discussed.

4) Lack of Justification for Sampling (Line 658)

- The authors' justification for data and class sampling is incomplete. The rationale for subsampling the **11 specific classes** for the analysis (Line 658) is not clearly articulated. Without a clear justification for selecting these particular classes, the generalizability of the reported results regarding feature segregation is questionable and may introduce selection bias.

### Questions
While the core intellectual efforts and the underlying experimental findings are commendable and demonstrate significant effort, the overall packaging and presentation of the manuscript detracts from its substantial contribution. The density of information, poor flow, and insufficient referencing in the main text make it unduly challenging for the reader to immediately grasp the rigor and context of the work. The authors are **strongly encouraged to significantly refine the narrative clarity and academic referencing** to appropriately honor the complexity and value of their research.

### Soundness
2

### Presentation
1

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
Across artificial and biological neural networks (macaque visual system) for vision, they show that positive weights emphasize objects, while negative weights encode context.

### Strengths
- Original study asking an important question about the role of negative vs positive synapses in the visual system and ANNs.
- Careful analyses which seem to mostly support the claims of the abstract.
- Clear writing for the most part.

### Weaknesses
- I am not entirely sure (but I may have missed some of the reasoning steps) that the claim of the abstract about the macaque visual system ("Our results demonstrate that both artificial and biological vision systems segregate features by weight sign: positive weights emphasize objects, negative weights encode context") is fully supported by the analyses provided. Could you please explain how the analyses support that claim?

- In the abstract, it is said: "Notably, some units closely approached Dale's law: the positively projecting units exhibited localized features, while the negatively projecting units showed larger, more dispersed features." How is this observation related to Dale's law, which states that a single neuron releases the same neurotransmitter at all of its synapses?

### Questions
- In order to test the emergence of (approximate) Dale's law in the artificial networks, one one need to run a statistical test to see if neurons tend to have more output connection weights of the same sign than expected by chance.

- Figures are a bit small. Consider enlarging in the final version.

### Soundness
3

### Presentation
3

### Contribution
3
