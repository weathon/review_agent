# CRISP - Complexity-based Reasoning of Internal Subprocessing

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
The growing deployment of artificial intelligence (AI) systems in safety-critical domains has underscored the need for transparent and trustworthy models. While existing explainability methods primarily focus on end-to-end interpretations, they often fall short of revealing the internal processing dynamics of deep networks. In this paper, we introduce CRISP a novel approach that decomposes neural networks into interpretable subprocesses, enabling a layer-wise analysis of hidden representations. Our method constructs interactive, low-complexity representations of input-output transformations within hidden layers, facilitating a deeper understanding of network behavior beyond final predictions. We present a framework and empirical validation for Convolutional Neural Networks (CNNs), demonstrating the method’s potential to support more fine-grained, process-level insights into model operation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes an XAI method for reasoning at the intermediate-layer level. The authors leverage the $QI^2$ metric, which quantifies the nonlinearity between the input and output spaces based on local input-output transformations, to construct anchors for XAI methods (feature attribution is performed using LRP). 

The approach is demonstrated on multiple examples using VGG-16 (trained for age classification) and U-Net (trained for railway segmentation). The results indicate that the proposed method successfully identifies meaningful and informative features.

### Strengths
1. The paper sheds light on explanations in hidden layers - an interesting and emerging subdomain within the broader field of XAI.
2. The qualitative analysis (right side of Figures 2-7) provides interesting insights and demonstrates the potential of the proposed method.

### Weaknesses
1. Contribution / Novelty: Most of the methodology builds upon [Geerkens et al. (2024)] and extends it to hidden layers. The novelty therefore lies primarily in leveraging previous work within the context of XAI for intermediate layers, which remains limited. The technical contribution - formulating the current problem as an input to the prior framework - appears rather straightforward.
2. Self-containment: The paper does not meet minimal readability requirements. Key definitions are omitted, making the paper non–self-contained and difficult to understand without consulting prior work. Specifically, $QI^2$ is not formally defined (only $QI^2R$ is provided). Furthermore, formal definitions are missing for $d_{RI}$ (Eq. 4), $d_{RO}$, $T_N^2​$ (Eq. 3), and $T^2_N$​ (line 172). Every term introduced in the paper should be clearly and formally defined.
3. Writing: A "Preliminaries" section is missing, and several definitions appear in the Methodology section even though they are not newly introduced in this paper.
4. Figures: The left side of Figures 2-7 is unclear, sometimes even invisible, and lacks proper explanation on axes X,Y meaning.
5. Evaluation: A systematic evaluation with quantitative metrics is missing. The authors do not propose measurable criteria to assess their method and provide no benchmark-based analysis. Consequently, it remains uncertain whether the presented (albeit interesting) visualizations were selectively chosen.

### Questions
1. Extendability: The authors present their method as a general one, in contrast to other approaches (e.g., AttnLRP). To support this claim, it would be expected to include experiments on additional architectures. Why were only CNNs used for evaluation?
2. Is there any quantitative metric to evaluate the performance of the proposed method?
If the method enables network decomposition, can it also be applied to hidden-output or hidden-hidden reasoning, in addition to input-hidden reasoning? The authors are encouraged to demonstrate this capability. Otherwise, the term “network decomposition” may not be accurate.
3. In line 163, $q$ is defined as the left-hand side of the pair in Eq. (1), yet it is also described as part of $\mathcal{D}$, which consists of pairs - please clarify.
4. What do $k$ and $v$ represent in Fig. 2?
5. The left-hand images in Figures 2-7 are not sharp enough; please improve their resolution.
6. Typos: Line 272: “Figure fig. 2”.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a framework for providing localized, layer-level explanations of neural network behavior. It employs the QI² metric to quantify the complexity of layer responses with respect to input patches of varying sizes. The authors claim as contributions the application of QI² to neural network analysis, demonstrating the generalizability of this descriptor, the ability to analyze intermediate network abstractions, and the potential to decompose neural networks into interpretable functional substructures. The proposed framework is evaluated on two different network architectures.

### Strengths
- The paper proposes a framework for analyzing the **internal mechanisms of neural networks**, which can be applied across different architectures, as demonstrated with **VGG16** and **U-Net**.
- The manuscript is **well-written** and includes **illustrative examples** that showcase the approach.
- The proposed strategy has the **potential to support model inspection in safety-critical AI applications**.

### Weaknesses
- **Darker consistent pathways:** The concept of “darker consistent pathways” is not clearly explained. You mention that it is based on **coherence**, but it is unclear to me  **why linear relationships (lower QI² values) are considered preferable** when aggregating them across patches. A more detailed explanation or illustrative example would help clarify the rationale behind this choice.
- **Scalability and unit selection:** Analyzing the **complete network** seems impractical in this framework. How are specific units selected for analysis? Do you analyze entire layers, individual channels, or both? How an early layer impacts in the next layer’s knowledge?
- **Interpretation of features:** The claim that early layers detect **skin tone** is unclear to me. Is this determined **visually**? How many people performed the analysis to categorize the focused regions? Are there additional examples to support this claim?
- **Low- and high-level features:** The observation that features of different levels appear at various depths of the network has been described in previous xAI studies (example: Zeiler et al 2011 [1]).
- **Integration with attribution methods:** The claim that integrating QI² with methods such as **LRP** “extends traditional explainability techniques” is **debated**. It seems that QI² rather **directs the analysis** than fundamentally extends existing methods.

[1] M. D. Zeiler, G. W. Taylor, and R. Fergus, "Adaptive Deconvolutional Networks for Mid and High-Level Feature Learning," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2011, pp. 2018–2025, doi: 10.1109/ICCV.2011.6126474.

### Questions
- **Segmentation experiments:** can you clarify the statement that the network exhibits a “classification-like behavior (Fig. 5a, Fig. 5b)” ?
- I believe the methodology could be **better understood if accompanied by a visual diagram**, which would help clarify the workflow and interactions between components.
- I included extra questions above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
**CRISP (Complexity-Based Reasoning of Internal Subprocessing)** is an interpretability framework that analyzes a neural network’s *internal* computations layer by layer. The method (1) decomposes activations into paired input/output patches at each module, (2) scores local transformation “complexity” using a nearest-neighbor–based QI² statistic to produce per-layer saliency profiles (MLQI²/SHLQI²), and (3) uses the most complex (salient) subregions as *anchors* to perform hidden-layer attributions (LRP-CMP) and qualitative inspection.

**Key ideas**
- **Complexity profiling:** Quantifies where a layer performs the most non-trivial transformations by comparing local neighborhoods before vs. after the layer.
- **Anchor selection:** Picks high-SHLQI² regions as representative internal “subprocesses” to analyze and attribute.
- **Hidden-layer attribution:** Propagates relevance from these anchors to visualize how evidence flows *within* the network, not only at outputs.

**Use cases**
- Demonstrated on **VGG16** (age classification) and **U-Net** (railway segmentation), yielding layer-wise narratives (e.g., early texture/edges, mid-level parts, late recombinations).

**Claims**
- Provides a modular, model-agnostic pipeline to surface functional substructures within layers.
- Offers a complementary perspective to output-centric XAI by focusing on internal transformations.

### Strengths
1) **Clear, modular pipeline**
   - Decomposes each layer into input/output patches → computes QI² complexity profiles (MLQI²/SHLQI²) → selects anchors → runs hidden-layer attribution (LRP-CMP).
   - Evidence: Step-by-step equations and algorithm blocks make the method replicable; consistent use across VGG16 and U-Net.

2) **Focus on internal processes (beyond output-centric XAI)**
   - Anchors are chosen inside the network (high-SHLQI² regions), enabling attribution and visualization of *internal* subprocessing rather than only final logits.
   - Evidence: Layer-wise narratives (early texture/edge processing, mid-level structure, late recombination) align with known CNN stages.

3) **Model- and task-flexible design**
   - Patchwise analysis and nearest-neighbor–based QI² scoring are architecture-agnostic in spirit and apply to different tasks (classification, segmentation).
   - Evidence: Demonstrations on VGG16 (age classification) and U-Net (railway segmentation) without changing the core pipeline.

4) **Actionable diagnostics**
   - SHLQI² profiles highlight “where” complex transformations occur, guiding targeted inspection, ablations, and potential network edits.
   - Evidence: Identified salient regions correspond to visually meaningful structures (e.g., rail boundaries, facial regions), informing qualitative assessments.

### Weaknesses
1) **Lack of baselines**
   - No head-to-head comparisons against established internal interpretability methods (e.g., concept discovery/ACE, Network Dissection, prototype/dictionary approaches, hidden-layer attributions).
   - Impact: Hard to judge whether CRISP offers improvements in fidelity, stability, or usefulness.

2) **Lack of quantitative results**
   - Results are largely qualitative; missing metrics for stability (across seeds/splits), localization (Pointing Game/IoU vs. parts), and causal impact (Δlogit/ΔIoU under ablation of salient regions).
   - Impact: Claims about “complexity hotspots” and functional substructures are not empirically substantiated.

3) **Insufficient human evaluation**
   - Claims are illustrated with a few images but lack systematic human studies (e.g., nameability/consistency of discovered patterns, expert ratings, task-relevance assessments).
   - Impact: No evidence that surfaced internal structures align with human concepts or aid human understanding in a repeatable way.

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a method to study input-output transformations between convolutional layers, named CRISP. CRISP first decomposes a feature map from convolutional layers into patches. For each patch, CRISP computes a set of neighbors up to distance d, where each tuple is a pair of input patch and resulting patch from two layers. The framework then assigns each neighborhood a QI score to estimate the (non-)linearity within it. The collection of QI scores is visualized in a 3D histogram, which users can use to identify patterns and guide downstream tasks (e.g., attribution, layer structure analysis, etc.). The paper showcases the advantages of the framework by analyzing two deep neural networks in detail.

### Strengths
- The usage of QI is original and seems interesting as a general tool for patch-based work in interpretability.
- The paper is clear and easy to read, even though some details could have been included in the main text (e.g., the identification of the patch dimensions through the receptive field).

### Weaknesses
- **Weak experimental setup**: The main weakness of the paper is the experimental section. The paper presents two visual analyses of deep neural networks using its method. However, there are no quantitative results in terms of metrics or user studies, and no alternative methods or frameworks are compared against. Together, these weaknesses result in a lack of contextualization for the benefits of the proposed method and make the work difficult to extend. Specifically:
   - **There is no comparison against other methods**. The only attempt is in the first section, comparing a heatmap produced by the proposed method with one reported in the LRP paper. A comparison for a single image is not enough and also the models on which they are computed are not the same since authors re-trained the network. The authors state that their method should be considered complementary rather than an alternative to attribution methods. If this is the case, an experimental setup could compare scenarios where users have access only to (multiple, not just LRP) attribution methods and those where they can combine (multiple) attribution methods with the proposed method. In addition, I believe the work seems similar in principle to the literature on Deconvolution, and specifically on the feature visualization method explained in the DeconvNet paper. Contrary to the related work section, DeconvNet does not use predictions or class information and achieves a similar goal of explaining transformations of the input through the layers. The authors are encouraged to explore the DeconvNet related literature for potential competitors.
   - **There are no metrics or quantitative measurements to measure the progress**. Beyond competitors, the method is tested visually without any metric that could help users or researchers express the improvement and novelty introduced by the paper. If the authors believe that current metrics are inadequate (as stated in the paper), they should propose new metrics to show what current methods miss and how the proposed method improves on them. Metrics are also important for future research. Researchers building on this work will struggle to measure progress otherwise. If no quantitative metrics are possible, at least a user study demonstrating that the proposed method aids interpretability compared to classic frameworks would be useful.

- **Reliance on user guidance** (connected to the previous point): This is an unclear point from the current description. From my understanding, the analysis is guided by the SHLQI histogram and there is no end-to-end automation for explanations. If so, the framework is better suited as a visual analytics tool rather than a standalone framework, and would benefit from user studies focused on specific tasks. Conversely, if the framework can be considered as standalone and the histogram can be bypassed for downstream tasks (such as by selecting patches that maximize an objective connected to QI), this automation should be properly evaluated as previously explained.


- **Claims about generalization**: The paper emphasizes the property of being model-agnostic and rules out potential competitors (e.g., work in mechanistic interpretability) based on this claim. However, the paper is focused and tested solely on convolutional neural networks, explicitly relying on receptive fields to identify patches. While it may be possible in principle to identify patches in other architectures, their semantic meaning would differ substantially compared to CNNs. As stated in the paper’s limitations, *“Demonstrating this generality empirically remains important future work”*. If so, claims about being model-agnostic should be removed and alternative competitors that work in CNN should be considered. Alternatively, the framework should be tested on at least a couple of different architecture families to support generalization claims.

### Questions
See weaknesses for general concerns. In this section, I would like to ask authors their rationale for excluding Deconvolution and all the related work that analyze how the input is transformed through the network.

### Soundness
1

### Presentation
3

### Contribution
1
