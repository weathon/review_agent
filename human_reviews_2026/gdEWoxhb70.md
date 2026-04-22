# Learning Concept Bottleneck Models from Mechanistic Explanations

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
Concept Bottleneck Models (CBMs) aim for ante-hoc interpretability by learning a bottleneck layer that predicts interpretable concepts before the decision. State-of-the-art approaches typically select which concepts to learn via human specification, open knowledge graphs, prompting an LLM, or using general CLIP concepts. However, concepts defined a-priori may not have sufficient predictive power for the task or even be learnable from the available data. As a result, these CBMs often significantly trail their black-box counterpart when controlling for information leakage. To address this, we introduce a novel CBM pipeline named Mechanistic CBM (M-CBM), which builds the bottleneck directly from a black-box model’s own learned concepts. These concepts are extracted via Sparse Autoencoders (SAEs) and subsequently named and annotated on a selected subset of images using a Multimodal LLM. For fair comparison and leakage control, we also introduce the Number of Contributing Concepts (NCC), a decision-level sparsity metric that extends the recently proposed NEC metric. Across diverse datasets, we show that M-CBMs consistently surpass prior CBMs at matched sparsity, while improving concept predictions and providing concise explanations. Our code is available at https://github.com/Antonio-Dee/M-CBM.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Mechanistic CBM (M-CBM), which introduces advancements in the field of mechanistic interpretability to the concept bottleneck framework. M-CBM takes a pretrained blackbox and trains a sparse autoencoder on its internal activations to extract human-interpretable concepts that are relevant for prediction. After having named and filtered the discovered concepts, a MLLM is used to label a non-random subset of images with these concepts. Subsequently, a standard CBM is trained on the discovered and labelled concepts for human-interpretable predictions. Furthermore, the authors introduce the Number of Contributing Concepts (NCC) metric, which measures the number of concept contributions required to explain $\tau$ of all absolute contributions. Results show improved performance on NCC and concept prediction, as well as qualitative prediction examples.

### Strengths
This work proposes to combine the two thriving fields of concept bottleneck models and mechanistic interpretability. As such, it is relevant in the current state of interpretability. It makes use of tested mechanisms from both fields, thereby fostering trust in the proposed method. The method is introduced clearly and the figure aids in understanding. M-CBM also shows NCC improvements compared to relevant baselines. Also, the extensive visualization of discovered concepts and examples in the Appendix is appreciated. Reproducible code is provided too.

### Weaknesses
1. While MechInterp is an exciting upcoming field, I am not convinced by the reliability of the concept annotation framework such that there is not a substantial amount of leakage in M-CBM. (i) To name the SAE activations a heatmap is used to localize the concepts. Heatmaps are famously unreliable [1] such that they might not capture what information the blackbox model is using. Then, the highlighted images are passed to the MLLM (Fig. 1), but it is highly unclear to me whether the MLLM truthfully adheres to the highlighted region, and whether the concept interpretation/naming is unique. (ii) It is a good leakage-reducing choice that concepts are subsequently labelled using a MLLM, rather than directly using the SAE activations. Still, there might be a considerable influence from the leaky concept naming critiqued in (i) because "Each call also includes a grid of the top-25 most activating images for the corresponding SAE neuron", which could lead the MLLM to mimic the concept annotation strategy from the leaky SAE, rather than focusing on actual concept presence. Additionally, the MLLM might also use class-leakage where it recognizes an object and determines a concept that relates to the object to be present even if it is not visible in the image. 

   To give a concrete example that might ease comprehension, lets say we are interested in the class "Radio". A SAE on a blackbox classifier might find a neuron that always activates when a radio is present. Then a good heatmap highlights most of the radio in each highly activated image. The MLLM that outputs concept names might consequently call the concept "analog control knobs and frequency dials", as it appears in many of the images and is often even highlighted. Note though that the concept name has nothing to do with the actual concept that the model uses, which is "presence of radio". Then, an MLLM is queried to concept annotate a selection of images. Two things that could lead the MLLM to annotate concept presence even if the concept is not there are (a) the 25 highly activated SAE images do not all contain the named concept, thereby the MLLM learns to still assign concept presence for radios even if it is not visible on the image. (b) The MLLM even without context might understand the presence of the object radio and that this object has the aforementioned concept, even if it is not visible, thereby labeling it positive. In the end, the concept labels essentially behave like a class label, but hidden behind an interpretable concept name, which does not accurately capture how it behaves and/or when it activates.

   I am not saying this is necessarily always the case, however, with the current evaluation setup I do not think the amount of this leakage is quantified. Therefore, it is very hard to say whether the extracted concepts (and therefore the method) are meaningful and interpretable.

2. I am not convinced that NEC / NCC are sufficient metrics for capturing information leakage (as stated e.g. in the Abstract). Similar to the previous example, lets say I repurpose a frozen black-box's classifier to be "concept predictor" and rename the classes to some good concept names. Then, initialize the Identity Mapping from my new concept predictor to the classes. As I understand it, this method would have the black-box's accuracy at NCC=1 while being completely uninterpretable and leaking (similar to VLG-CBM). Thus, I do not think the current evaluation setup is able to differentiate meaningful, interpretable posthoc CBMs versus uninterpretable class-leakage.

   To resolve this, off the top of my head, I could think of either performing human user studies or measuring alignment of discovered concepts with known ground truth concepts on a controlled dataset where actual concepts are known.

Other weaknesses:
* In my opinion, the proposed combination of CBM and SAE does not provide a lot of novel ideas but rather combines two existing approaches, which is meaningful, but not original.
* Some concepts in Figure 10, e.g. Tiger stripes or Yellow banana peel with brown spots essentially describe the class, thereby not providing any additional interpretability beyond a blackbox classifier.

[1] Adebayo, Julius, et al. "Sanity checks for saliency maps." Advances in neural information processing systems 31 (2018).

### Questions
I invite the authors to address any of my statements from the Weaknesses section, as they are my main source of concern.

For W1: 
* Could the authors provide the highly activated images used when querying the MLLM for obtaining the concept names from Figure 3, jointly with the heatmap overlayed, such that it is clear whether concept names relate to the highlighted image parts?
* What is the prompt used for dataset annotation?
* How do the top-25 most activating images affect the MLLM during dataset annotation? Can they be safely removed, or do they affect predictive performance downstream?

Other questions:
* In similar spirit to above, overlaying the heatmaps in Fig. 8-10 would provide additional informativeness.
* Was the consistency of dataset annotation for the 5x5 grid analyzed? I.e. when processing each image separately, would the annotations change? Because as far as I know, MLLMs are not the best at localizing inputs, therefore I am unsure whether a 5x5 grid might pose difficulties.
* Is there a specific reason why elasticnet with $\alpha=0.99$ was chosen instead of just using the simpler LASSO (i.e. $\alpha=1$).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose a new method for extracting concepts in concept bottleneck models by leveraging sparse autoencoders (SAEs) on top of a black-box architecture. They first train an SAE and identify concepts by examining the non-dead neurons. For each neuron, they select the most and least activating images from the black-box model’s dataset and use a multimodal large language model (MLLM) to assign concept labels. To train the concept bottleneck layer, they construct a semi-automated dataset containing 1,000 images per concept to determine concept presence. In addition, they introduce a softer measure of concept sparsity (NCC), with the aim of enabling more controlled and flexible comparisons across concept bottlenecks.

### Strengths
- The methodology is intuitive, and its benefits are evident: leveraging concepts already present in the black-box model enhances interpretability.
- The approach achieves performance improvements over baselines when controlling with NCC.
- The paper is clearly written, and most aspects of the methodology are well justified (with the exception of the weaknesses noted below).

### Weaknesses
While the motivation for NCC is intuitively explained (lines 303–305), it would be valuable to provide evidence for why the additional concepts are necessary. When concepts contribute substantially less than the dominant ones that NEC would capture, are they still useful and interpretable, or do they risk overfitting? NCC appears to strike a balance between interpretability and overfitting, but it is unclear how much the method benefits from this added flexibility. Would the same performance patterns hold under the stricter NEC criterion? Alternatively, is the improvement primarily due to the inclusion of negative concepts? Reporting NEC scores alongside NCC results could help clarify the extent to which this flexibility contributes to the method’s effectiveness.

### Questions
- While this is a clearly well-written and valuable paper, I remain uncertain about the role of NCC. One way to address this would be to also report performance under NEC; another would be to demonstrate that the extracted concepts have an intuitive and interpretable basis.
- Clarification: Beyond instructing the MLLM not to use class names as concepts, did you perform any filtering to ensure that class names were indeed excluded? (line 210)

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a mechanistic Concept Bottleneck Mode --- M-CBM --- that brings a new opportunity to extract concepts on the fly using Sparse Autoencoders. For annotating and giving the concepts a human-readable “entity," the authors prompt multimodal LLMs, with a special accounting for compute and costs. Notably, the authors introduce a new metric for measuring the sparsiness and consequently, expressiveness of the activated concepts. This metric is based on the prior work and evaluated throughout the paper. Finally, by rigorous comparisons with popular baselines in the field, M-CBM emerges as a robust and strong baseline, facilitating future research.

### Strengths
**The baselines are recognized in the field and strong-performing.** The authors conduct their comparison in a simple and rigorous manner evaluating the same backbone encoder for each model on the same dataset. Considering that some datasets require a broader concept set to perform well on, the authors employ larger backbone models for all baselines, e.g., on ImageNet.

**The idea is sound and well-presented.** I like the way how authors described their framework in Sec. 3. Effort in proposing a new metric for measuring sparsity leveraging concept contribution.
Also interesting how the authors ablate their meric vs. accuracy in Fig. 2 using the baseline VLG-CBM model

**Accounting for compute and costs.** When using (M)LLMs to assign each SAE neuron to concept name or to annotate some subset of data to create a matrix of "Present" / "Missed" concept per image, the authors mention a precise cost of the OpenAI API per one annotation.
As well as stating that money spent scale linearly with the number of concepts. Thus, one can do the math and calculate the total spendings.

**Demonstrating the concept "yes / no" selection and concepts pertaining to image.** In Fig. 3, the authors show a precise example of which concepts are being selected to a particular class and which receive the negative weight. The provided concepts look very meaningful and not monosyllabic, unlike concepts in many prior works that were obtained with less sophisticated methodology.

### Weaknesses
**Your NCC metric.** I do appreciate the introduced metric as the extension of the NEC metric from the prior work on VLG-CBM. As far as I understand, the parameter $\tau$ controls, let us say, the "sparsity" of the impactful concepts. Meaning that the model with smaller NCC (and also the smaller NEC metric) provides more explainable decision since only few the main important concepts are activated. So it is naturally good to keep both metrics relatively small, or contrary, a large NEC (NCC) can result in information leakage. Given that both metrics are pretty similar in their essence, I would like to see more comparisons and discussion between two of them, in flavour of your Fig. 2 (or the Fig. 3 from the VLG-CBM work). Importantly, results of VLG-CBM demonstrate only minor, even negligible, drop in accuracy when their NEC is small. Which leads me to questions:
1) Does the same holds with VLG-CBM under your NCC metric?
2) Does the accuracy drop of your M-CBM is also negligible under small NCC and NEC metrics?
In my opinion, offering discussion and, which is better, experiments on this merit in your manuscript will significantly increase the reader's confidence in your NCC metric.
Also, it would be nice to include results on NCC while sweeping $\tau$. Could it be that with $\tau = 0.5$ or less many concepts will satisfy this fraction of information and, thus, the decision-making process would be rubbish? How does this connect to the fact that even with randomly initialised concept bottleneck layer the model can give near-optimal accuracy when the VLG-CBM NEC metric is large?

**Regarding your comparison in Fig. 2 (a)**, I am also curious in the following: the VLG-CBM baseline uses up to 128 concepts for CUB --- see their Fig. 3 (c). Additionally, prior work Res-CBM [1] claims that the number of concepts should not be the larger the better, meaning there is some critical k, beyond which, the accuracy will drop. Do you think this will be observed when NCC (or similarly, NEC) are very large? Offer some discussion. Ideally, it would be nice to provide figures based on your Fig. 2 with very large NCC --- at least a couple of hundreds.

**Backbone model choice.** From "Sec. 5 Setup", I see that you use ResNet-18 for CUB, ResNet-50 for ISIC1028, and ResNet-50 for ImageNet. I think that using the same pretrained backbones architectures for all baselines on the same dataset is a consistent choice and it deserves a credit for being applied --- as long as the number of total parameters (both pretrained and trainable), including your SAE is mentioned --- I would like to see a simple table summarizing this. More importantly, in Lines 369-370, you mention that DN-CBM supports only CLIP backbone and, thus, you use their CLIP ResNet-50. However, CLIP ViT-B/16 is also supported by their framework and offer better results.
Moreover, such a backbone is the widely supported option, used almost in every work on CBM. To make the results more appealing and aligned with prior literature it is better to present a version of your Table 1 with this backbone.

**Missed concepts for ISIC2018.** I am not sure wether it is my issues or yours, and apologize for potential mistake, but I cannot verify the number of concepts (and those concepts itself) for the ISIC2018 dataset, because the file `isic_concepts.txt`, unfortunately, is not found in the repo.
However, I can confirm that, mentioned in Line 377, 278 concepts for CUB, and 2648 for ImageNet are matched.

**Table 1 with the main results.** I do acknowledge that you now have different backbones, but the different in accuracy of the baselines evaluated by you in your framework with the backbones you are using differs quite dramatically from the results reported in the original papers of your baselines and their concurrent works. To give a better understanding of such a difference, I will write a precise metrics form your baselines, and their competitors that you cite but not evaluate.
For example, LF-CBM, in their original work, results in 74.31% on CUB and 71.95% on ImageNet, while the competing work --- DN-CBM --- reports for them 67.5% on ImageNet with CLIP ResNet-50 backbone and 75.4% on ImageNet with CLIP ViT-B/16 backbone. At the same time, in their paper, DN-CBM shows 72.9% on ImageNet with CLIP ResNet-50, and 79.5% with CLIP ViT-B/16 --- both results are superior to those you have mentioned for DN-CBM on the ImageNet dataset (46.71% when NCC=5, and 57.24% with NCC=avg).
Furthermore, the work you cite, but do not compare with --- Post-hoc CBM and also their model PCBM-h --- report an accuracy of 73.6% and 80.01% on ISIC, resp., which makes them better than M-CBM --- so offer a comparison or discussion regarding this.

**Comparison in concept extraction with other baselines.** It would straighten your work if you presented an analogue of Fig. 3 for several other baselines, e.g., like Fig. 3 in the very recent preprint [2].

**Missing baselines and discussions.**  Despite evaluation very popular baselines, I think you are missing comparisons and (or at least) discussions regarding the following works [1,3,4,5,6]. All of them claim to outperform the long-standing LF-CBM to some extend, and are a decent option for comparison in the field. Summarizing a ranking those in your Table with results will further guide the community towards using better and more promising baselines.

**A minor comment.** As I see, in your codebase you implement the basic Adam optimizer with the old "coupled" weight decay. I would suggest to stick to the decoupled weight decay of ~0.1 and AdamW instead of Adam.

[1] "Incremental Residual Concept Bottleneck Models", 2024

[2] "Graph Integrated Multimodal Concept Bottleneck Model", 2025

[3] "Language in a Bottle: Language Model Guided Concept Bottlenecks for Interpretable Image Classification", 2023

[4] "Sparse Concept Bottleneck Models: Gumbel Tricks in Contrastive Learning", 2024

[5] "The Decoupling Concept Bottleneck Model", 2024

[6] "Improving Concept Alignment in Vision-Language Concept Bottleneck Models", 2024

### Questions
See the **weaknesses** part

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
The paper proposes M-CBM, a pipeline that (i) trains an SAE on a frozen backbone’s features, (ii) uses an MLLM to name SAE neurons, (iii) asks the MLLM to annotate concept presence/absence on a subset of images (partly from test), and (iv) trains a CBL and a sparse linear classifier. The authors also introduce NCC (Number of Contributing Concepts) as a decision-level sparsity metric.

### Strengths
1. Clear, modular pipeline with concrete engineering choices. Training of the CBMs is standard and well-described.  
2. NCC formalizes decision-level sparsity (concept contribution × weight) and can be computed per image/class.  
3. Results tables/curves are easy to read.

### Weaknesses
1. Methodological leakage / evaluation contamination.
The paper explicitly annotates concepts on 20–30% of the test set (per concept) “solely for a final CBL evaluation.” But concept labels on test are produced using the same pipeline (SAE activation pre-selection + MLLM conditioned on highly activating images). This risks circularity and optimistic estimates of “concept learnability/consistency,” since test annotations are themselves guided by backbone activations and neuron saliency that were derived from training. At minimum, this violates a clean separation of test supervision and model analysis and undermines the claimed generalization of the concept predictor. 
2. Reproducibility concerns due to proprietary dependencies.
Core steps depend on GPT-4.1 for naming/annotation and OpenAI embeddings for merging, with specific costs/timings reported; there is no open substitute evaluated. Reproducibility and fairness of comparison suffer because model behavior can change with closed APIs and prompts, and the paper does not provide robustness to alternative open MLLMs/embeddings.
3. Novelty over prior work is incremental.
DN-CBM already uses SAE-based concept discovery and naming; M-CBM mainly claims backbone-agnosticism and a different annotation path. NCC, while useful, is a straightforward generalization of NEC from weight-count to decision-level contribution. 
4. Baselines and fairness.
The paper drops the original (class-conditioned) VLG-CBM from main comparisons on grounds of leakage, and reports VLG-CBM{CA} as the “fair” variant—yet **ImageNet-scale VLG-CBM{CA}** is marked N/A, while M-CBM still uses thousands of GPT annotations, a heavy but different cost. The choice of backbones is also uneven (DN-CBM forced onto CLIP-RN50 across all datasets), which can disadvantage it (reported accuracy much lower than original paper) and muddle conclusions about accuracy–sparsity trade-offs. 
otential confirmation bias in concept discovery.
5. Concept naming/annotation is conditioned on highly activating examples of the SAE neuron and saliency computed from SAE decoder weights. This can bias the MLLM toward naming whatever the neuron already encodes, rather than establishing human-grounded semantics independent of the backbone. 
6. Inconsistent focus on “concept leakage” without any quantitative metric.
The authors repeatedly stress concept leakage but never evaluate it quantitatively. The proposed NCC only measures sparsity, not leakage strength or label information flow, so their claims of “leakage control” remain unsubstantiated.

### Questions
1. How do you prevent test-time contamination when concept labels on test are produced via the same activation-conditioned pipeline (SAE saliency + top-activating grids)? Please clarify precisely which model components see any information derived from test images and at what stage. 
2. Can you re-run the full pipeline without proprietary models (e.g., open-source MLLMs/embeddings) and report gaps in naming/annotation quality, costs, and final NCC-matched accuracy?
3. Provide sensitivity of your results to (a) SAE expansion factor, (b) pruning tolerance, (c) activating vs non-activating examples per neuron during naming/annotation.  
4. For fairness, could you supply ImageNet-scale VLG-CBM_{CA} with cost-matched budgets (e.g., same wall-clock as your GPT calls), or conversely report M-CBM results with the same annotation budget as VLG-CBM_{CA}?
5. On the absence of an explicit concept-leakage metric (e.g., Impurity Score).
The manuscript highlights concept leakage as a major challenge motivating M-CBM, yet the evaluation currently focuses on sparsity (NCC) rather than a dedicated leakage measure. It might strengthen the empirical section to include or discuss a quantitative leakage metric, for example, the Impurity Score or a mutual-information-based variant. Such an analysis would help clarify whether the observed improvements stem from genuine leakage mitigation or primarily from sparsity regularization.
6. On comparison with CBMs that explicitly address concept leakage.
Several works have proposed concrete strategies to quantify or reduce concept leakage. Positioning M-CBM relative to these methods, either through direct comparison or a discussion of conceptual differences, could provide valuable context and highlight the distinct contributions of your work beyond sparsity control.

### Soundness
2

### Presentation
2

### Contribution
2
