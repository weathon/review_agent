# Tracing Concept Circuits to Audit and Steer Vision Transformers

- Decision: Reject
- Scores: 2, 6, 6, 2

## Abstract
Advanced vision models, e.g., Vision Transformers (ViTs), might base their decisions on spurious cues, even for correct predictions. To ensure their safe deployment in high-stakes applications, it is essential to audit ViT decision-making processes and steer them away from unsafe predictions. Traditional interpretation methods typically attribute predictions to salient pixels or neurons. However, such simplified correlations often overlook the concepts encoded in internal representations, which can be the true causes of failures. To this end, we develop an interpretation toolbox, ViSAE, to trace the concept circuits from ViT representations. These circuits enable users to (i) audit models by identifying spurious shortcuts, and (ii) steer model behaviors by amplifying or suppressing specific concepts along influential paths. Specifically, we construct a neuroscience-motivated probing suite (63K images and 16K concepts) that mirrors the human visual cortex hierarchy. Building upon the data, we train Sparse Autoencoders (SAEs) to read concepts directly from the representations of ViT and trace their causal relationships. Extensive experiments and ablation studies show that our probing suite outperforms existing counterparts by 20$\times$ in concept coverage efficiency and 28.7\% in interpretation accuracy. We demonstrate that using ViSAE, we can identify spurious decision paths, localize concepts on pixels, and diagnose the model failure modes. Furthermore, our toolbox enables model steering by editing concepts within representations, which improves worst-group accuracy on the WaterBirds dataset by 48.2%.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work introduces a intepretability toolbox entitled ViSAE. The toolbox relies on Sparse Autoencoders (SAEs) to create monesemantic concepts that can later be combined with circuit discovery methods to trace decision throughout the network. The contributions of the work are along three axes, data, algorithm, and application. For the data, a new probing dataset is introduced using GPT-5 with the motivation of having more fine-grained concepts in the probing dataset compared to datasets such as Imagenet and MS-COCO. On the algorithm side, vision language models are used to label the concepts in the SAE, and a causal algorithm is used to trace concepts to the input. On the application side, the toolbox is demonstrated for the task of auditing and steering Vision Transfomers.

### Strengths
1. An extensive benchmarking of different types of SAE for vision data.
2. Nice figures both giving an overview of the work and showcasing the toolbox.
3. A clear list of contributions.

### Weaknesses
1. The novelty of the contributions is unclear. 

(a) Data: A key contribution is the new dataset that provides more fine-grained concepts compared to the more object oriented datasets like ImageNet and MS-COCO. However, the BRODEN dataset [1] that is mentioned already provides more fine-grained concepts specifically designed for this type of analysis. It is unclear why this dataset is not used as a baseline or discussed further. Furthermore, using language models to generate fine-grained concepts is also an established practice, see for example [2]. 

(b) Algorithm: The contributions on the algorithm side are connected to the top-down concept reading and the bottom-up causal tracing. The top-down concept reading automatically labels the features of the SAE into labeled concepts, and the bottom-up causal tracing visualizes the concepts in the input space. But both of these algorithms appear to be direct applications of prior works. The top-down concept reading appear to be the CLIP-Dissect procedure [3]. The bottom-up causal tracing also seems to closely follow prior works [4, 5]. It is unclear what the methodological contributions of this work are on the algorithm side.

(c) Application: Both auditing and steering are tasks that are possible to do prior to the introduction of ViSAE [6, 7, 8].

2.The experimental evaluation is limited. Both evaluation methods and baselines are throughout the paper not suitable to demonstrate the potential quality of the introduced toolbox. The probing dataset used as baselines should be replaced with existing fine-grained concept-level probing datasets. The baselines in Table 4 could also be improved. CBM are old, and newer alternatives like Post-hoc CMBs [9] seem more relevant. SpLiCE is a good baseline, but not enough on its own, and comparing to other methods like [6, 7, 8] would strengthen the analysis. Furthermore, the visualization in Figure 5 and 6 are nice, but are only qualitative. Established quantitative measures [10] should be used to evaluated the visualizations.

- [1] Bau et al., Network Dissection: Quantifying Interpretability of Deep Visual Representations, CVPR 2017
- [2] Yang et al., Language in a Bottle: Language Model Guided Concept Bottlenecks for Interpretable Image Classification, CVPR 2023
- [3] Oikarinen et al., CLIP-Dissect: Automatic Description of Neuron Representations in Deep Vision Networks, ICLR 2023.
- [4] Commy et al., Towards Automated Circuit Discovery for Mechanistic Interpretabiliy, NeurIPS 2023
- [5] Meng et al., Locating and Editing Factual Associations in GPT, NeurIPS 2022
- [6] Dreyer et al., Mechanistic understanding and validation of large AI models with SemanticLens, Nature Machine Intelligence 2025
- [7] Wu et al., Discover and Cure: Concept-aware Mitigation of Spurious Correlation, ICML 2023
- [8] Joseph et al, Steering CLIP's vision transformer with sparse autoencoders, CVPR Workshop on Mechanistic Interpretability for Vision 2025
- [9] Yuksekgonul et al., Post-hoc Concept Bottleneck Models, ICLR 2023
- [10] Hedström et al., Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations and Beyond, JMLR 2023

### Questions
1. How does the proposed new dataset compare to [1] and [2]?
2. How does the results look if [1] and [2] are used Table 3 and 5?
3. What is the difference between the CLIP-Dissect procedure and top-down concept reading?
4. What is the methodological novelty in the the bottom-up causal tracing?
5. How does the auditing and steering results look compared to other more recent and relevant baselines?
6. The visualizations in Figure 6 look good at first glance, but when looking closer they seem inconsistent. In the "lines" example, the woman's necklace is highlighted, but the straight black line of her dress is somehow not a "line". The "wooden" example has parts of a wooden fence highlighted, why not the rest? Similar inconsistencies appear in the other examples. How can we understand these inconsistencies?

- [1] Bau et al., Network Dissection: Quantifying Interpretability of Deep Visual Representations, CVPR 2017
- [2] Yang et al., Language in a Bottle: Language Model Guided Concept Bottlenecks for Interpretable Image Classification, CVPR 2023

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on the interpretability of Vision Transformers (ViTs) and contributes with the following three angles. First, it constructs a dataset from 7 sources and annotates the images with concepts inspired by neuroscience. Second, based on the current literature of Sparse Autoencoders (SAEs), it proposes a reading algorithm to assign an SAE feature a specific concept label based on the (image, concept) probing set, and uses an existing tracing algorithm to build connection graphs between SAE concepts. Third, it conducts experiments to interpret the information flow during ViT decision making and to steer model behavior by editing concepts through SAE interventions.

### Strengths
The paper is well written and is easy to read.

The fine-grained annotation significantly helps ViT interpretability through SAEs.

With the help of concept annotation, the top-down reading algorithm avoids human labor or imprecise summaries from models for SAE feature label assignment. Further, broader concepts allow for detailed diagnosis of failure modes and bring practical benefits, as reflected in the impressive steering outcomes. Such fine-grained concepts allow for discovering new connection graphs, which brings clear future research benefits.

### Weaknesses
The main weakness of the paper I spot is the extent of technical contribution. It seems the improvement mainly comes from the GPT-5 annotation process, which is technically incremental. For example, as shown in both Table 3 and Table 5, the dataset's quality is not that important - with the correct concepts being considered, using MSCOCO alone already achieves comparable results to the carefully curated dataset.

### Questions
(1) How are the numbers in Table 1 computed? What is the definition of “Concepts Covered by Images”?

(2) How do you obtain $c_m$ for computing $P_{nm}$? Is it obtained directly through the text description?

(3) In the experiment section for auditing, during the localization of concepts on pixels, how do you pick the layer to perform such attribution? How does this choice affect the heatmap? An ablation on this would be helpful.

(4) How do you launch the experiments in Section 3.2, part (2)? It seems there is a concept set mismatch for the ablation runs because the ground-truth concepts are obtained according to Section 2.2 (from the Ours-16K set) while the SAE concepts are obtained through ablation settings. If this is the case, the numbers reported are not meaningful, and thus the comparison is not valid.

(5) In Section 3.4, the accuracy reported in text (49.6) does not match that in the table (50.3). Why are they different? Also, similar to question (3), which layer do you use to pick the concept for steering? I suppose there are multiple SAE features that have been labeled as exactly the same background concept (because the total number of SAE features is likely exceeding the total number of available concepts). If this is the case, is intervention always effective regardless of the layer index?

(6) For section 3.2/3/4, are your SAE trained on cls token? Does switching to img token make a difference?

Misc: Table 1 seems to be missing SN under data source

### Soundness
3

### Presentation
4

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
The paper introduces concept circuits for ViTs: a layer-wise, directed graph where each node is a human-interpretable concept at a specific layer and edges indicate causal influence between concepts as they compose the final prediction. The authors 1) train sparse auto encoders per transformer block, 2) auto label each SAE feature with a text concept using a CLIP-based soft-WPMI alignment over a large, synthetic concept set, 3) perform within-model hard interventions by zeroing a source concept’s SAE activation and decoding, measuring the indirect effect on target concepts at later layers, and 4) create a layer-respecting DAG whose edge weights reflect these measured effects. Empirically, they show faithfulness via targeted ablations and qualitative circuits.

### Strengths
- Tackles a core gap in ViT interpretability: concept-level, layer-wise circuits.
- Uses within-model interventions rather than correlational probes.
- Clean, modular pipeline: per-layer SAEs on the residual stream + automated concept labeling.
- Produces DAG that attempts to capture compositional flow, providing a computation path that's interpretable.
- Enables steering by suppressing spurious concepts or amplifying robust ones.
- Targeted ablations yield consistent logit drops aligned with the inferred circuits.
- Overall, a good addition to the interpretability arsenal.

### Weaknesses
- The 16K concept set is produced by GPT-5, but there’s no systematic human audit.
- Prompts are given, but generation settings (e.g., temperature/seed/determinism) aren’t specified, so exact reproduction of the concept set isn’t guaranteed. As far as I know, GPT5’s API does not provide a deterministic option at all.
- Eq 4 clearly implies that this mapping is neither injective nor surjective. This is only briefly acknowledged and not sufficiently analyzed.
- The do operator is used to define indirect effects via within-model edits, but no explicit SCM/graph is specified.
- Table 3 shows that “Ours-16k” greatly improves accuracy when the ground truth for the test split is itself GPT-5 concept annotations drawn from their proving suite. This probably privileges their vocabulary vs LAION/Google.
- The alignment step assumes CLIP’s embedding geometry reflects concept presence. 

Minor: The X-ray/MRI analogy is confusing and potentially misleading.

### Questions
- What explicit SCM (variables, structural assignments) underlies Eq 5 and 6, and under which assumptions are the reported indirect effects identified? Do you treat the forward computation graph as the causal graph? Please specify.
- How are interventions kept on-manifold when zeroing SAE features and decoding?
- How often do multiple SAE features map to the same concept, and how many concepts receive none? What is the impact on edge weights and interpretation accuracy?
- What sampling settings produced the 16K concept list? Is the list reproducible?
- How is vocabulary circularity ruled out in 3.2/Table 3?
- To ensure acyclicity, edges must be $s\to t$ with $t>s$. Can you state this explicitly?
- Do steering edits transfer beyond WaterBirds?
- Did you try not pruning the initial pool?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces ViSAE, an interpretation toolbox for auditing and steering ViTs by tracing human-understandable concept circuits within their internal representations. While the presented method addresses an important problem and has many interesting components, the novelty and technical rigor are limited.

### Strengths
- The paper addressed the important problem of AI safety by building a framework that integrates multiple components, such as data curation, SAE methods, and use cases.
- The concept dataset presented in the paper would be a great resource for future research on developing concept-based explanation methods.

### Weaknesses
- The technical novelty is limited. There is prior work published in ECCV 2024 (https://arxiv.org/abs/2407.14499) of which core technique largely overlaps with that of this paper. This prior work has not been cited or discussed.
- The probing image set is limited. I agree that ImageNet is too focused on object-level tasks, and an alternative dataset is required. However, the probing image set of 64K images, while carefully curated and outperforming existing smaller datasets, is still relatively small for training SAEs to capture the full spectrum of visual concepts, especially when compared to the vast datasets (e.g., LAION with billions of images) used for pre-training large vision models like CLIP. Usually, for LLMs, SAEs are trained on huge pre-training datasets.
- The authors measured monosemanticity by looking into "whether each basis feature of an SAE consistently activates on images of the same semantics" I am not sure if this metric is suited for measuring monosemanticity. “Monosemanticity” means that *each feature* in the sparse code represents *only one distinct concept*, even if multiple concepts are present in an image. The current metric might not fully verify if a single feature is truly disentangled from co-occurring concepts within an image, but rather if the *set of images* activating it are similar.
- The analogy is misleading. I appreciate the authors coming up with an analogy to explain how their work is different from prior work, which I believe is a good practice. However, I think the analogy is misleading in the context of the paper. X-rays and MRIs both provide spatial representations of physical structures, differing in dimensions. In contrast, pixel attribution methods provide spatial saliency maps on the input, whereas concept circuits provide causal graphs of abstract, or even non-localizable *concepts* within a model's latent space. The shift is not merely from "average" to "slice-by-slice" spatial views, but from input-level spatial attribution to a graph of abstract causal relationships, which fundamentally differs in its nature.
- The labeling of the two steps in the tracing algorithm as "Top-down concept reading" and "Bottom-up causal tracing" can be confusing. "Top-down" typically implies moving from higher-level to lower-level information, while "Bottom-up" means the reverse. In this context, "concept reading" involves mapping latent features (mid-level) to human concepts (high-level), which could be seen as an interpretation *of* features. "Causal tracing" explicitly builds connections from earlier (lower) layers to later (higher) layers and ultimately the prediction, which is indeed "bottom-up." Clarifying the precise meaning of "top-down" in "concept reading" or refining the terminology could improve clarity.

### Questions
- The process for visualization (e.g., pruning nodes) has not been mentioned. For example, in figure 5, how were the edges and nodes to show determined?
- GPT-5 was used for automatic concept annotation by GPT-5. How was the accuracy of the process ensured?

### Soundness
1

### Presentation
2

### Contribution
2
