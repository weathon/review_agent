# Probing Human Visual Robustness with Neurally-Guided Deep Neural Networks

- Avg Score: 5.60
- Decision: Reject
- Scores: 8, 2, 6, 6, 6

## Abstract
Humans effortlessly navigate the visual world, yet deep neural networks (DNNs), despite excelling at many visual tasks, are surprisingly vulnerable to minor image perturbations. Past theories suggest human visual robustness arises from a representational space that evolves along the ventral visual stream (VVS) of the brain to increasingly tolerate object transformations. To test whether robustness is supported by such progression as opposed to being confined to specialized higher-order regions, we trained DNNs to align their representations with human neural responses from consecutive VVS regions during visual tasks. We demonstrate a hierarchical improvement in DNN robustness: alignment to higher-order VVS regions yields greater gains. To investigate the mechanism behind this improvement, we test a prominent hypothesis that attributes human visual robustness to the unique geometry of neural category manifolds in the VVS. We show that desirable manifold properties, specifically, smaller extent and better linear separability, emerge across the human VVS. These properties are inherited by DNNs via neural guidance and can predict their subsequent robustness gains. Further, we show that supervision from neural manifolds alone, via manifold guidance, suffices to qualitatively reproduce the hierarchical robustness improvements. Together, our results highlight the evolving VVS representational space as critical for robust visual inference, with the more linearly separable category manifolds as one potential mechanism, offering insights for building more resilient AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper explores the connection between robustness and human neural (fMRI) responses, demonstrating that guiding model representations towards neural responses improves robustness. Furthermore, the authors expand on this result, creating a supervision metric that encourages the learned manifold to have human-inspired beneficial geometric properties. The authors demonstrate that this manifold guidance also improves robustness without the need for human data.

### Strengths
This paper is an important computational exploration of an outstanding hypothesis in visual neuroscience (that robustness is an emergent property of human vision and the VVS).
The authors have demonstrated exceptional rigor with comprehensive analyses for each experiment with all appropriate controls for each of the experiments presented.
The paper goes beyond demonstrating correlation of neural guidance with robustness to introduce "Manifold Guidance" as a form of supervision that isolates this geometric constraint, confirming that representational geometry in a model is sufficient to transfer the robustness principle. A new tool in the toolbox for inducing robustness will be useful, and it will be interesting to see what other aspects of human alignment emerge from this type of supervision.
While the main results are for a single subject, the authors analyze 2 subjects in the supplemental demonstrating generality.
The authors test the results over multiple attack types demonstrating generality.
The authors clearly state (all but one of) the limitations of their experiments
I find the paper overall exceptionally well written and generally easy to follow. The discussion in particular is interesting and contextualizes the results well.

### Weaknesses
The main weakness of the paper is in overstatement of implied architecture generality of the robustness findings reported. The evidence is based solely on results from ResNet-18. While ResNet is an obvious choice, adversarial robustness is highly dependent on architecture, so the claim that this is a result applying to DNNs broadly is unsubstantiated. If the authors wish to make this claim broadly, they should demonstrate if this result holds for other DNN architectures such as VIT, bio-inspired CORnet (which the authors mention in related work), and a simpler architecture such as VGG-16. Alternatively, the authors could restrict their claims (including in the title) to be regarding only ResNet-18 
While the ordering of the robustness results (V1 to TO) is highly compelling and supports the central hypothesis, the magnitude of the quantitative improvement for manifold guidance (Figure 4) appears modest. The core finding currently rests on the trend of the means, not the statistical significance of the differences. This could be addressed by measuring variability across initialization seeds, and/or a statistical test for linear trend across regions.

Minor Points:
Formatting figures 5, 10

### Questions
I suggest the authors either:
1) Reduce the scope of the claim including in the title to claim these results only for ResNet, not DNNs generally
or
2) Test the results presented on a set of other model architectures including VIT, bio-inspired, and a simpler CNN model (no skip-connections).

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors co-train DNNs for image classification and alignment with fMRI recordings from the human ventral visual stream. They found that alignment to higher-order regions of the visual system (i.e., LO vs. V1) yielded more linearly separable neural category manifolds and, potentially as a consequence, larger adversarial robustness gains in the DNNs. These results suggest that the representational geometry shaped through transformations applied by the ventral visual system are a key reason why humans do not experience the same sensitivity to adversarial perturbations as DNNs.

### Strengths
- Lots of interesting ideas and I really like the manifold analyses. It's an elegant way of comparing human and machine vision. 
- Well done adding the weight decay controls.
- Very creative work.

### Weaknesses
- I'm struggling with the logic. Why would training on higher order layers lead to lower clean accuracy? I would actually guess the opposite. Higher order regions should encode more invariant object representations, which should train models to similarly have more tolerance to transformations as well as noise.
- No discussion about the inherent limitations of doing this with fMRI. How could that be a limiting factor for this work?
- "Past theories suggest human visual robustness arises from a
representational space that evolves along the ventral visual stream (VVS) of the
brain to increasingly tolerate object transformations" This is a weak sentence. And the follow up is at best underspecified, even for the abstract "To test whether robustness is
supported by such progression as opposed to being confined to specialized higher-
order regions..."

Minor: Line 220: "We first applied this analysis to human neural representations in each of the seven ROIs using NSD." I would use something like voxel activity patterns instead of human neural representations here. The latter is some latent property of the patterns that you hope to extract from the former.

### Questions
Questions:
- It would be good to know if any of the attack strengths are perceivable. Can you add that info?
- In the MDS: Any idea of what the dimensions capture?
- Why use regression slopes in Fig 3 but spearman in Fig 4? We should use spearman in both.
- "To further test the manifold hypothesis (DiCarlo & Cox, 2007) that identity-preserving transformations should be treated as part of the same category manifold, we enriched each category with adversarially perturbed variants of the test images, thus obtaining 100 samples in total per category, similar to previous work." The focus of the DiCarlo and Cox work was on identity-preserving transformations like translation and scaling. What about using these types of transformations? The more I think about it, the more confused I am that the authors focused on adversarial perturbations instead of naturalistic transformations for which there are a variety of popular benchmarks in computer vision (objectnet, imagenet-A, etc.).
- What about an adversarially trained model as a control?

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
This paper is motivated by the robstness of the human visual system to adversarial examples, which have been studied extensively in the computer vision literature. To better understand this phenomenon, the VVS in human brains is modeled using deep neural networks. Using fMRI data from humans looking at images, neural predictors are traubed to model the responses of regions of interest in the brain to unseen images. These predictions are then used to align the outputs of neural heads, which are added to traditional ResNet image classifiers. A related training method where representations are aligned with compact manifolds is also defined in order to test the manifold disentanglement hypothesis. Experiments show that neural guidance does in fact increase robustness, and the hierarchy of robustness when aligning models with different regions of interest aligns with that of the human visual system. Further experiments explore the learned representation spaces of the NG-models and demonstrate the effectiveness of the manifold guidance method.

### Strengths
* The results in this paper are surprising : simply training a model to mimic human brain activity confers adversarial robustness. Not only could this provide insight into the robustness of the human visual system to image perturbations, it could also inform our knowledge of machine learning robustness more generally.
* The results provide compelling evidence for the manifold disentanglement hypothesis and may be of note to cognitive scientists.
* The novel manifold guidance loss term that is defined in this paper may inspire future research in training models that are robust to image corruptions.

### Weaknesses
* The model presented doesn't engage much with the existing literature on adversarial robustness. I think that additional space should be devoted to the relationship between this paper and the adversarial training literature in the related work. It would be informative to know whether adversarially trained models also conform to the manifold disentanglement hypothesis.
* The robustness results would be more compelling if they weren't restricted to $\ell_p$ bounded adversarial corruptions. There are many types of common (non-adversarial) corruptions that humans are robust to which we want our models to be invariant to (such as those included in Imagenet-C [1]). These corruptions can also be used to define adversarial bounds which are distinct from $\ell_p$ (i.e. [2]). It's not clear from the results presented that these models are robust to anything other than $\ell_p$ imperceptible corruptions.
* Relatedly, I think it's important to note that just because a model is more robust to a specific adversarial attack, it is not necessarily more robust to adversarial attacks in general (as noted in [3]). It is possible that an informed adversary could design an attack that effectively targets neurally-guided models. I think that this should be noted as a limitation.
* Experiments were only run using fMRI data from four individuals. This low sample size may be unavoidable due to difficulties in collecting data.

[1] Hendrycks, Dan, and Thomas Dietterich. "Benchmarking neural network robustness to common corruptions and perturbations." arXiv preprint arXiv:1903.12261 (2019).

[2] Kaufmann, Max, Daniel Kang, Yi Sun, Steven Basart, Xuwang Yin, Mantas Mazeika, Akul Arora et al. "Testing robustness against unforeseen adversaries." arXiv preprint arXiv:1908.08016 (2019).

[3] Tramer, Florian, Nicholas Carlini, Wieland Brendel, and Aleksander Madry. "On adaptive attacks to adversarial example defenses." Advances in neural information processing systems 33 (2020): 1633-1645.

### Questions
* How does the performance of a DNN trained with neural guidance compare to an adversarially trained model in terms of robustness to adversarial examples?
* Are the results architecture dependent? Did you run experiments on anything other than ResNet architectures?
* Is the manifold guidance method of training useful outside of the neural guidance framework? Would you expect to see similar improvements in robustness if the manifolds for each class were not derived from neural data and instead chosen in some other (possibly unsupervised) way?
* How do your neural and manifold guidance loss terms impact the time complexity of model training?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper investigates the neural mechanisms underlying human visual robustness and explores how these can be transferred to deep neural networks (DNNs). Specifically, the authors test the hypothesis that robustness in human vision arises progressively along the ventral visual stream (VVS)—from early visual areas (V1, V2) to higher-order regions (e.g., TO, PHC)—rather than being localized solely in specialized cortical areas.
To evaluate this, the authors train DNNs to align their internal representations with human neural responses from consecutive VVS regions using data from the Natural Scenes Dataset (NSD). They demonstrate that DNNs aligned to later visual areas exhibit greater adversarial robustness than those aligned to early areas, establishing a clear hierarchy. The study further connects this effect to the geometry of category manifolds in neural representational space—showing that smaller manifold extent and greater linear separability in higher-order regions predict improved robustness. Finally, the authors introduce “manifold guidance”, aligning DNN category manifold geometry (rather than individual activations) with human neural manifolds, and find that it qualitatively reproduces the hierarchical robustness pattern.

### Strengths
Strong experimental validation: multiple controls, cross-subject replication, and attack-type diversity.


Novel conceptual framing: robustness as a byproduct of manifold geometry inherited from neural data.


Elegant theoretical integration: combines neuroscience principles with modern robustness evaluation.


High reproducibility: clear methodological exposition, use of open datasets (NSD), and code availability.


Mechanistic insight: links representational geometry to robustness, moving beyond surface-level performance metrics.

### Weaknesses
The reliance on fMRI-based predictors limits representational precision; neural alignment fidelity could improve with higher-resolution data (e.g., intracranial recordings).


The scope of tasks (primarily object classification) is limited; extending analyses to temporal or contextual visual understanding would strengthen generalization claims.


Manifold guidance, while promising, remains a coarse approximation; incorporating nonlinear manifold constraints could further clarify its efficacy.


The current framework does not model feedback and recurrence, known to be crucial for human visual robustness.

### Questions
Could the authors quantify how fMRI signal noise or predictor quality influences the strength of the robustness hierarchy?


Would the same hierarchical effect persist if neural alignment were performed using multi-subject averaged representations rather than subject-specific ROIs?


Can the manifold guidance loss be extended to self-supervised or contrastive settings to assess scalability beyond supervised classification?


How do the authors interpret the trade-off between accuracy and robustness observed in later VVS-guided models?

The authors have a few missing citations: In line 40 around the unnatural image degradations, the authors should missed Extreme Image Transforms (EITs) [Crowder et al., 2022; Malik et al., 2023, Biol Cybernetics, Malik et al., 2023, arXiv] which present a novel view of structural changes in the input images. 

When comparing previous work of Dapello et al., the authors should consider comparing their work to VOneNet [Dapello et al., 2020, NeurIPS], which fails under similar circumstances as EITs and is not widely tested on datasets like ImageNet-C [Hendrycks et al, 2019, ICLR]. 

The authors have uploaded a zip file for the code, but are encouraged to release it online on a publicly accessible platform.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies whether aligning DNNs to human neural responses with specifically-designed techniques can improve adversarial robustness. The authors proposed a dual-head architecture to train ResNet models with “neural guidance” by simultaneously optimizing classification tasks and alignment to fMRI responses from seven VVS regions. The study shows that guidance from higher-order VVS regions leads to greater robustness against adversarial attacks. To analyze and explain this phenomenon, they apply statistical modeling, Mean-Field Theoretic Manifold Analysis, and show that human neural category manifolds become less diffuse and more linearly separable along the VVS hierarchy. Finally, as opposed to point-to-point guidance, the authors introduce “manifold guidance” to match coarse geometric properties and effectively reproduce the hierarchical robustness effect.

### Strengths
- **Clear motivation and novel systematic investigation.** The paper studies and improves modes’ adversarial robustness via the alignment between DNNs and human visual systems. It provides a systematic examination of how robustness evolves across multiple consecutive human VVS regions, which is different from prior work that focuses on isolated areas such as just V1 or IT. This provides valuable insights into the hierarchical nature of visual robustness.
- **Rigorous experimental design and statistical analysis.** I find this paper especially interesting because it goes beyond only demonstrating the phenomenon but analyze how it occurs with the MFTMA analysis and further claims the importance of geometric structure with the manifold guidance experiments. The evaluation includes multiple baseline conditions, various NSD subjects (Figure 7), adversarial attacks and configurations (Figure 8), and tasks (Figure 9). 
- **Clear Presentation.** The paper is well-written with clear communication of the methodology and evaluation. Well-designed visualizations help make the concepts of the alignment between human visual system and DNNs accessible.

### Weaknesses
We thank the authors for submitting the paper to ICLR 2026! There are a few weaknesses listed below which I believe can make the paper better.
- **Limited absolute robustness gain.** While the hierarchical pattern is consistent, with results from different NSD subjects, the absolute robustness improvements of TO-guided models are small. They still show substantial vulnerability to adversarial perturbations which are imperceptible to humans. The authors already acknowledge this limitation but more further insights on future approaches which can scale to human-level robustness or if there is an existence of fundamental barriers would make the work more significant to the field. 
- **Manifold guidance limitations.** While manifold guidance reproduces the hierarchical pattern, the absolute improvements are very limited (Figure 4B). The linear approximation of manifold geometry seem to be too coarse. It would be good to see what additional information beyond manifold structure contributes to adversarial robustness in the full neural guidance condition. 
- **Architecture generalization and other related work.** The paper focuses on ResNet architectures, it is unclear whether the findings can generalize to widely-used architectures such as vision transformers, other CNN architectures, and biologically-inspired architectures such as CORnet. Also, many defense mechanisms seem related but not discussed in the paper such as biologically-inspired defenses beyond neural alignment, adversarial training which also improves manifold geometric, etc.

### Questions
- Does the manifold geometry cause robustness, or do both come from some other property of neural representations? Could you test this by techniques such as directly controlling manifold properties independent of neural guidance?
- What if you train a model to align to different VVS regions simultaneously? Would this provide the model with more information and help improve robustness? 
- There are many other robustness measures which come from human-model misalignment, such as common corruptions and out-of-distribution scenarios. Would the findings generalize to those scenarios?

### Soundness
3

### Presentation
3

### Contribution
2
