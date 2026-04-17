# Training Feature Attribution for Vision Models

- Decision: Reject
- Scores: 2, 2, 4, 6

## Abstract
Deep neural networks are often considered opaque systems, prompting the need for explainability methods to improve trust and accountability. Existing approaches typically attribute test-time predictions either to input features (e.g., pixels in an image) or to influential training examples. We argue that both perspectives should be studied jointly. This work explores training feature attribution, which links test predictions to specific regions of specific training images and thereby provides new insights into the inner workings of deep models. Our experiments on vision datasets show that training feature attribution yields fine-grained, test-specific explanations: it identifies harmful examples that drive misclassifications and reveals spurious correlations, such as patch-based shortcuts, that conventional attribution methods fail to expose.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces *Training Feature Attribution (TFA)*, a method that integrates *training data attribution (TDA)* with *feature attribution (FA)*. TDA identifies which training samples most influenced a given test prediction, while FA highlights which input features of the test sample were most relevant for that prediction. By combining these approaches, TFA identifies influential training examples *together with* the specific input regions that contributed to the model’s decision.

To accomplish this, the method uses *gradient cosine similarity* (Charpiat et al., 2019) to score training samples by their relevance to a test prediction. Since the full pipeline remains differentiable, a gradient-based feature attribution method is then applied on this measure, enabling attribution over training data.

The paper demonstrates the interpretability benefits of TFA through qualitative examples and provides quantitative results showing that the method significantly outperforms random attribution baselines.

### Strengths
- The paper clearly identifies a gap in related work and motivates why combining TDA and FA is useful.
- The writing is clear, structured, and easy to follow.
- The mathematical foundations are well connected to intuitive explanations.
- The paper offers an empirical evaluation, at least to some extent.

### Weaknesses
- **W1 A simple baseline is missing.** A trivial approach to the problem would be to use TDA to select the most relevant training samples, then pass those samples through the model to compute standard FA. While the paper’s joint TFA approach is theoretically motivated, it is unclear whether similar empirical results could be achieved with this simpler baseline. Including such a comparison would help strengthen the empirical case for TFA.
- **W2 Similar insights might be achievable with existing or trivial explanation methods.** For example, the observations in Sec. 5.2 may be attainable using counterfactual explanations, and the findings in Sec. 5.1 might also emerge from the simple baseline described in W1. If these methods provide similar insights, it raises the question of whether TFA is necessary. The paper should empirically demonstrate that TFA offers insights that go beyond those provided by existing or trivial baselines.

- **W3 The evaluation is relatively limited.** Given that the proposed method conceptually combines two existing techniques, a more comprehensive evaluation would be expected to show the added value of this integration. While the theoretical justification is clear, the empirical results do not fully reflect this strength. Moreover, much of the evaluation relies on a small number of qualitative examples, making it difficult to assess how broadly the method generalizes.

### Questions
- How does the proposed method compare to the trivial baseline proposed in W1? An explicit comparison would clarify whether the joint formulation provides meaningful advantages over the straightforward two-step approach.
- Are there scenarios or case studies where TFA clearly yields insights beyond existing explanation methods? If so, it would be valuable to empirically demonstrate this using a diverse set of baselines, covering different explanation families such as FA, TDA, counterfactual explanations, and others. This would help establish when and why TFA should be preferred in practice.

Lastly, I thank the authors for their effort and look forward to the rebuttal. If my concerns are sufficiently addressed and none of the other reviewers raises critical issues, I would be happy to reconsider and increase my score.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Summary
The paper introduces training feature attribution (TFA) that attributes a test prediction to spatial regions of specific training images by differentiating a gradient-cosine (grad-cos) training–test similarity score with respect to the training image, with qualitative demonstrations on Pascal VOC, a CIFAR-10 insertion study, and use cases on error analysis and spurious correlations.
The approach is positioned as unifying training-data attribution with feature attribution to answer “which parts of which training images most influenced this test prediction” in vision models.

Soundness
The core estimator—taking input gradients of a differentiable TDA scalar (grad-cos) w.r.t. a training image—is methodologically sound and consistent with standard saliency practices, and the analytic ridge-regression example supports the intuition at a toy scale.
However, the causal validation relies on single-step SGD updates on masked training images rather than full or multi-epoch retraining, which weakens claims about training-time causal importance. Quantitative faithfulness is limited to insertion tests without complementary deletion or broader sanity checks, leaving robustness underexplored.

Presentation
The paper is clearly written, well structured, and easy to follow, with intuitive framing and informative qualitative figures that aid understanding of TFA’s outputs. That said, plots and comparisons feel thin: key baselines (e.g., TracIn/representer as TDA backbones) are absent, and quantitative panels do not sufficiently probe stability, sensitivity, or scale, making the empirical evidence less convincing than the narrative suggests.

Contribution
The contribution is incremental: it instantiates a straightforward combination of existing training-data attribution (via grad-cos) and feature attribution (via input gradients with optional SmoothGrad) rather than proposing a novel attribution principle, estimator, or theory with guarantees. Practical value is moderate for error analysis and uncovering spurious correlations, but the lack of stronger evaluations and scalability studies limits impact at scale.

### Strengths
-Clear problem statement linking example-level and feature-level explanations, producing test-specific training-region maps that are easy to interpret qualitatively.
-Simple and implementable pipeline (grad-cos + input gradients + SmoothGrad) with demonstrations on common datasets and architectures, plus an analytic toy example for intuition.
-An insertion study on CIFAR-10 shows TFA-selected training pixels outperform random selections across multiple k, offering initial quantitative support.

### Weaknesses
-Limited novelty: primarily a composition of known TDA and FA components without a new estimator, objective, or theoretical advance.
-“Masking and retraining” claim is operationalized as a single-step update, not full retraining, weakening causal interpretation and overstating empirical claims.
-Missing comparisons to alternative TDA backbones (e.g., TracIn, representer point methods) and absent deletion tests or comprehensive sanity checks for attribution faithfulness.
-No runtime/memory profiling or scaling analysis for large datasets, leaving practical feasibility and batching strategies unclear.

### Questions
-Do the authors perform any full or multi-epoch retraining after masking salient training regions, or is the evaluation limited to single-step updates; if only the latter, can claims and terminology be aligned accordingly ?

-How does TFA scale to ImageNet-sized corpora in terms of wall-clock time and GPU memory when considering many train–test pairs, and what batching or approximate retrieval strategies are used ?

-How do results change if the TDA backbone is swapped for TracIn or representer methods, both qualitatively and via insertion/deletion metrics and stability analyses ?
-Can the authors include deletion ablations, gradient-sanity checks, and sensitivity to SmoothGrad hyperparameters to strengthen faithfulness claims ?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a training feature attribution that links test predictions to regions of specific training images.
Specifically, they combine the gradient cosine similarity TDF with a gradient-based feature attribution methods.

The paper includes a quantitative analysis of their method’s performance by comparing the loss changes after one additional training step: For this, they either mask everything apart from (i) the k most influential input pixels or (ii) k random pixels on the most influential training samples corresponding to a set of test images.
 Further, they provide evidence that their method can identify harmful examples driving misclassifications and reveal spurious correlations.

### Strengths
Overall the paper is clearly written. Further, the combination of training data attribution with feature attribution seems like a useful approach to analyse image models. The two use cases show interesting possible applications of the method.

### Weaknesses
Combining a relevant set of training images, e.g. maximally activating training samples for a certain neuron, with feature attribution methods like GradCAM is not uncommon in the explainability literature. I suggest the authors discuss the connections of their approach to such methods. E.g. [1] and [2] utilise frameworks based on max. activating training images and GradCAM saliency maps to identify spurious correlations in the training data. 

Related to the previous point, the paper could provide more ablations on their choices of (i) the TDA method and (ii) the feature attribution method. While it is a positive result, that the introduces method outperforms random saliency, this experiment could be extended to compare to simple baselines, e.g. most similar images based on LPIPS or CLIP similarities as TDA, as well as other kinds of saliency maps.

[1] Salient ImageNet, Singla and Feizi, 2022, https://arxiv.org/pdf/2110.04301

[2] Spurious Features Everywhere, Neuhaus et al, 2023, https://arxiv.org/pdf/2212.04871

### Questions
1. How does the choice of the smoothing hyperparameters (number of noise samples and standard deviation) affect your quantitative results?

2. How many of the training images included the squares in the results shown in Figure 5? 

3. In Figure 5, how would the GradCAM compare when computed on the train image instead of the test image, i.e. would it also highlight the square?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper builds on and combines two different attribution paradigms from previous work: feature attribution (FA), which explains individual predictions by highlighting the most important pixels in a given test image, and training data attribution (TDA), which identifies the training examples that most strongly influence a model’s prediction for that test image. This paper aims to unify these two components: given a test image, the proposed method not only identifies the most influential training samples (as in TDA) but also determines which regions within those training images contributed most to the prediction.

To identify relevant training samples (as in TDA), the approach follows the grad-cos attribution formulation. To additionally localize the salient regions, the method differentiates the grad-cos score w.r.t. the training image itself, which allows tracing down the impact to individual pixels, in a similar manner as a standard feature attribution.

The evaluation is primarily performed in a qualitative manner, demonstrating how given a test image, the attribution focuses on the correct object samples and regions within the training data. An additional quantitative insertion intervention shows that the reported saliency map indeed highlights regions with an above-average impact on the prediction. Finally, the paper provides two practical use cases, which demonstrate in partially synthetically crafted proof-of-concept experiments that the proposed method could be used to explain mispredictions or to detect spurious correlations.

### Strengths
Taking the gradient of the grad-cos attribution formulation with respect to the training samples — to the best of my knowledge —  represents a new direction in explainable AI, as it extends attributions beyond the test domain to the underlying training material itself. Rather than identifying which pixels in a test image impact a prediction, it allows uncovering which regions in specific training examples are most responsible for the prediction. The paper motivates the relevance of this extension by clearly presented, understandable qualitative attribution maps and two practical use cases.

The insertion-based intervention experiment serves as a sanity check, confirming that the highlighted regions indeed contain more predictive signal than random patches. Notably, the smoothing hyperparameter choice is examined in the supplementary material by an additional ablation study.

The paper is very well written and is easy to follow.

### Weaknesses
In contrast to the well-curated qualitative assessment of the proposed approach, the quantitative evaluation remains very limited: the provided insertion-based test captures only a single aspect of the attribution’s faithfulness and does not fully support the qualitative findings, e.g., the correct attribution object focus in Figure 3. In particular, given that segmentation masks are available for the Pascal VOC dataset used in the experiments, the qualitative results could have been easily complemented with quantitative localization metrics. Measuring the overlap between the attributed regions and the corresponding target masks could’ve ruled out concerns about potential sample selection biases.

As pointed out by the paper, there is some overlap in terms of goals between the proposed approach and prototypes, which also highlight relevant parts of training images. While there are differences, e.g. in that TDA can be applied to pretty much any trained neural network, I was missing a more in-depth discussion of the similarities and differences. Also it would be interesting to see if TFA applied to prototype networks lead to similar parts in the training data being identified.

Minor points:
* The mathematical notation in Sec. 3.1 and Sec. 2 is not completely consistent. This does not distract too much, but it would be nicer if it was unified.
* When the paper talks about data (e.g., l. 098), it would be good to always clarify whether training or test data is meant, since this is an important distinction here.
* The title can be read in two different ways (after having read the paper, it is of course clear, but not necessarily before). It could be misread as “Training (Feature Attribution)”, i.e. an approach that trains feature attribution methods. This could be avoided by something like “Attribution of training features…” or something like this, but then the abbreviations do not work out as nicely. Nothing critical, just something for the authors to reflect upon.

### Questions
Given the limited quantitative evaluation, I am a little bit on the fence regarding this paper. Strengthening the paper through more quantitative results would likely make me consider raising my score.

* Please provide additional quantitative evaluations to complement the strong qualitative results. Since the VOC dataset includes segmentation masks, would it be possible to state localization-oriented metrics (e.g., IoU, F1) to support the qualitative findings in Figure 3?
* Beyond the insertion test, consider adding quantitative measures that evaluate the spatial correspondence between attributed regions and target objects, as well as comparisons against random or background baselines.
* Could you discuss the relation to prototype approaches in more detail?

### Soundness
3

### Presentation
4

### Contribution
3
