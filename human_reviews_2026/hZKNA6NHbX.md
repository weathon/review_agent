# Adversarial Attacks Leverage Interference Between Features in Superposition

- Avg Score: 2.67
- Decision: Reject
- Scores: 0, 4, 4

## Abstract
Fundamental questions remain about when and why adversarial examples arise in neural networks, with competing views characterising them either as artifacts of the irregularities in the decision landscape or as products of sensitivity to non-robust input features. In this paper, we instead argue that adversarial vulnerability can stem from *efficient* information encoding in neural networks. Specifically, we show how superposition -- where networks represent more features than they have dimensions -- creates arrangements of latent representations that adversaries can exploit. We demonstrate that adversarial perturbations leverage interference between superposed features, making attack patterns predictable from feature arrangements. Our framework provides a mechanistic explanation for two known phenomena: adversarial attack transferability between models with similar training regimes and class-specific vulnerability patterns. 
In synthetic settings with precisely controlled superposition, we establish that superposition *suffices* to create adversarial vulnerability. We then demonstrate that these findings persist in a ViT trained on CIFAR-10. These findings reveal adversarial vulnerability can be a byproduct of networks' representational compression, rather than flaws in the learning process or non-robust inputs.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
The authors explore the existence of adversarial examples in neural networks.
They propose that superposition in feature space is created through dimensionality reduction, so that several input features change the same intermediate node in similar ways.
They experiment with artificial random data vectors and a two-layer linear network, as well as with CIFAR-10 and ViT as feature extractor, for which they train another two-layer linear network.
They show that changes to the input features lead to a predictable change in the output.

### Strengths
-

### Weaknesses
My main issue with this paper is that it solely investigates adversarial samples in linear models.
It is absolutely unclear if or how the conclusions drawn this paper would transfer to realistic models which are highly non-linear.


1. It is quite difficult to follow the argumentation. The terms "input" and "feature" are used in many different ways, without properly introducing them. It seems that sometimes, the term represents data elements, and sometimes network parameters or dimensions in feature spaces:

   a) Line 53 mentions some "input features" without providing any intuition what this would be, or what they would be input to.

   b) In line 104, it is mentioned that individual neurons contribute to multiple "features", but without understanding what a "feature" is, such a claim is meaningless.

   c) Line 110 uses the term number of "latent features" but it is unclear what this refers to.

   d) Line 132 discusses "input" shape, but it is not clear whether the authors refer to "input features" or to images.

   e) Line 141 discusses "input features" (which represent single data points) into class-associated groups -- this makes no sense since each input comes only from one class and, thus, this grouping cannot be done.


2. The paper is not clear in which kinds of concepts they want to investigate:

   a) In line 81, the authors define a set of "semantically meaningful [...] concepts", but there is no intuition given that a deep network learns such concepts.

   b) In line 137, the concepts are defines to be "class concepts", which are typically given as the final fully-connected layer, but they do not correspond to any semantics.


3. There are some conceptual issues within the experiments:

   a) In line 143, the authors just sum up features from groups to determine the assigned class, which would require that feature magnitudes relate to class contribution, and that magnitudes are comparable between classes. There is no reason to believe that this would be true for features extracted from real data.

   b) In line 145, input features are sampled from a uniform distribution, which indicates that elements in there are not correlated. This is clearly not what happens when you extract "input features" from real data. The expressiveness of this experiment is, therefore, low.

   c) In line 148, the authors use cross-entropy loss to train their two-layer network. It is unclear what the task is that they train for, what are the categories that they want to compare to, or how can they use MSE loss (line 151) for the same task as CE.

   d) Figure 1 shows an "adversarial example" that has very different data restrictions as their original data. In figure 1(b), input values can be negative, while original inputs are only positive. Hence, detecting this adversarial example would be simple, making it not to be adversarial (by the definition of adversariality, line 454).

   e) Figure 1(c) shows that the adversarial sample moves **linearly** in latent feature space. For their two-layer network, such a linearity assumption is useful since the model is linear. However, there is clear evidence [Rozsa2018, Figure 5.2] that a real adversarial sample (where we perturb the input and not the features) do not move linearly in latent feature space. Particularly, Propositions 1 (line 312) and 2 (line 327) clearly do not hold for deeper, non-linear networks.

   f) In line 269, the authors mention "geometric arrangements" of models, but it is unclear what "geometry" means here. Do they mean network topologies?


4. The evaluation of the proposed theory is not convincing:

   a) The authors only use a single adversarial image generation technique (PGD) which is well-known to induce perturbations across the entire input. More localized techniques, such as Carlini&Wagner need to be explored. Also, PGD has several parameters, but the authors do not provide any details.

   b) The authors only experiment with a single image dataset, CIFAR-10, which is known to be very noisy.

   c) It is unclear why the authors performed a two-stage training process for CIFAR-10. When training the network including the bottleneck directly leads to better-separated feature spaces, so figure 4(a) would rather look like figure 2(a).

   d) Also, it is not clear what the authors have changed via PGD: the original input images (non-linear operation), or the features extracted by the network (a linear operation). Line 390 uses variable $\mathbf x$, which was used for input features before.


5. While this paper provides a theoretical analysis, where a clear mathematical formulation would be required, there are several issues within the math:

   a) In line 100, the authors use the symbol $\Theta(de^2)$ without defining any of the parameters in there.

   b) The nomenclature changes between definitions. While Definition 3 uses $f(x)$ to represent the classifier, Definition 4 uses $g_w(x)$. The authors should be consistent.
   
   c) Line 402 uses "one-hot integers" $(a,b)$ and $P$, but it is entirely unclear what this would be.


[Rozsa2018] Andras Rozsa: "Towards Robust Deep Neural Networks", PhD thesis, University of Colorado Colorado Springs, 2018

### Questions
That linearity constraints hold in linear networks is obvious, and we do not need this analysis. The most important question would be whether such assumptions also hold for deep, non-linear networks, but the authors do not investigate this at all.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper argues that the adversarial vulnerability of neural networks stems from efficient information encoding in NNs. The adversarial vulnerability emerges from the interaction between architectural constraints and data semantics, Superposition creates arrangements of latent representations that adversaries can exploit.

### Strengths
1. The paper provides a novel explanation for adversarial vulnerablility that it comes from the interference between superposed features in neural representations.
2. The superstition framework provides an explanation for adversarial attack transferability.

### Weaknesses
1. The main body is based on empirical observations, which are based on a synthetic data set. It makes sense that authors need to enable testable predictions about adversarial mechanisms, though.
2. With reduced superposition, does the robustness improve?

### Questions
1. When the features are disentangled, is there evidence that a linear classifier is less vulnerable?

### Soundness
2

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
3

### Summary
The paper argues that many adversarial examples can be explained via feature superposition: when networks pack more features than dimensions, non-orthogonal directions create interference that adversarial attacks can exploit.

In toy settings, the authors show PGD perturbations closely match theoretically optimal perturbations determined by the latent geometry, and they show an optimal input direction linking decision boundaries to input-space attacks.

The paper further shows that input correlations constrain learned geometries across seeds, which in turn predicts attack transferability between models with similar arrangements. 

Moving to a ViT on CIFAR-10 with an inserted bottleneck, higher superposition (smaller m) reduces robust accuracy and increases transferability, mirroring the toy results. 

The paper reframes some adversarial vulnerability as a result of representational compression, and not just non-robust features.

### Strengths
- The paper shows the insights of feature superposition in toy and real models. The ViT+CIFAR-10 bottleneck testbed reproduces the toy-model trends (robust accuracy down, adv transfer up as m decreases)
- The paper is easy to read and organized clearly with setups descriptions and findings.
- The source code is attached in the supplementary for reproducibility.

### Weaknesses
- It would be great to show more on the realism of the ViT setting, the paper shows forcing superposition via a post-hoc bottleneck which is informative but feels somewhat engineered; it would be stronger to show the same geometry/transfer phenomena in an unmodified backbone (or at higher resolution / vision datasets beyond CIFAR-10).
- It would help the readers if some more discussion/experiments on diverse architectures are presented on real world high res computer vision datasets.  Transfer measurements are mainly across seeds of the same family; it’d be valuable to test cross-architecture transfer to stress the “shared geometry” claim.
- Apart from the certified-training in the arithmetic task baseline, there’s limited comparison against adversarial trained model or modern certified/empirical defenses on the vision benchmark.

### Questions
- In figure 4, I was wondering how is the transfer attack percentage calculated, is it on the subset of test set where the model is correct, as the clean accuracy would decrease with decreasing m ?
- In figure 4, What would happen if bottleneck m is more than 10 for cifar10, does it remain the same as 10?

### Soundness
2

### Presentation
3

### Contribution
2
