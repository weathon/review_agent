# Pooling Image Datasets with Multiple Covariate Shift and Imbalance

- Decision: Accept (poster)
- Scores: 5, 6, 6, 8

## Abstract
Small sample sizes are common in many disciplines, 
which necessitates pooling roughly similar datasets across 
multiple sites/institutions to study weak but relevant 
associations between images and disease incidence. Such 
data often manifest shifts and imbalances in covariates 
(secondary non-imaging data). 
These issues are well-studied for classical models, but 
the ideas simply do not apply to overparameterized DNN models. 
Consequently, recent work has shown how strategies from 
fairness and invariant representation learning provides 
a meaningful starting point, but the current repertoire 
of methods remains limited to accounting for shifts/imbalances in just a couple of covariates at a time. In this paper, we show how 
viewing this problem from the perspective of Category theory 
provides a simple and effective solution that completely avoids 
elaborate multi-stage training pipelines that would otherwise be 
needed. We show the effectiveness of this approach via 
extensive experiments on real datasets. Further, we 
discuss how our style of formulation offers a unified 
perspective on at least 5+ distinct 
problem settings in vision, from self-supervised learning
to matching problems in 3D reconstruction.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
1: You are unable to assess this paper and have alerted the ACs to seek an opinion from different reviewers.

### Summary
This paper aims to provide a general harmonization tool that can handle multi-equivariance and multi-invariance with respect to the images' covariates.

### Strengths
The problem to be solved in the paper is more interesting.

### Weaknesses
I don't see any obvious Weaknesses

### Questions
None

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this study, the authors introduced the category theory to reinterpret the problem of medical data pooling.  The study firstly elaborated on the MNIST toy dataset for explaining the morphism composition imposed on the latent space, then conducted experiments on ADNI and ADCP datasets by testing on the scanner parameter. Compared to the closely-related GE method, the proposed approach achieved both low time complexity and better performance reflected on ACC and MMD measurements.

### Strengths
1. The introduction of category theory to the medical data pooling is novel and interesting.

2. Despite the complicated topic, the paper is well-written and well-structured, therefore quite easy to understand.

3. Both the runtime and quantitative results are improved over the GE method, which are quite impressive.

### Weaknesses
1. Though it is novel and interesting to introduce category theory to this problem, I am not sure if it is really necessary to do so. As far as I understand and please correct me if I am wrong, there are no additional theoretical novelties other than quoting existing definitions to the paper.

2. In terms of the scanner parameter discussed in the experiments, it is also very common to apply different kinds of normalization tricks to resolve the issue. Unfortunately, I didn't see relevant discussion or experiments presented in the paper.

### Questions
Please see the weakness section.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper explores an interesting direction for giving a defined structure to the latent space by wrapping an understanding around it derived from category theory. The motivation is to find commonality among datasets in terms of the latent space and aid in data pooling / data harmonisation, something that is needed when data collection is expensive and data from multiple sources need to be combined.  The hope is that the theory is able to define structure and invariances/equivariances between them,, and invariance to shifts in feature distributions for clinical and demographic features when dealing with image data would be easily drawn. For quantifying  the shift, minimum distance and cosine sim are employed. The paper also describes a reinterpretaion of two differentiable models from adversarial and self-supervised learning families, and walks through a demonstration of latent space operation sequences on MNIST before developing notions of losses and evaluations on data. It reads almost like a position paper with few evaluations.

### Strengths
Even though elements of the ideas exist, as elaborated in literature, this is perhaps a first formulation of latent space equivalences in terms of category theory. The formulation expressed, developed and proved mathematically appears sound on a few reads of the text.

### Weaknesses
Both classes of functor mappings are linear or affine, given properties of identity, distributivity and commutativity. The development of these ideas for supra-linear mappings may present an uncertainty at the fundamental that is unknown at the moment.

### Questions
While converting to measurable spaces, does the scale of the Borel interval have an effect? That's something that may merit digging into. 

While evaluating, accuracy seems just an indicator and not really a measure of distributional discrepancy like mmd and adv. Do I understand it right?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Authors present a Category Theory-based formalism for covariate shifts
and a method derived from this formalism for pooling data with
covariate shifts for simple tasks. Images are mapped to a latent space
- using AE but I guess this can be other things - and covariate shifts
are modeled as transformations in the latent space. Invariance to
some covariate shifts, e.g., using different acquisition devices, are
modeled by enforcing similar latent space
representations. Equivariance is obtained by enforcing a chosen
transformation. Experimens show that the resulting model can retain
accuracy while removing differences between samples with covariate
shift in the latent space in cases where invariance is
desired. When equivariance is desired, authors show that the latent
space transformations are also good.

### Strengths
1. The general formalism Category Theory provides is elegant and
   contains a class of algorithms as special cases. This has the
   potential of inventing new directions for data pooling and
   addressing covariate shifts.
2. Authors provide a great introduction to the theory and maps the
   relevant aspects to practical problems pertinent to covariate
   shifts observed in application in neuroimaging.
3. Experimental results are motivating. Authors compare with some
   recent work to convince us that the model proposed is indeed
   useful.
4. The article is not necessarily providing a novel technique but
   rather discusses a theory that views already applied methods as a
   special case and thus allows further generalization. For instance,
   https://link.springer.com/chapter/10.1007/978-3-031-16431-6_1
   applied similar ideas for longitudinal imaging data.

### Weaknesses
1. The current method requires overlapping covariate values between
   data sets coming from different centers. Equation (9) seems crucial
   for the methods to work. However, until (9) the reader is left with
   the opinion that Category Theory would magically work for
   non-overlapping covariate values between different data sets. This
   should be clearly discussed early on in the article to avoid any
   confusions and - to be honest - false hope. Given two data sets
   whose covariate values do not overlap, the current method may not
   work at all. 
2. The emphasis on couterfactuals is rather questionable. There is no
   decoupling of the effects of different covariates on data. In other
   words, a transformation in the latent space may correspond to
   changes in multiple covariates. A well curated data set may avoid
   this problem, however, generally this will not hold. Therefore, it
   is difficult to discuss about counterfactuals and even
   interventions. The latent space transformations will not be able to
   account a missing causal model linking covariates, observed
   images and labels. I recommend authors to reconsider claims
   regarding counterfactuals.
3. Figure 5 is not explained at all. This seems like an important
   figure and requires further explanation in the text.
4. I find the claim that this approach simplifies recent
   works in the literature not substantiated. Furthermore, without
   explicit disentanglement nor the capability to do interventions, I
   am not sure to what authors may be referring.

### Questions
1. I encourage authors to reconsider their claims, especially
   concerning counterfactuals and models requirements of overlapping
   covariate values between different sets.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
