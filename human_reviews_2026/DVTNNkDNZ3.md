# Mirages of Misalignment: How Superposition Distorts Neural Representation Geometry

- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
Neural networks trained on the same tasks achieve similar performance, but this is
not always reflected in their measured representational alignment. We propose that
this discrepancy arises from superposition or mixed selectivity, where individual
neurons represent mixtures of features. Consequently, two networks representing
an identical set of features can appear dissimilar if their neurons mix those fea-
tures differently. This may explain why higher-dimensional networks, which are
less prone to compressing mixtures of features, often show better alignment than
smaller models with greater behavioral similarity. We formalize this through an
analytic theory predicting apparent misalignment for common linear metrics like
Representational Similarity Analysis (RSA) and Linear Regression, validating it
from random projections to real neural networks. Using sparse autoencoders and
K-Means to extract disentangled features while controlling for dimensionality, we
find that feature-based alignment reveals higher similarity, particularly for early
and lower-dimensional regions. Some comparisons show decreased alignment
with disentanglement, and RSA and Linear Regression often disagree in these
cases. Simulations predict that higher RSA relative to Linear Regression in neu-
ral space indicates shared inductive biases—a pattern confirmed in real data. Our
results demonstrate that superposition and dimensionality interactions obscure the
true alignment of lower-dimensional systems, while feature-based alignment al-
lows us to more directly interrogate performance-relevant sources of misalign-
ment, with important implications for model selection.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work argues that although representations of models might be dissimilar when using linear measures of similarity 
like Representational Similarity Analysis (RSA), the models might still have captured the same latent features. 
The paper says that it might be possible to find these features using sparse autoencoders (SAEs).  

They do an experiment on latent recovery.  

They also do an experiment on whether using SAEs to make higher dimensional representations can give higher RSA and 
linear regression prediction between penultimate layers of models, between layers 1-4 of a ResNet-50 model and a brain 
and between two brains.  

They find that RSA is higher between the SAE representations than between the original model representations, 
while linear regression is about the same. 
RSA also increases for the model to brain comparison. Linear regression increases in some cases.
For brain to brain there was no change.

### Strengths
**S1:** Work on representational similarity is very relevant for both representation learning and neuroscience. 

**S2:** It is true that it is possible to have two models with the same m latent variables, but where their lower dimensional 
(dimension of n < m) representations are not related in a linear manner.

### Weaknesses
**W1:** There is a flaw in the motivation for this paper. Using measures which measure linear similarity between model 
representations are not "systematically misleading" (line 22) and do not 
"erroneously measure their representations as dissimilar" (line 53). They simply measure linear similarity of the representations. 
See question **Q1**.  


**W2:** The evaluation in the experiments is not thorough enough to make any conclusions. It seems only one seed of every model is compared.
It also seems only one seed of SAE is trained for each hyperparameter choice and only the best one is reported. See questions **Q4,Q5**.



**W3:** The paper assumes that the neural representations are a linear map of the latent variables. This assumption is too strong, 
since neural networks allow for non-linear functions. At the very least a good argument should be given for why this is reasonable.
See question **Q2**.


**W4:**  There are some unsupported claims: 
- Line 16-18 in the abstract: "We derive an analytic theory that predicts this apparent misalignment for common linear metrics like 
representational similarity analysis and linear regression". 
However, the theory does not predict misalignment, it only says that misalignment is possible.

- In line 41-43 it says "even when models are trained on identical tasks and data, comparisons consistently 
reveal a persistent “alignment ceiling,”" and this is followed by three references [1], [2], [3]. However, none of these references 
talk about an "alignment ceiling" for models trained on the same data. [1] is about how similar model representations are with the brain,
[2] considers alignment of alignment metrics, and [3] is again alignment between models and the brain. This claim is repeated with the 
same references in line 112-114. 


References:

[1] Martin Schrimpf, Jonas Kubilius, Ha Hong, Najib J Majaj, Rishi Rajalingham, Elias B Issa, Kohitij Kar, Pouya Bashivan, 
Jonathan Prescott-Roy, Franziska Geiger, et al. 
Brain-score: Which artificial neural network for object recognition is most brain-like? BioRxiv, pp. 407007, 2018.


[2] Jannis Ahlert, Thomas Klein, Felix Wichmann, and Robert Geirhos. How aligned are different alignment metrics? 
arXiv preprint arXiv:2407.07530, 2024.


[3] Zirui Chen and Michael F Bonner. Universal dimensions of visual representation. Science Advances,
11(27):eadw7697, 2025.

### Questions
Since the motivation is flawed (**W1**) and both the theoretical(**W3,W4**) and experimental(**W2**) contributions are lacking, I recommend rejection. 



**Questions:**

**Q1:** Consider two vectors: (100, 0) and (1, 0.1). 
If we use Euclidean distance to measure their difference, we get a high value, so they are dissimilar with respect to Euclidean distance. 
If we now normalize the vectors to have length 1 and then measure their difference with Euclidean distance, we get a quite small value. 
So the normalized vectors are similar with respect to Euclidean distance. This is not because the measure was "wrong" in the first case 
and the vectors are actually similar, it is because we are measuring two different things. 
In the same way, if we use representational similarity analysis (RSA) to measure the similarity between representations of two models 
and get a low value, then the representations are dissimilar with respect to RSA. If we train sparse autoencoders (SAEs) to make higher 
dimensional representations and get a higher similarity when we compare these new representations, then that does not mean that RSA 
was wrong about the original representations being dissimilar. It only means that we are now measuring a different thing. 

If you want to argue that similarity between the representations from the SAEs is the similarity we should care 
about when comparing models, then that is fine, but it does require an argument. So why is it the similarity between the representations 
from the SAEs which is important when comparing models?


**Q2:** Definition 3.1 and assumption 1 (line 151-152): 
The paper assumes that the neural representations are a linear map of the latent variables. However, neural networks 
allow for non-linear functions. Why is this a reasonable assumption to make?


**Q3:** Line 156: What is your definition of $\Vert \cdot \Vert_0 $? 


**Q4:** Line 354-356: Did you only train one seed for each choice of hyperparameters? 


**Q5:** Line 358-359: "Alignment is taken over each SAE architecture and we report the SAE with the highest mean alignment increase 
for each metric." Does this mean that you trained multiple SAEs and only report the best one? 


**Q6:** Figure 4: With respect to the linear regression plot: Does this mean that you get better linear regression results when you 
predict the brain representations from the model compared to when you predict the model representations from the brain? 







**Additional Feedback:**

typos:


Line 32: question mark in reference.


Line 50-51: "to extract the representations" -> "to extract the features" 


Line 84: question mark in reference.


Line 95-96: "Linear Regression performed with these recovered features" -> "Linear Regression performed better with these recovered features"

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper examines the issue of representational similarity. The authors assume that neural representations are linear functions of some latent variables that have a higher dimensionality than the representation itself, and that are also sparse. Under this assumption, the authors prove that even if this latent representation is the same, two different linear functions of the same latent variable might result in very different representations according to the usual similarity measures (RSA and linear regression) if the linear mappings are different. The authors then present empirical results, which demonstrate that in some cases, the similarity of latent representations (extracted using a sparse autoencoder) is higher than that of the original raw representations.

### Strengths
The paper adds a new angle to the analysis of representation similarity, namely the consideration of latent variables. This has the potential to lead to new directions and insights. Typically, latent variables are currently not considered when comparing representations, so this aspect carries a certain novelty.

### Weaknesses
The key intuition, namely that models trained on the same data must have the same latent variables, and any differences result from different superpositions of these variables, is **extremely difficult to verify empirically**. To a certain extent, it is also a tricky assumption, because in a sense it is trivial: the representations in different models are computed from the same input, therefore they are indeed only different functions of the same "latent variables" (the input). Of course, the key is then to find latent representations that have a much lower dimensionality than the input (an aspect not considered in the paper), from which the representation is computed using a very simple function (eg, linear). 

But even so, if one tries to support the intuitive assumption via finding such latent variables with eg SAEs, and the resulting latent representations are more similar, we still don't know whether this higher similarity is theoretically inevitable due to using the same inputs (in which case it is not interesting), or indeed we found something fundamentally interesting: a common latent representation distorted by superpositions. The paper does not attempt to address this question. 

However, in the brain-brain experiment, RSA did not increase, maybe because the SAE was trained on different inputs? This seems to support the first explanation.

The theoretical analysis gives very little support as well, because even if the latent variables are the same, the models learn a linear mapping that is optimal for the task at hand, and **that is not likely to be a random mapping**. And we do not know whether functionally good mappings are similar or not, the paper does not address this.

Typos: missing ref in line 86. Paragraph title should be "brain to brain" and not "brain to model" in line 385.

### Questions
Please elaborate on why your methodology is suitable for testing whether your fundamental hypothesis that common latent representations are distorted by superposition? (And the higher similarity of latent representations is not simply an artifact of using higher dimensions and using the same dataset).

Let's assume that we are able to prove that you are right and latent representations tend to be more similar on the same task than neural representations. What is the practical implication? Isn't this an epiphenomenal statement?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper investigates why alignment metrics such as RSA and linear regression often yield low similarity even when networks encode identical features.
They proposed that this could arise from superposition — the tendency of neural networks to linearly represent more features than the number of neurons available.

They derive an analytic theory predicting how this feature superposition lowers alignment scores even when networks encode identical features. They then validate the prediction using (i) synthetic linear simulations and (ii) real-network experiments (ResNet, ViT, fMRI) with and without Sparse Autoencoder (SAE) preprocessing, showing that alignment increases after “de-superposing” representations via SAEs.

### Strengths
- The motivation is clear: the authors are trying to answer why the features learned by neural networks misalign even when networks encode identical features.
- They conduct experiments both on synthetic data and real-network experiments to show that alignment increases after using SAE to de-superpose.

### Weaknesses
- Some references couldn't be compiled: line 32, line 86
- The theoretical result is elementary: they are straightforward applications of random-projection algebra. No nontrivial mathematical results or bounds are proven.
- The SAE experiments are presented as confirmation of the theory, but no independent check is provided that SAE latents truly “recover” disentangled features.
The observed alignment gain could arise from dimensionality changes, normalization, or denoising rather than from resolving superposition.

### Questions
The paper focuses on alignment measured by linear metrics such as RSA and linear regression.
However, the features learned by neural networks are not necessarily linear, and representational similarity may depend on non-linear relationships.
- Have the authors considered evaluating alignment using other metrics, for example, kernel alignment, non-linear CCA, or mutual information measures
(see Huh et al., 2024; Insulla et al., 2025, Kornblith 2019,  Williams 2021, e.t.c)

The references i mentioned are:
- Huh et al 2024. The platonic representation hypothesis
- Insulla et al 2025. Towards a Learning Theory of Representation Alignment
- Kornblith et al., 2019. Similarity of Neural Network Representations Revisited. ICML
- Williams et al 2021. Generalized shape metrics on neural representations.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies whether feature superposition lowers standard linear alignment measures between representations. It gives simple analyzable models that predict drops in RSA and linear-regression alignment due to superposition, and then shows that “demixing” with sparse autoencoders tends to increase those alignment scores in artificial neural networks. Experiments span simulations, model to model, model to brain, and brain to brain.

### Strengths
- Clear and novel central hypothesis that raises an interesting point. I would say their work is original.

- Theory is interpretable and connects directly to quantities people already report in alignment papers.

- SAE experiments are interesting and well-designed (although I believe they can be improved by providing more experimental results)

### Weaknesses
I believe the significance can be improved by providing more experimental results (more model-model comparisons, baseline comparisons). I am a bit cautious about whether to accept the SAE experimental result as strong evidence of the author's claim that superposition is the reason why we see lower than expected RSA/LR measurements. This is mainly because the authors did not check whether their SAEs actually learned meaningful features and whether the same features are found across different models.

I think addressing the following points would strengthen the significance:

- Without evaluating the features learned in SAE, the SAE cannot be claimed to have disentangled superposition. As of now, without the analysis, the author's SAE is nothing more than just some nonlinear expansion to me. An increase in alignment after a nonlinear expansion, in general, is not very surprising, unless this nonlinear expansion is really due to the disentanglement of superimposed features. I think it will be most convincing if the authors can find common SAE features across models trained on a single dataset.

- Measuring alignments in different kinds of expansion and comparing them to the SAE expansion. If SAE expansion has a greater increase in alignment than other nonlinear expansions do, then that would strengthen the authors' claim. As a simple example, the similarity can be measured at (1) the initialization of the SAE, (2) kernel space (LR can be simply replaced with kernel ridge regression; for example, the authors can just use kernel ridge regression with RBF kernel). I am not sure if there is a kernel version of RSA, but even if it does not exist, the author can use a random feature model (like the initialization of SAE). As a note, the random feature version of RBF kernel is the random Fourier feature. These other nonlinear expansions can serve as baselines. 


I should highlight that I find the hypothesis of this paper and its preliminary findings very intriguing and thought-provoking. I am willing to increase my score if my concerns can be addressed.

### Questions
1. The asymptotic RSA alignment formula (Theorem 4.1) takes the form of CKA, in terms of the maps A_a and A_b (so not in terms of features like the typical CKA). Interestingly, I think the same formula will appear in the asymptotic limit of CKA: i.e. CKA(Y_a,Y_b) -> CKA(A_a,A_b) (which is the RHS eqn. 5). If that is the case, then the authors can also claim the CKA is also affected by superposition (which is not very surprising, but still it would be a nice additional side claim). Can the authors confirm this?

2. I think the authors should clearly state what they mean by "asymptotic" limit in the main text as well, when they explain the theory.

3. There are missing citations in the main text (shown as "?" in the pdf)

4. Line 103 "A central goal in neuroscience and machine learning is to compare learned representations across different system" is an odd claim to make. 

5. The notations M and m are used interchangeably. Please stick with one for consistency. (the same applies to N and n)

6. Compression is not defined in the main text. It is only defined as N/M in the figure caption, but that definition confuses me. In Figure 2, a compression factor <1 is highlighted as "No CS" but this implies that M=N is the CS threshold, which is wrong. Instead, M=K log(N/K) should be the CS threshold as stated in the main text. Shouldn't the compression factor  be defined as K log(N/K) /M for the CS/No CS highlighted areas to make sense in Figure 2?

7. Line 385 Brain to Brain, not brain to model

8. Why in Figure 2 do the authors report R^2 for LR, but in Figure 3, the authors report Pearson R?

9. In Figure 5, I think "Increase in RSA over features" should be corrected to "Decrease in RSA over features"

### Soundness
2

### Presentation
3

### Contribution
3
