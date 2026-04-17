# Semantic Robustness of Deep Neural Networks in Ophthalmology: A Case Study with Colour Fundus Imaging

- Decision: Reject
- Scores: 2, 6, 0

## Abstract
Despite the success of Deep Neural Networks (DNNs) in ophthalmic tasks, their robustness in real-world clinical settings remains uncertain.
This paper presents a case study on the semantic robustness of DNN models for retinal imaging. We first introduce a novel optimisation algorithm, DIRECT-LSR, to identify worst-case robustness against clinically relevant semantic perturbations, including geometric transformation, illumination distortion, and motion blur. Our method provides a reliable lower bound with theoretical guarantees, enabling a practical black-box robustness validation approach. Evaluating various commonly used DNN models on colour fundus photograph datasets, we demonstrate their vulnerability to semantic perturbations, particularly geometric transformations that drastically reduce model accuracy despite preserving clinically relevant features. As a secondary contribution, we show that a randomised data augmentation strategy can serve as an effective and accessible defence mechanism to improve models' reliability. However, since the performance gaps between clean and perturbed images persist, our results also highlight the need for more advanced defences in future work, offering insights for developing more reliable artificial intelligence systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the robustness of retinal imaging models to "semantic" corruptions such as geometric distortions, illumination changes, and motion blur (in contrast to adversarial corruptions). These corruptions are common in datasets of retinal images due to practical issues in data collection.
The authors parameterize the corruptions and present an algorithm (DIRECT-LSR) which can find distortion parameter values that are provably close to optimal, in the sense that corruptions will maximally affect classifier performance. Indeed, the found corruptions are shown to degrade performance stronger than corruptions found by PGD-like gradient-based methods. The authors then show that augmenting the training data with such corruptions improves model robustness with respect to these semantic corruptions.

### Strengths
- The investigated problem is practically relevant, for the applied field of retinal imaging.
- The parametrization of the corruptions seems reasonable and is explained clearly.
- The proposed algorithm, DIRECT-LSR, could be useful in other contexts as well.
- The investigation covers multiple models and datasets.

### Weaknesses
- The scope of the paper is very narrow. I see the potential value of this analysis to the medical field, specifically the subfield working on retina imaging, but for ICLR, this somewhat lacks generality. One would have to pitch the DIRECT-LSR algorithm as a general method, and I could see myself accepting such a paper, but this is not what the paper does.
- The presentation of the paper should be improved:
	- Figure 3 is a bit weird because the lines all overlap, so most of the colors never even show up - probably, a logarithmic y-axis would have been advisable.
	- Figure 4 is poorly designed, I'd recommend to focus on one dataset, then put the rest in the appendix as separate figures.
	- Table 2 is hard to read, it includes a few irrelevant models (as in, models that never perform best), and no confidence intervals or any notion of uncertainty. I would make a selection of models, then have the full table in the appendix. Likewise, I would remove the $L_{min} > 0$ rows and present them in an extra appendix table.
- The related work section is not very convincing, I would have liked to see works more closely connected to robustness of retinal imaging models rather than e.g. an arbitrary autonomous driving paper. 
- It seems like train / test splits were not conducted properly. If this is indeed the case, it would be a weakness.
- The four datasets are not sufficiently characterized. How large are they? What is the state-of-the-art classification performance on the datasets?
- The evaluation of human experts was not done perfectly. Ideally, human experts should have been asked to perform a classification task "blindly", in a true nAFC fashion, i.e. without access to the true labels, rather than just stating whether they think crucial features for the true class are visible or not. This would put model performance in perspective much better.
- The limits of all corruption parameters are neither justified nor visualized, making it hard to assess whether they are reasonable (e.g. there are levels of motion blur that would be absurd to expect in retina images). 
- The analysis is a bit thin, I would have liked to see e.g. whether the different optimization procedures (DIRECT-LSR, Bayes, random search) find very different optima, whether your attacks lead models to favor a certain class, etc.
- I realize this is a somewhat unfair criticism, but I find it hard to believe that random search outperforms PGD-style gradient-based optimization for the task of finding optimal corruptions. I must imagine that this is indicative of sub-par implementation or poor hyperparameters. Can the authors comment on this? (Maybe the random search effectively covers the entire parameter space?)
- Training the models in table 2 with exactly the same setup does not seem very principled to me. Different training setups might be optimal for different models, so I would find it more convincing if all models were trained in a way that yields maximum performance (up to the limit of the respective architecture and model size). Another reasonable decision would have been to compute-match models. I would have also liked to see reference performance values on these tasks from literature.
- Again a somewhat unfair point, but the paper lacks certain shibboleths that one would expect from groups that work in this field. For example, the phrase "out-of-distribution robustness" does not appear even once, although this is exactly what this paper is about at its core. I have tried to down-weight this point in my assessment, but it does not instill confidence.

Overall, I feel conflicted about this paper. In principle, I see the value of the idea of finding provably difficult points in bounded parameter spaces for parametric corruptions, and I can imagine that evaluating robustness to such corruptions is relevant in the retinal imaging field. The biggest argument in favor of the paper is that the proposed algorithm could be useful in other settings. But the paper itself suffers from poor presentation, and there are some "smells" (such as the train-test-split issue, model comparison on equal footing, bad PGD performance, etc). The paper also has a very narrow topic scope and many questions remain open. Based on the related work, I am also lacking perspective on e.g. what models are considered gold standard for these tasks, and how robustness has been evaluated in the literature. I'm submitting an initial rating of 2 for now, but I am curious about other reviewers' thoughts and generally willing to increase this score, provided that presentation is improved and the paper is championed by someone else.

### Questions
- I would find the paper more valuable if the idea was to implement a benchmark, where people submit retinal imaging models, and maximally strong "semantic adversarials" are created specifically for these models, eventually leading to a public leaderboard of the most robust retinal imaging models. Do you plan on implementing this? 
- What was the train- / test-split of the datasets? It reads in line 440 as if the dataset was not split, and validation was done on a random subset of the training set.
- What is the Schwefel function? This should probably explained in at least one sentence.

### Soundness
3

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
3

### Summary
This paper presents a thorough and well-motivated framework for evaluating the semantic robustness of deep neural networks (DNNs) in retinal image diagnosis. Rather than relying solely on pixel-level adversarial perturbations, the authors propose a semantic perturbation framework that operates at the level of medically meaningful transformations (e.g., illumination, lesion intensity, vessel sharpness). The goal is to assess whether retinal disease classifiers remain stable under clinically relevant but semantically preserving changes. This is a timely and practically significant contribution to the intersection of adversarial robustness, explainable AI, and medical imaging, with strong potential implications for clinical reliability and model validation.

### Strengths
1. The work addresses a real and pressing gap: most medical AI robustness studies focus on pixel-level noise or domain shifts, ignoring semantic perturbations aligned with clinical reasoning. The proposed semantic perturbations are interpretable, controllable, and grounded in medical semantics, which enhances both transparency and clinical adoption potential.
2. The framework systematically defines semantic dimensions and corresponding transformation operators. The perturbations are implemented in a way that maintains plausible clinical realism, avoiding synthetic artifacts.
3. Evaluation across multiple datasets and models demonstrates the framework’s generality. The correlation analysis between semantic robustness and adversarial robustness is particularly insightful—it shows they are distinct but complementary.

### Weaknesses
1. The framework is empirical and descriptive; it lacks formal definitions or theoretical guarantees of semantic robustness (e.g., invariance under a semantic transformation group). A more formal link to robustness theory (e.g., Lipschitz continuity under semantic metrics) would strengthen its academic rigor.
2. The chosen perturbations, while clinically plausible, are manually curated and limited to a few dimensions. There is no discussion on how to generalize or learn semantic perturbations automatically (e.g., via generative models or disentangled representations).
3. Experiments focus on binary diabetic retinopathy grading. It remains unclear how well the framework generalizes to multi-class or multi-label medical tasks (e.g., glaucoma, AMD). The lack of external validation on unseen imaging modalities (e.g., OCT) limits generalizability.

### Questions
1. How do you define the boundary between semantic and non-semantic perturbations, especially when pixel-level changes may indirectly alter semantic meaning?
2. Could your semantic perturbation framework be adapted for unsupervised discovery of semantic factors using disentangled or generative representations?
3. Do you have any insights into why ViT architectures (if included) appear more semantically stable than CNNs, or vice versa?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper proposes a framework for evaluating semantic robustness of deep neural networks in color fundus photography. The authors focus on three types of distortions, namely geometric transformation, illumination changes, and motion blur. The paper tries to utilize a non-gradient optimization method based on the DIRECT algorithm.

This paper should be rejected because there are significant shortcomings including the following points:

1- The proposed algorithm (DIRECT-LSR) is a simplistic modification to the original DIRECT algorithm.

2- The paper does not provide any justification in using the DIRECT-LSR to evaluate robustness. In fact, the authors do not formally define what robustness means in the context used in this paper.

3- The paper claims be able to evaluate robustness of neural networks to semantic perturbations and changes. However, the experiments simply use geometric transformations, illumination changes, and motion blur as examples of semantic perturbations. These are not in fact changing the semantic context of the images, and thus are practically irrelevant. Again, this is due to the fact that the paper doesn't clearly and formally define robustness and semantic sensitivity of neural networks. 

4- The paper does not provide any comparison with other state-of-the-art algorithms designed for assessing robustness.

5- Evaluation against human expert assessments and protocols used for this evaluation is not clearly discussed.

6- This paper is an application paper, focused on a narrow domain (retina fundus images). Its applicability beyond this domain is questionable.

### Strengths
The paper attempts to evaluate the robustness of neural networks to semantic perturbations.

### Weaknesses
The paper suffers from many weaknesses with major shortcomings listed below:
1) Lack of novelty. The paper simply applies a least-squar-regression to the DIRECT algorithm.
2) Lack of motivation behind the use of this optimization algorithm. Why not evaluate with a gradient-based approach?
3) Lack of comparative evaluations against the state-of-the-art.
4) Lack domain generalizability. It is unclear how this method can be applied beyond fundus images.
5) Insufficient experimentation. The paper simply uses three types of image manipulation. Would this method work for adversarial attaches? Or actual semantic perturbations, like content manipulation? The paper also only evaluates six neural nets for robustness. The claim that the proposed method can evaluate neural network robustness should design a more comprehensive framework for evaluating major neural networks' robustness.

### Questions
None.

### Soundness
1

### Presentation
1

### Contribution
1
