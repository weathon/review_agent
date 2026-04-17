# When Priors Backfire: On the Vulnerability of Unlearnable Examples to Pretraining

- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Unlearnable Examples (UEs) serve as a data protection strategy that generates imperceptible perturbations to mislead models into learning spurious correlations instead of underlying semantics. In this paper, we uncover a fundamental vulnerability of UEs that emerges when learning starts from a pretrained model. Crucially, our empirical analysis shows that even when data are protected by carefully crafted perturbations, pretraining priors still furnish rich semantic representations that allow the model to circumvent the shortcuts introduced by UEs and capture genuine features, thereby nullifying unlearnability. To address this, we propose $\textbf{BAIT}$ ($\textbf{B}$inding $\textbf{A}$rtificial perturbations to $\textbf{I}$ncorrect $\textbf{T}$argets), a novel bi‑level optimization formulation. Specifically, the inner level aims at associating the perturbed samples with real labels to simulate standard data-label alignment, while the outer level actively disrupts this alignment by enforcing a mislabel-perturbation binding that maps samples to designated incorrect targets. This mechanism effectively overrides the semantic guidance of priors, forcing the model to rely on the injected perturbations and consequently preventing the acquisition of true semantics. Extensive experiments on standard benchmarks and multiple pretrained backbones demonstrate that BAIT effectively mitigates the influence of pretraining priors and maintains data unlearnability. Code is available at https://github.com/zhli-cs/BAIT.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces BAIT to solve the problem that existing UE methods fail when fine-tuning on clean pre-trained image-classification models by binding perturbations to incorrect target labels. Even though the proposed method seems to work only on small models (compared to the CV models mostly talked about nowadays) for image classification (which happens to be one of the easiest models to train from scratch for now), the results on common datasets look promising.

### Strengths
1. Despite some discussions in previous research, there are few solutions before to address the failure of UEs on pre-trained models.
2. The method description, experiment design, and the results seem convincing.
3. Despite some formatting issues, the writing and presentation are generally good.

### Weaknesses
1. The findings highlighted at the beginning of the paper that current UEs fail on pre-trained models do not seem to be that surprising, as this issue has been expressed in many previous studies.
2. From a practical perspective, the key to solving this problem, in my opinion, is to fine-tune a larger model using the additional data collected. Although this experiment will be more difficult, the authors should try to discuss the practical significance of this issue.
3. The paper doesn't seem to provide a thorough analysis or discussion of the relationship between pre-training data and fine-tuning data, or their impact on training, which is exactly what I'm most concerned about. If I perturb some data to prevent it from being trained and the model from learning, I'd first justify that the reduction in model performance (e.g., accuracy) is specific to the new data, and not to the original data. For example, a face recognition model might still accurately recognize the pre-trained data, but fail to recognize the perturbed data used in subsequent fine-tuning. This isn't a requirement for the author to verify this on a face recognition model, but I'd appreciate more in-depth analysis of this aspect.
4. The authors use three figures in Figure 1 to demonstrate the failure of existing UEs on pre-trained models. Figures 1a and 1b are convincing, but Figure 1c is not. Although PT and TS have significant differences in the dynamics of parameter updates, the authors appear to assume that parameter updates correspond to the model's actual learning of semantic knowledge from the data, a claim they haven't carefully justified. In fact, the relationship between the update amplitude of model parameters, training epochs and model performance is more complicated ([How far are we from true unlearnability?(ICLR'25)]).
5. The experimental part uses the Imagenet pre-trained model, but there is no experimental result on Imagenet and other larger datasets.
6. Although the authors discussed transferability, they only discussed the transferability of pre-trained models and architectures, and lacked discussion of the transferability of more important downstream tasks. Furthermore, lack of verification and discussion of the data protection ratio (i.e., the poisoning ratio).
7. Some papers also discussed the content related to UEs and pre-trained models, but the authors did not cite or mention them. [Learning the Unlearnable: Adversarial Augmentations Suppress Unlearnable Example Attacks(ICCV-AROW)] [UnSeg: One Universal Unlearnable Example Generator is Enough against All Image Segmentation(NeurIPS 2024)]
8. Some citation formatting issues

### Questions
Could the author focus on answering the first three concerns in the weakness section? It would be even better if the authors could provide some additional explanation or verification for the 4th point in weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper asks whether unlearnable examples are effective when training begins from a pretrained model. The authors find that error-minimizing noise, transferable unlearnable examples, robust error-minimizing noise, linearly separable perturbations, and a few other unlearnable examples cannot consistently reduce the test accuracy over the course of training. The authors propose optimizing class-wise, error-minimizing perturbations that make it so that when someone trains a new model starting from pretrained weights, the test accuracy can be reduced.

### Strengths
Based on Figure 1 (a) it appears that the proposed class-wise perturbations reduce the test accuracy of both pretrained and trained-from-scratch models, whereas other unlearnable example methods only reduce test accuracy for train-from-scratch models. I also agree that for unlearnable examples the case of utilizing pretrained weights is much more likely (and a more realistic scenario) than train-from-scratch, so unlearnable examples must be able to hold up to this approach (please see weaknesses 3. on how to properly see whether the current method holds up to defenses).

Starting from ImageNet pretrained models, Table 1 shows that their unlearnable examples reduce test accuracy by a wide margin. Their perturbations can prevent training (by reducing accuracy to nearly chance) on CIFAR-10, CIFAR-100, and SVHN.

Because unlearnable examples like [2] use a surrogate for a particular dataset (i.e. CIFAR-10), the authors consider a fair comparison where they re-implement [2] but using an imagenet-pretrained model as the surrogate. Here, they again find that their approach is better.

### Weaknesses
1. The "core innovation" (L180-181) is an error-minimizing noise, which is explored in [2]. Technically, the original error-minimizing noise of [2] also "binds perturbations to designated incorrect labels that are semantically different from the ground truth, deliberately steering learning away from genuine semantics." (L157-158) because [2] uses a pretrained surrogate model. By using a pretrained surrogate, the perturbations are like features of the pretrained model. 

2. I think Eq. 1 does not line up with what the authors are proposing. The authors state that "the inner level mirrors the standard UE objective by injecting perturbations that discourage the model from encoding genuine semantic" implying the optimization is over the perturbation but the inner "s.t." objective says that we seek to find a theta (model params) that minimizes the expected loss (train a model on perturbed images). The $\theta_i$ probably requires an additional "level" of optimization because those aren't fixed.

3. Line 427 "Resistance to defense strategies". The only defense that should be considered is ISS [1] : I am particularly interested in an evaluation of *just ISS* (different JPEG compression qualities must be tried: 0.9, 0.8, 0.7, etc) instead of all the other "defenses" (cutout, cutmix, mixup, etc.) because JPEG has been shown to be so effective. This paper broke a number of existing "unlearnable datasets" but this submission only considers cutout, cutmix, mixup (which can no longer be considered a defense after the ISS paper). Additionally, being a class-wise perturbation, I recommend looking at the orthogonal projection [3] defense (Section 4.4 of [3]). They argue class-wise perturbations can be easily broken.

4. Line 249: "we evaluate the effectiveness of the proposed optimization framework against pretrained backbones on standard benchmarks". The paper doesn't specify what evaluation means exactly. Do you mean that we finetune only the last linear layer of a model? If so, does the last linear layer starting from random weights or the pretrained weights? Alternatively, do you mean that we finetune all model parameters? Based on Eq. 2 it seems the answer is finetune all model parameters starting from pretrained model parameters, but this has to be stated clearly in words somewhere. Using the phrase "craft effective UEs against pretrained backbones" (L083) makes it sound like only the last layer is finetuned.

[1] Image shortcut squeezing: Countering perturbative. Liu et al. 2023

[2] Unlearnable Examples: Making Personal Data Unexploitable. Huang et al., 2021

[3] What Can We Learn from Unlearnable Datasets?. Sandoval-Segura et al., 2023

Minor: 
- Acronyms are used before they are defined: L072 has "EMN" but defined until L267. Please check if there are other instances of acronyms.

### Questions
1. If the authors believe their statement of Eq. 1 is correct, can they explain it in more detail to an undergraduate? How does it line up with the steps they propose in L176-188? Ideally follow the format of the last sentence of pg.3 of [2] where they say "this is a min-min bi-level optimization: the inner minimization...finds the bounded noise that...while the outer minimization finds the parameters that..."
2. For L313 "Reimplementeed baselines with imagenet-pretrained priors", did you use a surrogate model that was finetuned on the target dataset? For example, for generating CIFAR-10 perturbations, did you first finetune the ImageNet-pretrained model on CIFAR-10 or did you use the ImageNet-pretrained weights directly as the surrogate?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper propose a method to generate unlearnable examples for pretrained models. The pretrained model can be robust to unlearnable examples generated by previous methods since there is prior knowledge and model is able to ignore the shortcut. This paper propose to change the learning target while optimizing the perturbation, which violates the model prior and thus generates effective perturbation. The experiments are conducted on different datasets and model backbones and show consistent improvement on misguiding the model learning proess. Compared with other methods targeting for model-from-scratch, this method works better on pretrained models.

### Strengths
The proposed strategy for generating perturbations with changed target effectively attack the learning process of a pretrained model.

Some analysis on parameter updates provide insights on difference between scratched model and pretrained model.

Adequate ablations on datasets and model backbones to show the generalizability.

The comparison is made on both randomly initialized and pretrained surrogate model, both showing good improvement with previous methods

### Weaknesses
No major weaknesses in this paper. See the questions part for minors.

### Questions
In figure 3, what samples are used to test the surrogate model accuracy? Are they used in the perturbation optimization step?
Also, just to confirm, is the perturbed data in figure 3 generated with the optimized surrogate model or it is changing as the perturbation optimization goes on?

In table 1, other methods are under the setting where the surrogate model is randomly initialized, what is the result if testing the proposed method under this setting? This could be a strong proof of generalizability

### Soundness
3

### Presentation
3

### Contribution
3
