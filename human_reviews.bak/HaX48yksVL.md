# Machine Unlearning Fails to Remove Data Poisoning Attacks

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6, 6

## Abstract
We revisit the efficacy of several practical methods for approximate machine unlearning developed for large-scale deep learning. In addition to complying with data deletion requests, one often-cited potential application for unlearning methods is to remove the effects of poisoned data. We experimentally demonstrate that, while existing unlearning methods have been demonstrated to be effective in a number of settings, they fail to remove the effects of data poisoning across a variety of types of poisoning attacks (indiscriminate, targeted, and a newly-introduced Gaussian poisoning attack) and models (image classifiers and LLMs); even when granted a relatively large compute budget. In order to precisely characterize unlearning efficacy, we introduce new evaluation metrics for unlearning based on data poisoning. Our results suggest that a broader perspective, including a wider variety of evaluations, are required to avoid a false sense of confidence in machine unlearning procedures for deep learning without provable guarantees. Moreover, while unlearning methods show some signs of being useful to efficiently remove poisoned data without having to retrain, our work suggests that these methods are not yet ``ready for prime time,'' and currently provide limited benefit over retraining.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper evaluates existing machine unlearning methods based on several data poisoning attacks, including indiscriminate attacks, targeted attacks and Gaussian poisoning attacks. Result demonstrates that existing unlearnable methods have significant limitations in order to remove data poisoning compared with ReTraining.

### Strengths
Studying unlearning methods on data poisoning attacks is interesting.

The paper introduces a new Gaussian poisoning attack to evaluate machine unlearning, providing alternate auditing choice beyond MIA.

### Weaknesses
There is lack of reasonable or theoretical analyses on why Gaussian poisoning is a good unlearning metric.

The explanation on why unlearning methods fail on data poisoning attacks is not convincing. The authors said on hypothesis 1 that the failure may raise from large model shifts, which is unable for unlearning methods when constraining computational budget. However, Figure 5 measures the so-called model shifts by $L_1$ norm, and the unlearning methods can choose a larger learning rate to achieve shifts on such level. 

Furthermore, the authors said on hypothesis 2 that unlearning only with clean samples fail to counteract shifts with in the orthogonal subspace, and use gradient ascent with poison samples is not ideal. This is very confusing for me. Why defending data poisoning attacks cannot use poisoned samples? Intuitively, erasing the effect of poisoned samples requires information of poisons to some extent, which is reasonable and don’t always degrade the overall performance.

### Questions
Why $I_{indep}$ will be distributed as a standard Gaussian RV? Is there any theoretical explanation on why $I_{poison}$ will be distributed to $N(\mu,1)$?

Can you provide further explanations as to why Gaussian poisoning is a suitable evaluation metric compared to MIA? Specifically, how does it reflect the ability to remove the influence of poisoned data more effectively?

Why EUk performs so poorly on indiscriminate data poisoning?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper evaluates whether a set of existing machine unlearning methods are capable of removing adversarial data that was intentionally introduced into the training procedure of a model. This sort of attack is called data poisoning. The authors evaluate both vision and language models.

In terms of poisoning attacks, the paper looks at the traditional targeted and untargeted attacks, and introduces a novel attack which they dub ‘Gaussian poisoning’. Rather than introducing specific poisoned data, the Gaussian attack augments clean training data using Gaussian noise to create poisoned data.

The paper comes to the conclusion that none of the currently-available machine unlearning methods can successfully remove the poisoned data after training.

### Strengths
Originality:
* From what I can tell, this is the first paper to consider the effect of unlearning on data poisoning attacks
* Introduction of a novel Gaussian poisoning attack that seems to highlight inadequacies in existing unlearning methods and provide a more sensitive method of measuring the effectiveness of unlearning
* Interesting investigation into the proposed hypotheses regarding why unlearning fails in section 6.

Clarity:
* Contributions and results are laid out clearly

Significance:
* Unlearning is a very relevant problem given the introduction of GDPR, CCPA, and CPPA. Especially relevant in language models.

### Weaknesses
* Some of the figures need improved readability. E.g. It would be worthwhile to export Fig 1 as a PDF.  Figs 2,3,4,5 would benefit from larger font sizes
* Fig 3 has a mistake in that reference is made to a “dashed orange” line that does not exist – I assume this is in reference to the dashed black line.
* Fig 5 has no axis labels and is a bit confusingly described. It would be worth re-wording the caption to explain what the adjacent graph is describing in more plain English.
* It is a bit confusing that in Fig 3 orange bars are used for TPR rates but in Fig 4 directly below it is used for poison success rate.

### Questions
* It is not entirely clear to me what L349-350 is saying. Namely what does “10 epochs of retraining-from-scratch” imply?
* In Fig 3(a) why do some of the orange lines go above the “no unlearning” line? Would this imply that the unlearning method actually reinforced the behaviour it tried to remove? Ditto on Fig 4(b) and Fig 2(b)

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
3

### Summary
The paper investigates the effectiveness of current machine unlearning methods, particularly in addressing data poisoning attacks. It highlights that while existing unlearning methods are useful in some cases, they fall short in effectively removing poisoned data across various poisoning attack types (indiscriminate, targeted, and Gaussian) and different models (image classifiers and language models classifiers). Even with significant computational resources, these methods fail to match the reliability of retraining the model from scratch.

### Strengths
**Strengths of this Paper**

1. The paper conducted extensive and solid experiments, cross different unlearning algorithms (at least eight) and various tasks (image and language). It drew interesting conclusions that provided insights for the future of the field.

2. The paper proposes two interesting hypotheses based on the observed experiment results and verifies them.

3. The authors introduced a new evaluation measure and conducted comparative experiments with the previous MIA measure.

These strengths highlight the contributions of the paper to the research on machine unlearning.

### Weaknesses
1. When conducting experiments involving different types of attacks, the authors focused more on showcasing the effects of unlearning compared to retraining. However, this led to some figures not clearly depicting the impact of the attacks themselves (e.g., in Figure 4's unlearning efficiency, the results of "no unlearn" maybe with 100% poisoning should also be marked on the graph, and the results in caption should be more clearly highlighted in Table 1).

2. Based on my exploration of methods in this field, there are several highly effective unlearning algorithms, such as:     
[1] Machine Unlearning via Null Space Calibration     
[2] SALUN: Empowering Machine Unlearning via Gradient-Based Weight Saliency in Both Image Classification and Generation.   
These methods were not mentioned in the paper, even in the related work section.

3. Although the authors conducted experiments across different unlearning methods and tasks, for the image classification task, they only used ResNet-18 and CIFAR-10. This raises the question of whether the observations and conclusions are limited by the model architecture and the scale of the data.

### Questions
Same with the Weaknesses part.

For Weaknesses 2., it is recommended that the authors conduct a more comprehensive survey and categorization of unlearning work. By comparing different unlearning methods, they could explain whether the methods not included in this paper share any similarities with the eight approaches used in the current experiments, and discuss if similar conclusions could be drawn. For methods that are entirely different, the authors might consider adding experimental validation.

For Weaknesses 3., for the issues related to data and model structures, the authors could similarly provide further exploration or attempt to validate the main conclusions of the paper across different setups.

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
4

### Summary
This paper investigates the effectiveness of machine unlearning algorithms by comparing various state-of-the-art methods in the context of data poisoning attacks. Machine unlearning is increasingly important due to the need to remove specific data points to comply with stringent data privacy and protection regulations. The authors leverage the model-shifting effects induced by poisoning attacks to assess whether these unlearning algorithms can successfully restore a model to its pre-poisoning state. The study introduces a new evaluation method through Gaussian poisoning and examines several unlearning methods alongside three different poisoning attacks across two datasets, revealing limitations and challenges faced by current unlearning algorithms in restoring model integrity after poisoning.

### Strengths
1. The paper is well-written and introduces all necessary preliminary information clearly

2. A wide array of machine unlearning algorithms is studied and compared

### Weaknesses
1. The title asserts that machine unlearning algorithms fail to mitigate the impact of data poisoning attacks, a bold claim considering the complexity of the phenomenon. 


2. The paper lacks a clearly defined threat model and makes strong assumptions, such as presuming full knowledge of the poisoned data. 


3. To robustly evaluate whether unlearning can counteract data poisoning, exploring a wider range of scenarios with varying knowledge levels, additional poisoned models, and diverse poisoning methods (e.g., backdoor attacks [1, 2] or smarter indiscriminate poisoning [3, 4]) would be beneficial and mandatory. 

4. Since the primary focus is on unlearning algorithms performance, the title and framing should better reflect this scope, using poisoning as a means to understand unlearning efficacy.


5. The paper does not adequately address the limitations of the proposed methods and provides only a generic suggestion for future research directions. 


[1] Gu et al. Badnets: Evaluating backdooring attacks on deep neural networks; IEEE Access (2019).

[2] Barni et al. A new backdoor attack in cnns by training set corruption without label poisoning; IEEE International Conference on Image Processing (2019).

[3] Cina et al. The hammer and the nut: Is bilevel optimization really needed to poison linear classifiers?; International Joint Conference on Neural Networks (2021).

[4] Geiping et al. Witches' brew: Industrial scale data poisoning via gradient matching; ICLR 2021.

### Questions
Have you considered testing your approach across a broader range of poisoning methods, such as backdoor attacks or indiscriminate poisoning, to fully evaluate the efficacy of unlearning algorithms against diverse attack types?

Can you elaborate on the specific limitations of your approach and provide more targeted directions for future research that could address these challenges?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper evaluates the efficacy of several machine unlearning methods across several tasks, including one introduced by the authors: Gaussian poisoning.

### Strengths
* Overall I think the exposition of the work was good, and the authors did a good job of surveying the state of the field, and seemed to do a thorough job of experimentation. 
* The authors make a convincing point about issues with current machine unlearning evaluations, and experimentally show issues with existing methods. 
* The proposed Gaussian poisoning method is pretty intuitive and well motivated, and seems to do a good job at classifying machine unlearning.

### Weaknesses
* It seems like Sommer et al. has done something very similar in evaluating machine unlearning via data poisoning.  This might also apply to a lesser extent to Marchant et al. and Goel et al.
* The Gaussian poisoning method is intuitive, but it would really have been nice to see some more in depth analysis in a toy setting as to what extent the Gaussian samples are encoded in model weights via gradients. 
* It's nice to see your method correlates with other poisoning from Geiping et al. Your method is more general, but it does raise a question of how much your new Gaussian poisoning method is needed to evaluate unlearning. I think distinguishing the need for it here is important. 

The following aren't weaknesses that I deducted any "points" for, but things I think should be fixed:
* Figure 5 axes need to be labeled. In general Figure 5 needs to be cleaned up and explained better. Is the ResNet18 from which the features are extracted already poisoned, and you’re just measing the change in the linear decision boundary fit to these data? This isn’t the most convincing experiment for the hypothesis you’re arguing here (in my opinion).
* A couple of typos: Line 73 deleted -> delete, and Figure 2b should be “No Unlearning” but is Np Unlearning”. Might be good to check for others again as well.

### Questions
* I’m thinking out loud here, but I’m actually a bit surprised that the Gaussian poisoning method works well against convolutional architectures  like ResNet18. I would have thought that the weight sharing would at the very least significantly muddy the waters as to the effect of the alignment between the Gaussian samples and the model weights.

### Soundness
3

### Presentation
3

### Contribution
2
