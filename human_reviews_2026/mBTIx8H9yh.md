# Efficiently Attacking Memorization Scores

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 2, 2, 4

## Abstract
Influence estimation tools—such as memorization scores—are widely used to understand model behavior, attribute training data, and inform dataset curation. However, recent applications in data valuation and responsible machine learning raise the question: can these scores themselves be adversarially manipulated? In this work, we present a systematic study of the feasibility of attacking memorization-based influence estimators. We characterize attacks for producing highly memorized samples as highly sensitive queries in the regime where a trained algorithm is accurate. Our attack (calculating the pseudoinverse of the input) is practical, requiring only black-box access to model outputs and incur modest computational overhead. We empirically validate our attack across a wide suite of image classification tasks, showing that even state-of-the-art proxies are vulnerable to targeted score manipulations. In addition, we provide a theoretical analysis of the stability of memorization scores under adversarial perturbations, revealing conditions under which influence estimates are inherently fragile. Our findings highlight critical vulnerabilities in influence-based attribution and suggest the need for robust defenses. All code can be found at https://anonymous.4open.science/r/MemAttack-5413/

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This manuscript examines whether memorization-based influence estimators
including direct label-memorization scores and several common proxies can be adversarially
manipulated under black-box access. The motivation is clear and timely: in emerging
data-valuation and data-market workflows, contributors have economic incentives to inflate their
samples’ “value,” and platforms increasingly rely on memorization-style scores to allocate
credit. The authors frame the problem through a sensitivity/stability lens, arguing that
sufficiently accurate models admit high-sensitivity queries that yield large memorization scores,
and they propose a simple input-space attack (a pseudoinverse, “PINV ,” transformation) that
requires only model outputs. Empirically, across MNIST, SVHN, and CIFAR-10 with standard
convolutional architectures (e.g., VGG/ResNet/MobileNet), PINV consistently increases
memorization scores and several proxy measures while reportedly preserving test accuracy. The
threat model is presented cleanly (Figure 1) and the experimental sweep is broad. Overall, the
paper is relevant, readable, and potentially impactful for the robustness of data-valuation
systems.

### Strengths
(i) A crisp and realistic threat model centered on black-box feasibility; (ii) a simple,
reproducible attack (PINV) that does not rely on gradients or training internals; (iii) broad
experiments across datasets and architectures, including both proxy metrics and
label-memorization proper; (iv) an attempt to aggregate results with an “expected attack
advantage” (EAA) notion; and (v) an accessible exposition that keeps the reader oriented to the
data-market use case.

### Weaknesses
The central theoretical development which is the
claim that sufficient accuracy implies the existence of high-sensitivity queries leading to high
memorization would benefit from more explicit notation (specific norms, probability spaces, and
quantifiers) and a self-contained statement/proof sketch in the main text. As written, the jump
from stability lemmas to a constructive black-box adversary is intuitive rather than formal; a
short proposition that instantiates the theory via the PINV transformation would close this gap.
On the empirical side, EAA is described narratively but lacks a precise definition and uncertainty
reporting in the main text; many table entries appear rounded to ±0.00, which obscures
variability and effect sizes. Because different proxies live on different scales, cross-proxy
comparisons are hard to interpret without normalization or per-proxy analysis; adding confidence
intervals/effect sizes and showing the correlation between proxy lifts and label-memorization
lifts would strengthen the claims. The paper asserts minimal impact on test accuracy, but a small
summary figure/table in the main text (rather than only in an appendix) would make this
immediately verifiable. Finally, the ethics statement feels overly categorical given that the paper
demonstrates a practical way to game valuation: please acknowledge plausible misuse and
outline mitigations (e.g., stability-aware scoring, robust estimators, adversarial screening before
awarding credit).

### Questions
The work sits at the intersection of adversarial ML and
data valuation. Prior studies have highlighted fragility in influence-style attributions and the
manipulability of certain valuation heuristics; this manuscript narrows the focus to
memorization-based estimators and argues that even black-box adversaries can induce
meaningful lifts. If the theoretical link between accuracy/stability and manipulable
high-memorization queries is clarified and tied more explicitly to the proposed attack, the paper
would offer a useful conceptual advance alongside its practical finding: platforms that depend on
memorization-style scores are at risk without additional defenses. For example, can the author comment on whether the influence estimation can be combined with prior defense method, such as "Trustworthy Machine Learning through Data-Specific Indistinguishability" and "Preventing Verbatim Memorization in Language Models Gives a False Sense of Privacy"?

### Soundness
3

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
3

### Summary
The paper investigates the robustness of data attribution methods under adversarial manipulation. Specifically, it considers a threat model in which the data value is quantified by a sample’s memorization score, and the data provider seeks to construct examples that maximize this value, i.e. achieve higher memorization scores. To this end, the authors propose a theoretical framework for designing such attacks and develop a corresponding attack method based on the pseudoinverse of the original image matrix. They empirically evaluate how effectively the memorization score can be manipulated across three datasets and three variants of memorization metrics. Experimental results demonstrate that the proposed pseudoinverse-based attack, which aligns with their theoretical formulation, substantially outperforms existing heuristic baselines.

### Strengths
This paper studies a realistic and practical threat model where the adversary can play strategically to earn more benefits from the data company. The set-up is written clearly and easy to follow.

### Weaknesses
1. Connection between theory and the proposed algorithm (PINV). The connection between the theoretical result and the proposed algorithm PINV appears somewhat weak. In line 292, the paper notes that $q(z)=z^{-1}$ has sensitivity approaching infinity and, following the theorem, suggests that the pseudoinverse of an image (matrix) could serve as an effective attack method. However, if the underlying intuition is that high-sensitivity transformations lead to stronger attacks, then an element-wise inverse of each pixel value might seem an even more direct choice. The authors are encouraged to clarify why the pseudoinverse specifically, rather than other high-sensitivity transforms, is the theoretically motivated or empirically preferred option.
2. Clarity and soundness of Theorem 3. Theoretical Result (Theorem 3) raises several conceptual and clarity concerns.
    - Several statements in the theorem are difficult to interpret:
        - "let the dataset size $n\in\mathbb{N}$ be such that there exists $\gamma>\delta$ such that $n\geq 1/\gamma$": it is unclear what this condition is intended to ensure.
        - "There exists algorithm $\mathcal{A}$ with memorization score of at most $\delta$": the memorization score depends on both $\mathcal{A}$, $\mathbf{z}$ and $q(\mathbf{z})$. Does this statement mean that for some specific $\mathbf{z}$ and $q$, the score is bounded by $\delta$, or that bound holds universally for all $\mathbf{z}$ and $q$ under $\mathcal{A}$.
        - "$\mathcal{A}$ is $(\alpha, \beta)$-accurate ... but it must be the case that $\alpha\geq \gamma\Delta n$ and $\beta\geq \frac{\delta}{2\gamma}$" could perhaps be reformulated more cleanly, e.g. "$\mathcal{A}$ is $(\alpha,\frac{\delta}{2\gamma)$-accurate ...  $\forall\alpha\geq \gamma\Delta n and \alpha < 1/10$". Clarifying the intent of this statement would improve readability.
    - The case where $q$ s a constant function (with zero sensitivity, thus belonging to $\mathcal{Q}_{\delta}$ seems to contradict the theorem’s conclusion, as in this case $\mathrm{Pr}[|q(z)-q(P)|\geq \gamma\Delta n]=0$. The authors are encouraged to clarify how this case fits into the stated result -- it could be because I did not fully understand the results and the clarify of the theorem (e.g. the points illustrated above) might help.
3. Clarity and consistency of presentation. While the introduction and related work are clearly written, the later sections are difficult to follow due to inconsistent notation and missing definitions.
    - Inconsistent notation.
        - The function $q$ takes a dataset $z$ as input in most cases, but between lines 292–293, it appears to take an individual data point instead.
        - "$\mathcal{A}$ refers to the “model” in lines 154–158, but later denotes the “training algorithm,” for example in Theorem 3.
    - Missing details and unclear definitions.
        - The definition of the attack methods in Section 5 is not fully specified. For instance, for the main method PINV, how is the pseudoinverse defined for a three-dimensional image tensor (n_channel × width × height)?
        - Figure 2 defines the “Accuracy Game,” but this concept is not revisited later in the paper.
        - Section 3.2 introduces stability and provides two lemmas about its properties, yet these are not referenced elsewhere in the paper. The brief mention in the discussion of Theorem 3 is also unclear.
4. Practicality and detectability of the proposed attack. The proposed attack may be easily detectable, as the inversion of a natural image would likely fall far outside the natural image distribution. In the discussion section, the authors state that “manual inspection does not scale,” but it seems plausible that modern foundation models could readily distinguish natural images from their inverted counterparts. Additional discussion on this detectability issue and potential countermeasures would strengthen the practical relevance of the work.

### Questions
Please check the details in "Weaknesses"

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper targets towards the problem of manipulating data valuation score, specifically, memorization score. It gives a high-level formal definition of the problem, and then provides a theoretical analysis on the existence of 'highly influential point' under constraints. The theorem mainly state that. for very precise algorithm the memorization score is not too high globally, it has a good probability that there exists a newly introduced sample that is of high memorization score. Built on this theorem, the paper proposes a method to take an (pseudo) inversion of an image as the input for higher memorization score. Results show that over several combination of model and dataset, this method shows good result on proxies of memorization score.

### Strengths
- The paper targets an important problem of attacking data valuation method
- The paper provides a theoretical analysis on the condition of model and input so that the input has a high chance to gain high memorization score.

### Weaknesses
- Firstly, the study of manipulating data valuation algorithms is not a brand-new idea. The paper itself has mentioned a related work [Influence-based Attributions can be Manipulated](https://arxiv.org/pdf/2409.05208) [1]. Moreover, there is an ICLR 2025 Paper that is not mentioned in this work, but the setup is very similar: [Adversarial Attacks on Data Attribution](https://arxiv.org/pdf/2409.05657) [2]. In fact, the threat model of this work is very similar to [2]. The only difference I see is that this work aims to attack memorization score, while [2] targets data valuation methods such as influence function and TRAK. The authors may need to justify their novelty beyond the theoretical analysis. Moreover, for a fairer comparison, the authors may need to consider adding [2] as a baseline attack method for comparison.

- Though there exists a theoretical analysis, it appears to me that the assumption might be too strong, that it may not apply to modern machine learning framework: that a model is precise enough, robust to a set of queries, and has moderate memorization scores globally. The authors need to demonstrate that for real ML system, it is at least close to this case, Otherwise, the importance of this theorem may be weakened because the assumption is too strong.

- The designed ``PINV`` method is flawed. Firstly, it is unclear how the theoretical analysis links with the ``PINV`` method. More importantly, the method, that is to take inversion of an image is very likely to output non-sense image, with a label that is non-sense to the image. For example the original input is a cat image with a label "cat", after inversion, the image becomes almost pure noise, but then the label "cat" is still there. This is NOT adding influential points, this is actually adding garbage input that is easy to be filtered by current machine learning system. In works such as [2], adding new input is done through perturbating image input with real semantic meanings by small amount of noise that is not distinguishable through human eyes, in other words the perturbation budget is very small. This also holds true for other baseline methods the paper choose in the experiment. However, the proposed method, which takes the inversion of the input, would have very large perturbation budget  $||x - x^{-1}||$. If one just want to add points that appear influential without caring about the budget or semantic meaning, there are countless ways to do it e,g, labeling a cat as dog/labelling noise that is generated by diffusion model with labels like cat or dog.

### Questions
See weaknesses,

Moreover,
- Could you provide some visualization example of images after ``PINV``?
- Could you list the budget required for baseline and your method?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes attacks on memorization-based data valuation scores, claiming to make minimal assumptions.

### Strengths
(1) Section 3 is well-written.

(2) The authors do a good job of putting the paper together with theoretical motivations, followed up by experiments.

### Weaknesses
(1) The paper proposes implicitly to use memorization scores for data valuation, usually the literature relies on influence scores or SHAP. I am not convinced that memorization scores are the right metrics for valuation — if the training dataset only contains high-memorization samples will it be able to learn anything useful? Since high-memorization scores usually are outliers or mislabeled points.

(2) The crux of the paper is the PINV attack, but the images coming from this attack are unnatural. I don't think one has to target a particular image to create the pseudoinverse of or even create a pseudoinverse in practice -- one can minimally distort a tiny part of any image and this could perhaps be enough to memorize it.

My main reason for rejection is 1. If you could find another application for these attacks that would be better in my humble opinion.

### Questions
(1) The EMD attack is not clear. Aren't you trying to create perturbed images to begin with? so how is the distance calculated?

(2) The DF attack needs knowledge of the classifier. Are you treating this as a baseline or your own attack? Looks like the latter to me.

### Soundness
2

### Presentation
3

### Contribution
2
