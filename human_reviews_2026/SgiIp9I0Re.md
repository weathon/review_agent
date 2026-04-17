# JCGEL: Joint Color and Geometric Group Equivariant Convolutional Layer

- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
Translation equivariance is one of the key factors for the widespread effectiveness of convolutional neural networks (CNNs) in computer vision.
Building on this principle, group equivariant architectures have been extended beyond translations to encompass both color and geometric symmetries, which commonly arise in vision datasets.
However, despite the commuting nature of their respective group actions, color and geometry have typically been addressed in isolation by theoretical and approximately equivariant approaches.
In this paper, we introduce a \emph{joint color and geometric group equivariant convolution layer (JCGEL)} via weight sharing across the commuting group actions.
Our approach 1) improves robustness in imbalanced regimes, 2) yields factorized representations that separate color and geometric group-related factors, and 3) scales effectively to real-world datasets.
To validate these effects, we instantiate the layer within standard CNNs and evaluate across long-tailed and biased datasets, disentanglement learning benchmarks, and real-world classification tasks, where our model consistently outperforms baselines.
As a drop-in replacement for standard convolutional layers, JCGEL demonstrates generalization across a variety of vision tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a novel group convolutional network (GCNN) designed to capture variances in both hue and geometric variances. Whereas existing paper have considered these transformations separately, JCGEL designs for equivariance to both using the same convolution layer. Specifically JCGEL is designed to be equivariant to planar translation, rotation/mirroring, and hue shifts within images for classification and disentangling tasks. Furthermore a variant of JCGEL is able to enforce approximate equivariance rather than perfect equivariance used in existing baseline methods, which the authors argue is better able to accommodate real world data where perfect equivariance is difficult to observe.

Specifically, the main building blocks of JCGEL are (1) an equivariant hue and geometric lifting layer; (2) a joint geometric and hue group convolution; (3) a learnable weight for soft equivariance; (4) group batch normalization tailor to ensure equivariance to geometric and hue.

The paper showcases equivariance to rotations and hue shifts empirically and exhibits improved classification performance on imbalanced datasets and disentangling ability on standard benchmarks (3DShapes etc.). JCGEL also showcases improved classification performance on real world datasets such as CIFAR100 and Oxford Pets.

### Strengths
1. Related works is detailed and well accounted for.

2. JCGEL achieves improved generalization/classification/disentanglement results while using fewer parameters than baseline methods such as CEConv, AE-Net and standard convolutions, and the experimental set up seems sound.

3. Visualization such as T-SNE and DCI is well presented to show the interpretability benefits of designing for equivariance to both geometric and hue transformations.

4. The equivariance metric introduced is able to capture its performance in both geometric and hue shifts.

### Weaknesses
1. The motivation for the paper is a little weak, as there is no specific use case where the proposed method would fundamentally change how vision can be applied. Currently the main motivation is improved metrics.
2. The experimental section lacks larger real world datasets such as ImageNet, as well as data augmentation techniques to address either geometric or hue-shifts, such as color jitter, augmix [1], or greyscale.
3. The method uses the geometric lifting from many existing methods such as [Cohen 2016] and the hue-lifting from CEConv. The novelty stems from combining existing techniques rather than introducing methods that extend G-CNN to new transformations.


[1] Dan Hendrycks, et al. "Augmix: A Simple Data Processing Method to Improve Robustness and Uncertainty." ICLR 2020.

### Questions
1. How does the model perform if ImageNet auto augmentation is not used? How does the model perform when baseline methods are augmented with rotations, color jitter etc?

2. What is the computational overhead required  for JCGEL (in terms of runtime), I assume it increases in a similar manner as reported originally for GCNNs in [Cohen, 2016]?

3. I am a little confused about Figure 5. Are the authors able to provide classification accuracy that corresponds to the second row of T-SNE projections? The qualitative results between JCGEL and AE-Net seem very close, does this mean that color information is not important since CEConv does poorly.

4. How would JCGEL perform on larger datasets such as ImageNet. While the datasets reported are real world, they are relatively low resolution and small.

5. Is there a visualization for both geometric and hue lifting in the style of Figure 2(b). As it currently stands (a) and (b) are not contributions as such, since they can also be individually reproduced by baseline methods.

6. As JCGEL uses a similar hue-lifting as CEConv, would one expect it to perform exactly on par with CEConv in Figure 1?

A few typos:
1. Line 763: goupr -> group
2. Line 645: Roll -> Role

### Soundness
2

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
JCGEL unifies geometric and color equivariance into a single drop-in equivariant convolution layer.

This combination leads to an architecture which is equivariant to the product group $G_\text{geo}\times G_\text{color}$. This representation is useful as it imbues networks with inherent equivariance, useful for efficient learning in scenarios where train datasets might be biased in one or both of these properties. Authors introduce two varieties of their network, one being strict equivariance and another being soft equivariance, where in the soft case the degree of equivariance is learned from data.

Authors demonstrate their approach both in classical, controlled settings such as MNIST, as well as a range of more challenging, real world datasets.

Additionally, authors demonstrate advantages of their approach such as improved performance on long-tailed and biased datasets.

### Strengths
The paper is well structured and easy to read, and the maths appears correct. The central conclusions are well-supported by the choice of experiments and the experimental procedure.

The contribution of this paper is also clear, introducing a network which is equivariant over two properties: colour and the dihedral group, and this is positioned well within existing literature as it follows naturally from those papers.

### Weaknesses
1. I could not find any information on limitations of the work. In particular, combining the D4 and Hn groups would presumably lead to a higher computational overhead. Information on the extent of this in the form of training times, RAM requirements etc. would be important to make a comprehensive assessment of the utility of this work.
2. GCConv (i.e. the original group convolutional architecture proposed by Cohen et al.) is not included in any experiments. Is there a reason for this? It seems like an important baseline otherwise for geometric equivariance.
3. The definition of $\tau$ in the context of imbalance experiments is I think not defined until the appendix, making it difficult to parse what is going on upon first reading. In particular, when reference is made to specific values of $\tau$ on page 7, this is not easy to follow.
4. The bias experiments set $\tau_c=\tau_g \forall c, g$. While this is useful to expose the utility of the joint group action, in my view it would also be important to test some scenarios where this is not the case. Understanding whether combining the two components of group convolution deteriorates performance on either one of those two components would be important to understand, particularly in the case where $\tau_g \text{ or } \tau_c \rightarrow\infty$ . 
5. Strict JCGEL is omitted from the real world datasets (Table 4) and the disentangled results (Table 3). This does not appear to be addressed in the paper.

### Questions
Is there a difference between the subscripted $\tau_r$ and $\tau_g$? The use seems to switch between the appendix and the main paper.

In figure 3, are these two plots shown for the same training regime? If so, why is it the case that given JCGEL so clearly outperforms comparators on the cross entropy loss, that a greater difference is not observed in the evaluation accuracy. Is the testing set also long-tailed? Why does evaluation loss increase at all as training epoch increases? This is not clear to me.

### Soundness
3

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
5

### Summary
The authors introduce a neural network that is equivariant to hue transformations, and 90 degree rotations and reflections about the x and y axes (D_4). The network is designed according to the prescription offered in [1,2]. This problem is somewhat interesting, but as far as I understand, it isn’t particularly challenging. Experimental analysis validates the proposed solution.

[1] Cohen et al. ICML 2016
[2] Kondor et al. ICML 2018

### Strengths
**Originality.** The work appears original.

### Weaknesses
**Quality.** I rate the quality of this work as fair. The idea is somewhat interesting, but the mathematics are imprecise and should be reviewed, and the empirical analysis could be improved. Regarding the mathematics, some of the equations are quite difficult to parse. They should be made more reader friendly or paired with adequate written explanations. Regarding the experimental analysis, some experimental design decisions should be clarified (see questions/comments) and the baselines should include [2].
 
**Clarity.** The organization is fine, but the writing is not clear and should be reviewed (see notes in questions/comments and possible typos). 
 
**Significance.** I rate the significance of this work as medium. While the utility of the work is clear from the experiments, the idea that matrix groups (whether for color or geometry) can be combined is unsurprising. This has been shown empirically in both in previous geometric equivariant works (e.g. [1]) and color equivariant works (e.g. [2]). Moreover, the product of two matrix groups is itself a matrix group and theoretical works have also shown convolutional neural networks can be designed for them (e.g. [3]).
 
[1] Esteves et al. ICLR 2018
[2] Yang et al. ICLR 2025
[3] Kondor et al. ICML 2018

### Questions
**Questions/Comments**
- Line 141: it probably makes sense to also define the group action \alpha as being an action on X here
- The notation in eq 3 is inconsistent. What is i?
- Line 153: this sentence is confusing. It would be helpful to separate the discussion about group convolution and the lifting layer.
- As far as I can tell this paper considers hue equivariance but not color equivariance. This should be clarified.
- The definition of the action of H is not provided. It is described as being a subgroup of SO(3) (line 183) but also defined as commutative (eq 8). The inconsistency is noted on Line 215 but not explained, please clarify.
- Line 259. I don’t understand what is being explained here.
- Line 290: What is the reason for varying n here?
 
**Possible typos**
- Line 033: works have been → works have
- Line 039:  including graph → including graphs
- Line 132: e x → \alpha(e, x)
- Line 151: and forming the group through integer summation → (delete)
- Line 159: refferred → referred 
- Line 200: Here the hue shift is described as \mathbb{Z}^2
- Line 222: Then, composed of the above → The

### Soundness
2

### Presentation
2

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
This paper proposes a “joint color and geometric equivariant convolutional layer” by combining a standard D4 geometric group with a cyclic color-transformation group, which simply forms a direct-product symmetry. The construction reduces to weight-sharing across these commuting actions, yielding what is essentially a simple combination of existing geometric group-equivariant layers and color-equivariant layers. Experiments are conducted on synthetic equivariance tests, imbalanced MNIST, disentanglement benchmarks, and several image-classification datasets

### Strengths
1. The paper is easy to follow.
2. Experiments are conducted to demonstrate that intended equivariance is indeed achieved on synthetic dataset

### Weaknesses
1. Since the color transformation group and the $D_4$ geometric group action commute, the method effectively reduces to a direct-product construction. As a result, the paper simply combines (Cohen & Welling, 2016) and (Lengyel et al., 2023) in a straightforward and largely trivial manner: lift the input to a larger group and apply group convolution in the lifted space. There is no conceptual novelty.
2. All technical components used in the paper are standard, and the work does not provide any meaningful theoretical or architectural contribution.
3. The experimental section is also weak. Demonstrating equivariance on synthetic data only verifies that the implementation behaves as expected. The claimed gains on real datasets are unconvincing: (i) there is no discussion of the computational overhead caused by performing group convolution on a larger product group, and (ii) the empirical improvements are marginal (e.g., on CIFAR-100). In fact, publicly reported results under similar training setups substantially exceed those shown here; for example, see the benchmarks: https://openmixup.readthedocs.io/en/latest/mixup_benchmarks/Mixup_cifar.html?utm_source=chatgpt.com

### Questions
Please see the previous section

### Soundness
2

### Presentation
2

### Contribution
2
