# Brain-inspired $L_p$-Convolution benefits large kernels and aligns better with visual cortex

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 3

## Abstract
Convolutional Neural Networks (CNNs) have profoundly influenced the field of computer vision, drawing significant inspiration from the visual processing mechanisms inherent in the brain. Despite sharing fundamental structural and representational similarities with the biological visual system, differences in local connectivity patterns within CNNs open up an interesting area to explore. In this work, we explore whether integrating biologically observed receptive fields (RFs) can enhance model performance and foster alignment with brain representations. We introduce a novel methodology, termed $L_p$-convolution, which employs the multivariate $L_p$-generalized normal distribution as an adaptable $L_p$-masks, to reconcile disparities between artificial and biological RFs. $L_p$-masks finds the optimal RFs through task-dependent adaptation of conformation such as distortion, scale, and rotation. This allows $L_p$-convolution to excel in tasks that require flexible RF shapes, including not only square-shaped regular RFs but also horizontal and vertical ones. Furthermore, we demonstrate that $L_p$-convolution with biological RFs significantly enhances the performance of large kernel CNNs possibly by introducing structured sparsity inspired by $L_p$-generalized normal distribution in convolution. Lastly, we present that neural representations of CNNs align more closely with the visual cortex when -convolution is close to biological RFs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper primarily investigates variations in local connectivity patterns within CNNs, examining whether incorporating biologically inspired connectivity structures can improve model performance and increase alignment with brain representations. Specifically, the authors introduce Lp-convolution, which utilizes the multivariate p-generalized normal distribution (MPND). The proposed adaptable Lp-masks aim to bridge the gap between artificial and biological connectivity patterns, finding optimal configurations through task-based adaptation to enable strong performance in tasks requiring flexible receptive field shapes.

### Strengths
* Experimental results indicate that CNN neural representations exhibit a stronger alignment with the visual cortex when the Lp-mask shape approximates a Gaussian distribution. 

* Testing the conformational adaptability of Lp-masks in the Sudoku challenge yielded interesting results, highlighting the flexibility of this approach.

### Weaknesses
* Consistency in terminology would improve clarity; alternating between “Lp-convolution” and “Lp-mask” can be confusing. Using a single term throughout would make the concepts easier to follow. 

* The mention of Vision Transformers (ViTs) in the introduction feels tenuous, as they are not included in subsequent experiments, nor are they closely related to the main theme of the paper. 

* In lines 108-110, where it is stated that “CNNs have rectangular, dense, and uniformly distributed connections, as opposed to the circular, sparse, and normally distributed connections in biological neurons,” this description would benefit from supporting references regarding the shapes of receptive fields in biological neurons. It’s also worth questioning whether this statement accurately characterizes CNN weights, as CNNs trained to model retinal ganglion cells, for instance, have demonstrated sparse weight patterns ([1]-[5]). 

* Lines 137-138 mention that “we optimized parameters of p and σ in MPND (Fig.1e, Eq.1)...” However, Eq.1 and the text do not define σ. It’s also recommended that the authors confirm Eq.1’s form by referencing the standard expression of a multivariate Gaussian function. 

* Integrating Lp-masks in CNNs does not appear to significantly improve recognition accuracy across datasets. Comparing this approach to ViTs, it’s unclear if it achieves current state-of-the-art performance. 

* The justification for using large, sparse kernels feels somewhat weak. Aside from achieving marginal improvements in RSM alignment with the visual cortex, it’s unclear how this approach benefits contemporary computer vision tasks. 


References:

[1] Maheswaranathan, Niru, et al. "Deep learning models reveal internal structure and diverse computations in the retina under natural scenes." BioRxiv (2018): 340943. 

[2] Tanaka, Hidenori, et al. "From deep learning to mechanistic understanding in neuroscience: the structure of retinal prediction."  Advances in neural information processing systems 32 (2019).

[3] Lindsey, Jack, et al. "A unified theory of early visual representations from retina to cortex through anatomically constrained deep CNNs."  arXiv preprint arXiv:1901.00945 (2019).

[4] Yan, Qi, et al. "Revealing fine structures of the retinal receptive field by deep-learning networks."  IEEE transactions on cybernetics 52.1 (2020): 39-50.

[5] Zheng, Yajing, et al. "Unraveling neural coding of dynamic natural visual scenes via convolutional recurrent neural networks." Patterns 2.10 (2021).

### Questions
* In the authors' claim regarding "Lp-convolution with biological constraint," specifically the "Gaussian structured sparsity," what theoretical and empirical evidence supports this biological constraint? 

* Across various experiments, since ppp is a learnable parameter, what typical values does it converge to, and are there any observable trends or variations across different datasets? Could the authors interpret these findings in relation to biological insights?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Lp-convolution, a novel approach to convolutional neural networks (CNNs) inspired by biological visual processing. The work addresses fundamental differences between artificial and biological visual systems: while traditional CNNs employ rectangular, dense, and uniform connectivity patterns, biological visual systems feature circular, sparse, and normally distributed connections. Additionally, the paper tackles the longstanding challenge that large kernel sizes in CNNs typically don't improve performance despite increased parameters.

The key innovation is the introduction of Lp-convolution, which uses multivariate p-generalized normal distribution (MPND) to bridge these biological-artificial differences. The method implements trainable "Lp-masks" that can adapt their shape through parameters, enabling flexible receptive field shapes that better match biological patterns. Technically, this is achieved by applying channel-wise Lp-masks that overlay onto convolutional kernels, with shape parameters that can be trained for task-dependent adaptation.

The authors demonstrate several significant findings: Lp-convolution improves the performance of CNNs with large kernels, with optimal results achieved when the initial p parameter approaches 2 (matching biological Gaussian distribution). Moreover, neural representations show better alignment with the visual cortex when connectivity patterns are more biologically plausible. 

The practical impact of this work is threefold: it enables effective utilization of larger kernels in CNNs, achieves more biologically plausible artificial neural networks, and maintains compatibility with existing CNN architectures.

### Strengths
The paper is overall well-written, with sections that tell a clear, sequential story. The usage of bold characters to highlight important parts is particularly appreciated. This is a very strong contribution across multiple aspects:

* Connectivity patterns as inductive biases are largely unexplored within this community. The biological inspiration effectively guides the search for plausible connectivity patterns, and the approach proposed in this submission is particularly apt. The work presents a complete narrative, from biological mechanism inspiration to the implementation of Lp-convolution for neural activity prediction in V1 and representational similarity analysis.

* The paper's approach to addressing large kernel network training challenges could potentially bridge the performance gap between transformers and CNNs in image classification tasks.

* The mathematical formulation is sound and accessible, with figures (e.g., Figure 1 and 2) that effectively illustrate concepts and build intuition about parameter effects.

* The choice of the Sudoku challenge adds significant value, serving as an excellent demonstration of the model's capabilities in an easily understandable context (especially for $L_p$ mask shapes)

* The Appendix comprehensively addresses potential questions, demonstrating thorough consideration of the work's implications and limitations.

### Weaknesses
I don't think significant weaknesses are present in this work. The paper should be accepted as it is.

### Questions
**(More of a curiosity)** For future developments of this work, it would be interesting to explore connections with anisotropic diffusion (Perona & Malik, *Scale-space and edge detection using anisotropic diffusion*, 1990). In standard convolution, there exists a well-established mapping between convolution operators and isotropic diffusion processes (as explored in Scale-Space theory, particularly in Koenderink, *The structure of images*, 1987; and Lindeberg, *Scale Space theory in computer vision*, 1994). How might Lp-convolution relate to or extend these theoretical frameworks?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces $L_p$-Convolution by integrating the multivariate p-generalized normal distribution into a $L_p$ masks for convolution filters. It allows the network to adapt to different receptive field shape and train efficiently with large kernel. The paper show $L_p$-Convolution has a advantage in tasks such as sudoku challenge.

### Strengths
The idea of a $L_p$ convolution is original and could have wide application in visual tasks that require flexible receptive field size or in general tasks that require both local and global information from visual input. Showing its ability to transfer to any pretrained network greatly lowers the threshold to apply this for a wide range of tasks. The choices of Sudoku task and the follow up ablation analysis is solid and demonstrates well the strength of this method. Out of all papers that take inspirations from neuroscience and try to utilize it to improve neural nets, this paper stands out in actually providing a fundamentally different implementation of CNNs.

Other that than the points mentioned above, the benchmark testing was thorough and presented clearly.  The visualization of both the $L_p$ mask and convolution is very helpful for understanding the concepts. The writing is very clear.

### Weaknesses
The paper does show through table 1-4 that $L_p$-CNNs can train with large kernels and have some advantage in robustness as well as accuracy in benchmark test. However the improvements over baseline models are small. I don't think these numbers convinces me how useful the $L_p$-CNNs could be. Aside from the sudoku task, the paper didn't really show the advantage of efficiently trained large kernal $L_p$-CNNs through a task that actually could really benefit from large kernels. I would suggest including some more tasks that requires processing of context, or even tasks ViT excels at for comparison. 

For the robustness benchmark as well as the Sudoku task, it could be informative to include performance of ViTs as well. 

Lastly, it is pretty well established throughout the paper that $p_{init}=2$ is the most useful for most task, and resembles the most to the biological system. I am not sure it is worth having another section (sec. 6) dedicating to comparing similarity of RSM across different $p$s. If the author were to demonstrate it is also a better model for brain representational alignment that I would recommend doing a more thorough study including more datasets, brain regions and variation of CNNs.

### Questions
See weakness for suggestions.

Potential typo:
line 159: a solution to the of large kernel problem in CNN -> a solution of the large ...

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposes a brain-inspired approach of constraining the weights of convnets with a p-generalized Gaussian envelope. The authors demonstrate some minor improvements in performance on relatively small image datasets such as CIFAR-100 and TinyImageNet. They further claim that the learned representations of more "brain-like" convnets have higher representational similarity to the mouse visual system than their more classical counterparts.

### Strengths
+ Novel inductive bias for convnets motivated by biology
 + Overall fairly well-written paper
 + Well-motivated and well-executed experiments

### Weaknesses
- The effect sizes are really small, calling into question the practical impact
- Only "toy" datasets are explored
- Experiment on representational similarity not convincing


### Detailed explanation

While I find the paper well motivated and the idea original, I see the paper mostly as a negative result given the small effect sizes observed across most of the tables and the "toy" nature of datasets such as CIFAR-100 and TinyImageNet. The paper now has undergone several revisions that only reinforce this conclusion.

There is a statistically significant improvement due to Lp-Conv for some classical architectures, but not all of them (e.g. ResNet). Generally, the improvements are small (a few percent). Given that architectural modifications alone can now push accuracy on CIFAR-100 >90% (https://arxiv.org/abs/2304.05350v2), the 1–2% improvements in the 60–70% range feel insignificant. Experiments on more modern architectures such as RepLKNet (Table 3) show the same pattern, if anything with decreasing effect size (<1% improvement). Similarly, the transfer learning experiment using ConvNeXt-V2 (Table 4) shows close to no effect. There are no experiments on closer-to-real-world datasets like ImageNet (although that's by now a fairly standard problem that can be done on a consumer GPU), although I should say that I do not expect major effects in that experiment, either. The data simply show that the inductive bias doesn't do much.

The experiment on representational similarity yields equally small effect sizes, again insignificant on many architectures. In addition, the comparison is done to several mouse visual areas, some of which aren't even part of the "ventral" stream for which a convnet trained on image classification would be a reasonable model.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
1
