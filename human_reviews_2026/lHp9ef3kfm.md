# Connecting Independently Trained Modes via Layer-Wise Connectivity

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Empirical and theoretical studies have shown that continuous low-loss paths can be constructed between independently trained neural network models. This phenomenon, known as mode connectivity, refers to the existence of such paths between distinct modes—i.e., well-trained solutions in parameter space. However, existing empirical methods are primarily effective for older and relatively simple architectures such as basic CNNs, VGG, and ResNet, raising concerns about their applicability to modern and structurally diverse models. In this work, we propose a new empirical algorithm for connecting independently trained modes that generalizes beyond traditional architectures and supports a broader range of networks, including MobileNet, ShuffleNet, EfficientNet, RegNet, Deep Layer Aggregation (DLA), and Compact Convolutional Transformers (CCT). In addition to broader applicability, the proposed method yields more consistent connectivity paths across independently trained mode pairs and supports connecting modes obtained with different training hyperparameters.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes an algorithm for finding low-loss paths between independently trained models across a wide variety of architectures. The proposed method is based on the empirical observation that the variance of trained model parameters (i.e., the sum of squared parameter values) tends to remain approximately constant. Using this approach, the authors demonstrate that continuous paths with small training losses can be found in various architectures trained on CIFAR-10. They also show that low-loss paths can be identified even between models trained with different hyperparameters, which result in different parameter variances.

### Strengths
* The paper proposes an algorithm that enables verification of mode connectivity across a variety of architectures.  
* It demonstrates that low-loss paths can be discovered even between models trained with different hyperparameters.

### Weaknesses
* It is unclear whether Algorithm 2 can always successfully discover a path between models with different parameter variances.  
* The experiments rely almost entirely on CIFAR-10.  
* The method does not necessarily yield paths with low test loss.  
* Although the paper claims that paths can be found between models trained with different hyperparameters, the actual experiments are restricted to a very limited setting.

### Questions
## The mathematical notation could be clarified to avoid confusion.

For example, $l_x$ is used to denote the $x$-th layer. However, $x$ is also used to represent a hyperparameter, and this dual use may cause confusion. In addition, $x$ commonly represents the input, so using it as a hyperparameter symbol is unconventional.  

Furthermore, the variance of the $l_x$-th layer is written as $\mathrm{Var}(\theta_{l_x})$. However, this expression does not refer to the variance of a distribution in the usual statistical sense but rather to the squared L2-norm of the layer parameters. Using the term “variance” might imply that $\theta_{l_x}$ is a random variable drawn from a distribution determined by the training algorithm. Since the intent seems to be to compute the variance across the dimensions of $\theta_{l_x}$ itself, it would be clearer to simply write $\Vert \theta_{l_x} \Vert^2$.

## It is unclear whether Algorithm 2 consistently works as intended.

As I understand it, Algorithm 2 searches for a path from a model with larger parameter variance to another with smaller variance. It is used to align the variances of two pretrained models so that Algorithm 1 can then find a low-loss path between them. In other words, if Algorithm 2 fails to produce an intermediate model whose variance matches that of the destination model, Algorithm 1 may not work properly.  

At line 402, the authors write:  
> Unlike the results in Figure 2, the L_2 distance here is not expected to converge to very small values, because the destination point in Algorithm 2 is the origin, which is not itself a low-loss mode.

This suggests that Algorithm 2 does not always produce models with small L2 norms. This might be acceptable if we could guarantee that models with small parameter norms necessarily yield large losses. However, that is not always the case. For example, when a convolutional layer is followed by layer normalization, scaling the convolution weights does not change the layer output. Consequently, the variance of such weights could be arbitrarily small without degrading performance. Therefore, if one of the pretrained models happens to have a much smaller parameter norm, the proposed method might fail. Unless Algorithm 2 can **always** find the lowest-variance model with performance comparable to the original, this remains a potential limitation.

## The experiments are biased toward CIFAR-10.

Although various architectures are tested, the dataset choice is limited to CIFAR-10. It would be more convincing to include experiments on larger-scale datasets such as CIFAR-100 or ImageNet.

## There is no guarantee that models with low test loss will be found.

As acknowledged by the authors, the proposed algorithm ensures low training loss but not low test loss. If the purpose of this study is to provide new insight into the generalization properties of SGD, this limitation should be considered a notable weakness.

## It would be helpful to include experiments where hyperparameters other than weight decay are changed.

The paper claims that low-loss paths can be found between models trained with different hyperparameters. However, in practice, the only hyperparameter varied is the weight decay, which mainly affects the scale of the parameters. It is well known that the sharpness of the loss landscape is also influenced by other hyperparameters such as the learning rate. At line 314, the authors mention:  
> Empirically, we find that starting from sharp minima can cause our algorithm to fail to find feasible paths.

However, the paper does not provide experiments that explicitly examine the effect of sharpness. It would strengthen the work to include experiments where other hyperparameters, such as the learning rate, are varied to demonstrate the broader applicability of the proposed method.

### Soundness
3

### Presentation
2

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
This paper proposes LLPF, a new algorithm that implements mode connectivity, i.e. finding a low-loss path between two modes obtained by independent training.

### Strengths
- The presentation is clear and the paper is easy to follow.
- The experiment results verify the effectiveness of the proposed algorithm.

### Weaknesses
- I don't see why this problem is important. In my understanding, mode connectivity is more like a phenomenon that helps us better understand the loss landspace, instead of a challenge that needs to be solved by an algortihm. Perhpas the authors can describe more of the practical usefulness of their algorithm?
- The algorithm looks trivial, and is not explained at all. For example, why do you need to project back to the variance sphere at each step? How do you ensure there is no loss barrier between the point and the projected point (e.g. M1 and M2, in the context of Figure 1)? how do you guarantee M3 is not too far away from M2?

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a novel method to find connecting low-loss paths between models by a layer-wise search. For each layer, the method explores the sphere of layer-parameters with similar variance. The method is sensitive to the order in which layers are traversed, arguing that the order should follow the data-flow. The method is applicable to modern architectures where standard linear or Bezier interpolation approaches fail.

### Strengths
- The proposed method to find low-loss paths is novel and interesting.
- The proposed method is tested for recent model architectures.
- The notion of a _variance sphere_ and the layer-wise variance correction step are intuitive and sound.
- The empirical evaluation shows the method works for a wide variety of model architectures and results are consistent across seeds.
- Algorithms are clearly presented and linked to geometric reasoning.
- Consistent paths across seeds and architectures suggest a shared geometric structure in modern networks, potentially revealing deeper properties of the loss landscape.

### Weaknesses
- The experiments are _somewhat_ limited to visualizations of loss/accuracy trajectories; no quantitative comparison of path quality (e.g., path length, interpolation efficiency, energy landscape visualization).
- The effect of layer order, variance correction, and training steps is not systematically analyzed in an ablation study.
- The approach is iterative and layer-wise. The paper acknowledges high compute requirements but gives no runtime analysis.

### Questions
- You state that “minima with near-zero training loss are generally flat.” Could you quantify this or connect it to measurable flatness metrics (e.g., relative flatness [3], Fisher-Rao Norm [2])?
- The observation that low training loss implies flatness can be explained for the CE loss: Walter et al. [4] have shown that flatness relates to model confidence and model confidence increase wit ha decreasing CE loss.
- Have you checked whether sharp minima with near-zero loss actually fail your algorithm? That would support your empirical claim.
- Since the paper’s method depends on starting from flat minima, and relative flatness [3] formalizes layer-wise curvature-weight norms, there seems to be a conceptual overlap. Could the variance-sphere assumption be seen as implicitly fixing relative flatness per layer?
- Adilova et al. [1] found that certain layers, or layer clusters, have a stronger impact on the loss barrier. How would ordering the layers according to their _importance_, i.e., influence on the loss barrier, impact your method? 
- Adilova et al. [1] also found that different directions in parameter space have different impacts on the loss value. It would be interesting to see whether the layer-wise updates of the proposed methods correspond to the space perpendicular to the training space and averaging direction.

References: 

[1] Adilova, Linara, et al. "FAM: Relative Flatness Aware Minimization." Topological, Algebraic and Geometric Learning Workshops 2023. PMLR, 2023.

[2] Liang, Tengyuan, et al. "Fisher-rao metric, geometry, and complexity of neural networks." The 22nd international conference on artificial intelligence and statistics. PMLR, 2019.

[3] Petzka, Henning, et al. "Relative flatness and generalization." Advances in neural information processing systems 34 (2021): 18420-18432.

[4] Walter, Nils Philipp, et al. "When Flatness Does (Not) Guarantee Adversarial Robustness." arXiv preprint arXiv:2510.14231 (2025).

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a new algorithm (Low Loss Path Finding) to mode connect two independently (from different initializations) trained (with different data ordering and augmentations) networks. The paper tests the algorithm on architectures that are not common in this line of work (MobileNet, ShuffleNet, etc), claiming that they are more recent. The authors claim that their method is more reliable—producing consistent paths for networks trained with different hyper-parameters---though they don’t actually cite or confirm if the prior work fail on these settings.

### Strengths
- The authors revisit non-linear mode connectivity to a broader set of architectures beyond the commonly studied ResNet/VGG models.
- I believe the discussion of the models trained with different hyper-parameters and the focus on different variances is an important point that is not addressed in the prior literature. The algorithm and the discussion of data flow order is sound.

### Weaknesses
- The paper does not provide clear motivation for why demonstrating mode connectivity in MobileNet, ShuffleNet, or the other selected architectures is important or beneficial. What practical or theoretical insights would we gain from showing mode connectivity in these specific models? The choice of architectures appears arbitrary and is not justified. Moreover, the proposed architectures (MobileNet, ShuffleNet, EfficientNet, RegNet) are roughly contemporary with the mode connectivity literature itself, contradicting the claim of addressing "modern" architectures. Addressing different model families like transformers, diffusion models could have been a fresher take on this topic.
- The architectures in the (linear) mode connectivity literature are indeed outdated. The paper should cite Juneja 2022, and Altıntaş 2025 for discussion of transformer-based models and NLP domain.
- Even though they claim to be more effective, the paper doesn’t compare the proposed method against the baseline methods (NEB, FGE, and SRPO) on the architectures that are evaluated in the prior work and the architectures that the paper expands in.
- The experimental setup appears to be inherited from prior work without significant extension or deepening. Even all three baselines (Garipov, Draxler, Benton) conducted experiments at the CIFAR-100 scale. The paper does not appear to push beyond the established experimental boundaries in terms of dataset scale, complexity, or diversity of evaluation metrics
- The paper claims the method "generalizes beyond traditional architectures" and supports a "broader range of networks," but the selected architectures are not particularly modern and the experimental validation does not convincingly demonstrate this generalization beyond what prior methods might achieve.
- The benefits of the algorithm is not validated. According to Table 6, algorithm 1 takes 21 hours on an H100 GPU for a ResNet18 on Cifar-10. If I am not misreading this table, the cost of the algorithm is quite high. Training a ResNet18 or 20 on Cifar-10 shouldn’t take more than 10 minutes.

### Questions
- Could the authors explain the different phases in L2 distance in Figure 2? Is there any pattern emerging? I am also curious if the authors have an insight about behavior in the top subfigure and its implications for the local geometry?
- Why were these specific architectures chosen? What are the practical implications of achieving mode connectivity for these specific architectures?
- Can the authors provide experimental results comparing their method to NEB, FGE, and SRPO on the same set of architectures? How does the proposed method's computational cost compare to these baselines? Table 6 could incorporate the runtime for these baseline methods. 
- The authors could draw connection between the muP regime and their algorithm to provide insights. The paper could also benefit from discussing scale invariances.

### Soundness
2

### Presentation
2

### Contribution
1
