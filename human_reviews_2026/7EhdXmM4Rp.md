# Weighted Conditional Flow Matching

- Decision: Reject
- Scores: 6, 2, 2, 2

## Abstract
Conditional flow matching (CFM) has emerged as a powerful framework for training continuous normalizing flows due to its computational efficiency and effectiveness. However, standard CFM often produces paths that deviate significantly from straight-line interpolations between prior and target distributions, making generation slower and less accurate due to the need for fine discretization at inference. Recent methods enhance CFM performance by inducing shorter and straighter trajectories but typically rely on computationally expensive mini-batch optimal transport (OT). Drawing insights from entropic optimal transport (EOT), we propose Weighted Conditional Flow Matching (W-CFM), a novel approach that modifies the classical CFM loss by weighting each training pair $(x, y)$ with a Gibbs kernel. We show that this weighting recovers the entropic OT coupling up to some bias in the marginals, and we provide conditions under which the marginals remain nearly unchanged. Moreover, we establish an equivalence between W-CFM and the minibatch OT method in the large-batch limit, showing how our method overcomes computational and performance bottlenecks linked to batch size. Empirically, we test our method on unconditional generation on various synthetic and real datasets, confirming that W-CFM achieves comparable or superior sample quality, fidelity, and diversity to other alternative baselines while maintaining the computational efficiency of vanilla CFM.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
To overcome the computational burden of OT-CFM, this paper proposes to weight the FM loss by per-pair Gibbs weight $w_{\epsilon}(x,y)$ and it leads to the W-CFM. The W-CFM loss approximates the entropic OT up to marginal tilting. They also show large-batch equivalence to OT-CFM. On the choice of tuning parameter $\epsilon$, the paper discuss a huristic way. By simulation and experiements on image datasets, they show the W-CFM has competitive or better FID and path straightness.

### Strengths
1. The paper is well written and easy to follow.
2. The training of W-CFM is as cheap as I-CFM, while recovering OT-like straightness.
3. The formal link to EOT and marginal tilt validate the weighting on loss.

### Weaknesses
The choice of $\epsilon$ can still be relatively cumbersome? In the paper, they set $\epsilon = \kappa \sqrt{d}$, while $\kappa$ is chosen by grid-search.

### Questions
Some minor questions:
1. How senstive of the W-CFM to the choice of cost function?
2. How senstive to $\epsilon$? According to simulation, looks quite sensitive?

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
4

### Summary
This paper proposes W-CFM, which leverages the Gibbs kernel as a weight function to reduce FM loss. The primary motivation for W-CFM is to straighten FM's path and thus improve FM's performance. The experimental results show that W-CFM performs well on unconditional generation in simulation and toy datasets such as CIFAR-10.

### Strengths
1. The topic of this paper is essential. Making the FM path straighter is key to reducing NFE and accelerating the sampling process while maintaining high generation quality. Reflow is one way, but it suffers from two-stage training and requires traversing the entire reverse process. W-CFM is one-stage and does not necessarily traverse the entire reverse process, which is interesting.

### Weaknesses
However, this paper contains some major concerns:

1. The motivation for introducing the Gibbs kernel to achieve a straighter path is unconvincing. Firstly, this paper does not offer any theoretical justification for this motivation. Meanwhile, the experimental results cannot support the motivation either. For example, Fig.2 shows that the path of OT-CFM is straighter than that of W-CFM. Meanwhile, image generation tasks also show that W-CFM cannot achieve this motivation. Under the same NFE, if the path is straight enough, the FID should show a "lerp". However, Fig. 3 indicates that the FID of W-CFM is similar to that of OT-CFM. 

2. The motivation for using the Gibbs kernel itself is unclear. The related content spans 155-161 lines. But the author just claims that I-CFM can be written as a weight-function formulation. Then, they use the Gibbs kernel as another invariant of the weight function. Is there any specific reason here? In my view, directly leveraging a $l_{2}$ norm between GT and $x$ as the weight function can still work.

3. W-CFM is sensitive to the hyperparameter. As shown in Fig. 2, the hyperparameter directly determines W-CFM's performance, hindering its implementation in i2v and t2v tasks.

### Questions
1. Why cannot such a straight line accelerate the sampling process? The motivation for this question is that Rectified flow [1] mentions that a straighter path can reduce the NFE, which is the essence of Reflow. In this paper, a straighter path performs similarly to FM under the same NFE, according to Table 3, which is confusing.

2. Can the author report the performance of W-CFM in the hyperparameter=1 situation? The motivation for this question is that Fig. 2 shows that the smaller the hyperparameter, the straighter the path. A natural question is: what about using 1 directly? 

3. Can you offer the experiments on different solvers, such as the Euler solver? The motivation for this question is that the Euler solver is the mainstream for FM, so we need to ensure W-CFG works well with it.

[1] Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. Liu et al. ICLR 2022.

To sum up, the topic of this paper deserves more attention. But because of the weaknesses, it is difficult for me to be convinced that W-CFM can work. Therefore, I rate it as reject temporarily. If the author can clarify these concerns, I am willing to increase my score.

### Soundness
1

### Presentation
2

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
The paper introduces Weighted Conditional Flow Matching (W-CFM). The key idea is to weight each training pair using a Gibbs kernel, allowing the method to approximate the Entropic Optimal Transport (EOT) plan without the high computational cost of explicitly solving the OT problem and without being constrained by batch size. The authors derive theoretical conditions under which this approximation preserves the original marginals and prove that W-CFM becomes equivalent to OT-CFM in the large-batch limit, provided the marginals remain unchanged. The method is evaluated on both toy and image datasets.

### Strengths
The authors propose a method that approximates EOT-CFM by simply reweighting pairs sampled from an independent plan, thus avoiding the need for mini-batch OT computations. They provide a theoretical justification for its equivalence to OT-CFM in the large-batch limit. However, this reweighting results in tilted marginals, i.e., $\tilde{\mu}_\epsilon(dx)=\frac{\exp{(-\phi\_\epsilon(x))}}{Z^1\_\varepsilon}\mu(dx)$ (eq. 10), where $\mu(dx)$ is the original marginal, so the authors analyze the conditions under which this tilting can be neglected. They also introduce a procedure for selecting the regularization coefficient $\varepsilon$, which controls the trade-off between the straightness of trajectories and the degree of marginal tilting in the proposed framework. Extensive experiments show that the proposed method achieves performance comparable to previous approaches.

### Weaknesses
### **Terminology**

- In the statement of Theorem 1, there are $\phi, \psi$, whereas in Equation (2) they appear as $\phi_\varepsilon, \psi_\varepsilon$. Moreover, Theorem 4.2 from [1] states that $\phi, \psi$ are called _EOT potentials_, not _Schrödinger potentials_ and Remark 4.3 therein notes that there is already some inconsistency in the naming.  

### **Theory**
- As I understand it, $\mathcal{L}\_{\text{W-CFM}}$ can be considered an accurate approximation of $\mathcal{L}\_{\text{EOT-CFM}}$ only if the EOT potentials corresponding to the entropic plan satisfy

$$-\phi\_\varepsilon(x)\psi_\varepsilon(y) \approx C$$

which seems a very restrictive assumption meaning that the potentials are constants.
  
- Furthermore, there is a notation overload in Equation (12), since $f_\varepsilon(x)$ was already defined as $f_\varepsilon = \exp(\phi_\varepsilon(x))$. The conditions below (12) for marginal preservation are rather broad and obvious. It is not clear how the constant between the two measures affects the tilting – it seems that this constant should be close to 1, but the sensitivity of this constant is not discussed.

- Proposition 3 appears rather trivial, as the assumption that the tilted marginals are not tilted is too strong. In this case, $q_\varepsilon = \pi_\varepsilon$, since $\exp(-\phi_\varepsilon(x)) = Z^1_\varepsilon$ and $\exp(-\psi_\varepsilon(y)) = Z^2_\varepsilon$, and therefore, from Equation (8), it follows that $Z_\varepsilon = Z^1_\varepsilon Z^2_\varepsilon$. For the case $q_\varepsilon = \pi_\varepsilon$, the mini-batch approximation of the true entropic plan is a standard result [2]. Moreover, the proof relies mainly on intuitive reasoning and does not provide references for rigorous derivations.

---

### **Practice**

- W-CFM has a notable limitation – a trade-off controlled by $\varepsilon$ between preserving the marginals and maintaining straight trajectories. In Table 5, larger values of $\varepsilon$ prevent marginal distortion but result in less straight trajectories. Since $\varepsilon$ appears under the exponent, the tilting of marginals seems to be very sensitive to its value. Moreover, I did not find a sensitivity analysis for $\varepsilon$ on the image domain.

- Finally, the performance improvement is modest, as shown in Table 2, where even I-CFM achieves comparable results.

[1] Nutz, Marcel. "Introduction to entropic optimal transport." Lecture notes, Columbia University (2021).

[2] Hundrieser, Shayan, Marcel Klatt, and Axel Munk. "Limit distributions and sensitivity analysis for empirical entropic optimal transport on countable spaces." The Annals of Applied Probability 34.1B (2024): 1403-1468.

### Questions
- In Table 2, the performance of the proposed method is compared with I-CFM and OT-CFM. Could the authors also report the training time to clarify how large is the computational advantage of the proposed method?
- It would also be helpful if the authors included results with different cost functions, such as the $\ell_1$ distance, though this is minor.
- In addition, could the authors provide an analysis of the trade-off with respect to $\varepsilon$ on image data?
- Finally, how is this work related to [1] (Theorem 4.3) and [2] (Eq. 10)?

[1] Zhang, Shiyuan, Weitong Zhang, and Quanquan Gu. "Energy-Weighted Flow Matching for Offline Reinforcement Learning." ICLR 2025.
[2] Park, Seohong, Qiyang Li, and Sergey Levine. "Flow q-learning."  ICML 2025.

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
4

### Summary
The authors propose the method for learning the Conditional Flow Matching models with straighter and shorter trajectories. The authors propose to modify the optimization objective by incorporating the Gibbs kernel, i.e., weightening of training pairs (x, y), and call their method Weighted Conditional Flow Matching (W-CFM). The modified objective coincides with the objective of Conditional Flow Matching with Entropic Optimal Transport plan for training pairs (x, y), but with different “tilted” marginals. Authors address the issue with “tilted” marginals in practice by optimizing over the stochasticity of EOT parameter $\epsilon$ and show examples where marginals are not “tilted”. In addition, the authors show that under some assumptions their method W-CFM coincides with the EOT-CFM method in the limiting batch size case.

On practice authors, compare their method with other Conditional Flow Matching methods with different data couplings. The methods are compared on toy 2D datasets as well as on high dimensional image datasets.

### Strengths
- The authors propose a method for training a CFM model that lets one to approximate an entropic optimal transport plan for training data pairs, without extra compute as in OT-CFM/EOT-CFM.

- The authors test their approach on several data types and analyze different aspects of CFM models.

- The method delivers decent results on image data, outperforming other CFM approaches by both fidelity and diversity.

### Weaknesses
The main weakness of the method is **“tilted” marginals**:

- There are no clear types of cases where marginals are not tilted (except for very limited case described in Proposition 2). In contrast the OT-CFM/EOT-CFM methods are asymptotically unbiased and when batch size grows to infinity they recover the ground truth OT or EOT plan. While in W-CFM there are no such parameters that guarantee the lack of tilting.

- In that light the claims of authors that distributions are not “tilted” significantly remain unsubstantiated.

- Proposition 3 is built on a “mild regularity” assumption (no marginal tilting and bounded support) and in my opinion these assumptions are not “mild” at all, since authors show very few cases where marginals are not “tilted” and bounded support is also a strong assumption.

The practical methodology for choice of $\epsilon$ **remains unclear.** Authors say “we search over a grid of k values spaced uniformly in log scale and select the smallest value for which the relative variance starts flattening, following an ”elbow rule” heuristic akin to the selection of the number of principal components in PCA”. But still I do not catch the overall algorithm. Can authors explain in more details?:
- As far as I understand the bigger the $\epsilon$ the closer ratios between “tilted” and original marginals, i.e, $f_\epsilon$ and $g_\epsilon$ in Eq 12, to constant. In that sense the bigger $\epsilon$ one takes the less “tilted” the marginals, but this undermines W-CFM purpose, because then the EOT plan is closer to an independent plan. Then the choice of $\epsilon$ remains even more controversial and unclear.

The experiment results are **ambiguous.**
- On the MoG -> 5 Gaussians problem W-CFM delivers not the straightest trajectories (Table 1, NPE), but shows the best distributions fit (Table 1, W2)
- While on 8 gaussians -> moons the W-CFM is worse in both fitting the target distribution and trajectories straightness.
- While on image dataset the W-CFM in majority cases indeed deliver better fidelity and diversity then I-CFM and OT-CFM, it holds not in all the cases and the gains w.r.t. other methods are rather marginal.
- The authors compare their method with OT-CFM, but I couldn't find information wherever they use mini batch Optimal Transport or mini batch Entropic Optimal Transport (Sinkhorn) to generate training data pairs. While the comparison with mini batch EOT with the same $\epsilon$  would be desirable for clean experimental comparison.

### Questions
- Can authors explain the methodology for choosing $\epsilon$ more thoroughly and extrapolate on lines 253-255?

- Which mini batch pairing algorithm, i.e., mini batch Optimal Transport or mini batch Entropic Optimal Transport with some $\epsilon$, was used as a part of OT-CFM in the experimental part of the paper?

### Soundness
2

### Presentation
3

### Contribution
1
