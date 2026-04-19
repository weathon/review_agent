# Variational Rectified Flow Matching

- Decision: Reject
- Scores: 6, 3, 6, 5

## Abstract
We study Variational Rectified Flow Matching, a framework that enhances classic rectified flow matching by modeling multi-modal velocity vector-fields. At inference time, classic rectified flow matching 'moves' samples from a source distribution to the target distribution by solving an ordinary differential equation via integration along a velocity vector-field. At training time, the velocity vector-field is learnt by linearly interpolating between coupled samples one drawn from the source and one drawn from the target distribution randomly. This leads to ''ground-truth'' velocity vector-fields that point in different directions at the same location, i.e., the velocity vector-fields are multi-modal/ambiguous. However, since training uses a standard mean-squared-error loss, the learnt velocity vector-field averages ''ground-truth'' directions and isn't multi-modal. Further, averaging leads to integration paths that are more curved while making it harder to fit the target distribution. In contrast, the studied variational rectified flow matching is able to capture the ambiguity in flow directions. We show on synthetic data, MNIST, and CIFAR-10 that the proposed variational rectified flow matching leads to compelling results with fewer integration steps.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Variational Rectified Flow Matching (VRFM), a framework that improves classic rectified flow matching by incorporating multi-modal velocity vector fields based on a variational perspective. Previous flow matching approaches average out directions, leading to curved integration paths and hindering accurately fitting the target distribution. VRFM, by contrast, captures the multi-modality of flow directions, thus preserving directional diversity. Experimental results on synthetic data, MNIST, and CIFAR-10 demonstrate that VRFM achieves promising results with fewer integration steps.

### Strengths
- Significance: This paper addresses a notable limitation in flow matching models, specifically the ambiguity at path intersections that results in curved sampling trajectories. By tackling this ambiguity, the proposed approach demonstrates clear improvements over existing rectified flow models, particularly in low NFE settings.

- Originality and clarity: The paper is well-written and easy to follow, clearly presenting concepts. Interpreting the flow-matching objective through variational inference to reduce directional ambiguity is conceptually sound, adding a meaningful perspective to the flow-matching framework.

### Weaknesses
- In line 53, the authors claim, “This results in trajectories that are more straight”.  It is unclear why reducing ambiguity would inherently lead to straighter flows. Including a theoretical proof or a detailed explanation to clarify this result would strengthen the argument.

- For completeness, the paper should include proofs demonstrating that the learned distribution from VRFM preserves the marginal data distribution, as established in Theorem 3.3 in [1].

- The most concerning part of this paper is limited evaluation and performance compared to the recent papers. The empirical evaluation is restricted to MNIST and CIFAR-10, which limits the generalizability of the findings. Extending the evaluation to additional datasets, such as ImageNet 64x64, would improve the generalizability of the findings. Furthermore, the reported results of VRFM in low NFE regimes (e.g., 104 FID for 2 NFE on CIFAR-10) are less compelling, given the recent advances [1,2,3] in reducing sampling costs in diffusion (or rectified flow) models. For instance, reflow on rectified flow (e.g., 2-rectified flow) achieves a 4.85 FID with a single step [1]. Results showing VRFM’s performance with the reflow technique would provide a more competitive comparison. 

- It would be valuable if the authors could provide results on conditional generation setting.

### Questions
- Is there any reason for designing the input of the posterior encoder as [x0, x1, xt, t]?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper introduces Variational Rectified Flow Matching as a method to model multi-modal velocity and ambiguity in the data-space-time-space domain. The properties of Variational Rectified Flow Matching are studied and validated through experiments with visualizations on low-dimensional synthetic data. Compelling results are demonstrated on the synthetic data, MNIST, and CIFAR-10 datasets.

### Strengths
1. The paper presents a valuable observation: because the vector field is parameterized via a Gaussian at each data-domain-time-domain location, ambiguity cannot be captured.

2. The analytical experiments with visualizations are well-executed and contribute significantly to validating the theoretical analysis.

### Weaknesses
1. Current evaluations are too weak and this paper lacks enough experiments on more real-world and complex datasets, such as the AFHQ, CelebA and ImageNet datasets. The evaluations on these benchmarks are necessary for demonstrating the effectiveness of the proposed method.

2. The authors should sufficiently discuss these missing related works [1][2][3][4][5], and compare with them in the experiments.

[1] Nguyen B, Nguyen B, Nguyen V A. Bellman Optimal Stepsize Straightening of Flow-Matching Models[C]//The Twelfth International Conference on Learning Representations. 2024.

[2] Song, Yang, et al. Consistency models. arXiv preprint arXiv:2303.01469 (2023).

[3] Yang, Ling, et al. Consistency flow matching: Defining straight flows with velocity consistency. arXiv preprint arXiv:2407.02398 (2024).

[4] Yan, Hanshu, et al. Perflow: Piecewise rectified flow as universal plug-and-play accelerator. arXiv preprint arXiv:2405.07510 (2024).

[5] Kim, Dongjun, et al. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. arXiv preprint arXiv:2310.02279 (2023).

### Questions
I am curious whether Variational Rectified Flow Matching can enhance performance through "reflow", similar to classic rectified flow matching as discussed in [1].


[1] Liu, Xingchao, and Chengyue Gong. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." The Eleventh International Conference on Learning Representations.

### Soundness
2

### Presentation
2

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
The paper introduces a novel framework, Variational Rectified Flow Matching, which addresses the limitations of conventional Rectified Flow (RF) methods in capturing ambiguity in velocity distributions. By incorporating an additional latent variable z drawn from a Gaussian prior, the framework models multiple modes of ambiguity. An encoder is used to derive the posterior distribution p(v∣xt,t,z) at a specific sample xt and time t. This approach is claimed to better capture ambiguity and improve the empirical performance of diffusion model generation.

### Strengths
The strengths of the paper are as follows:

- Clear, easy-to-follow presentation with strong empirical performance compared to baseline methods.

### Weaknesses
The weaknesses of the paper are listed below:
- The motivation of the paper is unclear, particularly in the Introduction. The statement at line 73, “Importantly, variational rectified flow matching differs in that it enables modeling ambiguity in the data-space-time-space domain, i.e., the goal is a model where flow trajectories can intersect,” does not clarify how allowing flow trajectories to intersect resolves the ambiguity problem.
- As I understand, using an additional latent variable z when modeling the velocity at (xt,t) can partially address the ambiguity problem, as different values of z capture different modes or sources of variation in the velocity distribution. However, VAEs are known to sometimes experience mode collapse, where the same high-density velocities may be generated from multiple modes of z. How does the proposed method handle this issue? To further address this concern, I suggest including a “diversity metric” in the experimental protocol to measure the variety of generated samples. 
- Furthermore, I believe RF can address the ambiguity problem by performing multiple rectifications. When training stabilizes, the optimal velocity at each sample xt at time t becomes unique, eliminating ambiguity. Even without rectification, existing methods such as OT-FM can mitigate ambiguity by improving the coupling between x0 and x1, resulting in less ambiguous directions at (xt, t). Are there any theoretical or methodological benefits of the proposed approach compared to these methods? Without such justification, it’s difficult to attribute the improved performance to the additional variable z.
- Can different values of z affect the visual quality of the generated samples? If x0 is kept constant, does varying z introduce significant variance in the generated samples? Or is there a specific value of z that results in low-quality samples? 
- There is no theoretical guarantee that the proposed approach will achieve a better straight flow when addressing the ambiguity problem.

### Questions
Refer to the Weaknesses section for my concerns and questions.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper proposes a novel method using variational inference to address the ambiguity of velocity vector fields that classic rectified flow matching fails to capture. By introducing latent variables and modeling ambiguity through a mixture model of velocity vector fields, the method enables more accurate data distribution capture and efficient integration. Experimental results demonstrate that the proposed approach achieves comparable performance to existing methods with fewer steps on synthetic data and image datasets.

### Strengths
- The paper is well-structured and easy to read.
- The proposed method integrates VAE and flow matching in a straightforward manner, offering novelty in its ability to learn vector fields with uncertainty. Furthermore, the high performance on MNIST and CIFAR-10 datasets suggests that the hypotheses and approaches of this study are reasonably valid.

### Weaknesses
- The proposed method in this paper introduces latent variables and their inference model, enabling the capture of overlapping vector fields when they occur. However, it is necessary to clarify the setting that assumes such overlapping vector fields. The overlap and ambiguity in question here initially seemed to imply that while the flow from a specific $x_0$​ to $x_1$​ is uniquely determined, there exist different vector fields at a particular time and spatial location that may intersect. However, during inference with the proposed method, a data point $x_0$​ is sampled from the source distribution, followed by the sampling of a latent variable, which is then used as the initial value to integrate an ODE based on a vector field determined by it. This implies that there is uncertainty in the direction from the initial value $x_0$​, meaning there is an assumption that it could lead to a different $x_1$​. Is this setting reasonable? The uncertainty involving deterministic flows crossing and the possibility of tracing different flows from the initial value (i.e., the $x_0$ and $x_1$​ pairings are not unique) appear to be mixed. The authors should clearly distinguish between these and clarify which aspect they are aiming to address.
- Considering the above points, while the proposed method may indeed enable faster transitions to the target due to the learned flow being linear even when vector fields overlap, during inference, $z$ is sampled from the prior and thus is not determined solely by $x_0$. As a result, the model could reach a different $x_1$​, which may not be desirable from the perspective of two-sided conditioning flow matching. 
- The proposed method requires an inference model and employs separate encoders for each of $x_0​,x_1$​, and $x_t$​ with the same structure as the encoder in $v_\theta​$. This implies a significantly larger number of learnable parameters compared to existing models, and although the encoders are not used during inference, it is not entirely fair in terms of parameter count relative to previous research (even if the speed remains similar). Therefore, it would be necessary to evaluate the impact of the size of the inference model’s encoders by modifying their size and investigating how it affects performance.

### Questions
- I would like the authors to respond to the points I raised as concerns regarding the above weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
