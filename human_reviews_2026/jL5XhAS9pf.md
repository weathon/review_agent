# Flow Matching on Unordered Sets

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Flow matching has achieved promising performance across a broad spectrum of data modalities (e.g., image and text). However, there are few works exploring their extension to unordered point sets. Indeed, previous generative models are mostly designed for vector data, with a natural ordering along dimensions.  In this paper, we present unordered flow, a type of flow-based generative model for generating point sets. Specifically, we propose a lifting approach where we convert unordered data into an appropriate function representation, and learn the probability measure of such representations through function-valued flow matching. For the inverse map from a function representation to unordered data, we introduce a particle filtering method that first warms up the initial particles with Langevin dynamics and then updates them until convergence through gradient-based search. We have conducted extensive experiments on multiple real-world datasets, showing that our unordered flow model is highly effective in generating set-structured data and significantly outperforms previous baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this work, the authors present building flow matching on unordered point sets. In particular, a lifting method first converts unordered data into a function representation and a flow matching model is trained on function-valued mapping. The authors further introduce a particle filtering method to generate from the learned function representation. Experiments on synthetic and real-world benchmarks shows the proposed method achieves competitive performance in generating unordered set.

### Strengths
1. The paper is well-written and easy to follow. 
2. This work investigates flow matching models on unordered sets which is a valuable direction for building more general generative models. 
3. On benchmarks shown (e.g., Earthquakes and COVID-19), the proposed method shows competitive performance.

### Weaknesses
1. The empirical study is on small scaled dataset. It's unclear how the proposed method works on larger scale. 
2. Another line of works that builds generative models on function space [1,2] is also related to the proposed method, but they are not discussed in the paper. 

Reference:
[1] Dupont, Emilien, et al. "From data to functa: Your data point is a function and you can treat it like one." arXiv preprint arXiv:2201.12204 (2022).
[2] Du, Yilun, et al. "Learning signal-agnostic manifolds of neural fields." Advances in Neural Information Processing Systems 34 (2021): 8320-8331.

### Questions
1. What is the size of datasets in empirical study? It would help better understand the scale of the benchmark. 
2. What is the architecture of the flow matching model? And also what is the model size compared to baselines?
3. I think the proposed method can be applied to broader problems like 3D point cloud generation. Did the authors try 3D generative tasks and how the proposed method works?
4. In Eq.5, to adaptive rescale the Gaussian variance, one will need to compute pairwise distances over all points. I'm curious how much compute overhead this leads to, especially for data with large number of points.

### Soundness
2

### Presentation
3

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
This paper presents the unordered flow that converts unordered data into the function representation with adaptive variances and learn it through function-valued flow matching. To return back to the unordered set from the function representation, the paper uses a particle filtering method that randomly initializes the particles with Langevin dynamics and then performs gradient ascent on the particle updates. Moreover, the paper proposes de-duplication procedure as noise filtering to handle redundancy. Finally, the paper experiments on multiple real-world datasets to show the effectiveness of the proposed method on generating set-structured data.

### Strengths
•	The paper is well-written and clearly-organized.

•	The details of the method are considered in depth. The adaptive variance in function representation, the de-duplication as noise filtering are designations that improve the sample quality.

•	The experiments and the ablation studies show the effect of the adaptive variance, the Langevin warm up and the noisy peak filtering.

### Weaknesses
•	The sample efficiency of the proposed method seems to be low due to the redundancy during the conversion between the unordered set and the function representation.

•	It would be beneficial to compare the proposed method with some recent flow matching method on point cloud generation like [1], since the main design is to use function-valued flow matching.

•	[1] Lan, Y., Zhou, S., Lyu, Z., Hong, F., Yang, S., Dai, B., ... & Loy, C. C. (2024). GaussianAnything: Interactive Point Cloud Flow Matching For 3D Object Generation. arXiv preprint arXiv:2411.08033.

### Questions
•	Why exactly does the Langevin warm-up “effective to make gradient-based search robust to noisy peaks”? It seems that it is still possible that the sampling being trapped in the local maximums after the warm-up, since the target distribution still has spurious modes. Can this be explained in mathematical details?

•	In the second step of the de-duplication procedure, how to determine “small in size” quantitatively?

•	What is the relationship between the proposed method and other flow matching method on point cloud generation? Is the conversion between the unordered set and the function representation sample efficient?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a generative modelling framework for unordered sets (e.g., point‐sets) via, mixture function representation, flow matching on function space, and an inverse transform. The mixture representation ensures the permutation invariance. The generative pipeline include: a flow-based function generation, and an inverse transform to map the function to a discrete set via a two-stage process (warm‐up with Langevin dynamics on particles + gradient‐based refinement).

### Strengths
1. The paper has clear motivation. The unordered nature of sets is a real and non‐trivial challenge in generative modelling

2. The proposed approach is novel. By mapping the point sets into mixture function allows permutation invariance and naturally leads to a flow model in function space. 

3. They showed performance gains on multiple real-world datasets. .

### Weaknesses
1. It is not very clear that the flow model is guaranteed to generate the desired mixture function, with $N$ mixtures. For example, it may introduce spurious modes.   

2. I think the description on the inverse mapping to recover the points from the generated mixture representation using the trained model $u_{\theta, t}$ is very heuristic. There is no recovery guarantee. Identifying the mixture parameters itself is not trivial. 

3. The expeirmental results is limited (lack higher dimensional case) and some parts are not very clear. For example, it is not very clear why the generated function in Fig. 2(b) is consistent with Fig. 2(a) training data. They appear to have different set sizes.

### Questions
1. For general audience, it might be a bit difficult to build a mental picture of the distribution of the functions drawn from $\eta_0$. Could the authors provide some examples, and provide some intuition on why it is a good base distribution?  

2. Does the inversion procedure produce exactly valid sets?

3. How does the performance of the proposed algorithm scale with the data $D_X$ increases? In the synthetic experiments, the authors only showed point clouds in 2D. What happens if you consider 3D point clouds, or even higher-dimensional point clouds? 

4. How does the performance scale with the number of points $N$?

5. Could the authors report computational cost (training time, inference time) vs other methods; include scaling experiments for varying set size $N$.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper extends the flow matching model to unordered point sets. It presents the problem of unordered data generation as generating a function that defines a set of Gaussian distributions centered at the observed points with adaptive variances. The authors then train a functional flow matching to generate the function and use particle filtering to zero in on the points. With experiments on the benchmark unordered point sets, they show superior performance of their method compared to recent baselines.

### Strengths
1. The method's presentation was clear. I appreciated the clarity with which each theoretical result was explained
2. The experiments do show a higher distributional similarity compared to the baselines

### Weaknesses
Major:

1. The approach appears to be very similar to that of Bilos et al., where the authors used noise from a stochastic process (a Gaussian process) to add noise to the temporal point sets. I acknowledge that the current method (unordered flow) extends this to any point set. However, the experiments in the current paper show temporal point processes only; therefore, without comparing with Bilos et al. (empirically or theoretically), it is difficult to judge the effectiveness of the current approach.

2. The current experiments section only explores unconditional generation. However, the real usefulness of a generative model is the controlled generation. Is there an advantage of this method in conditional generation vs others?

Minor:

Can the author present a quantitative comparison vs the baselines for the proof-of-concept studies?

### Questions
Comments:

Page 3 line 144 - the second Gaussian should have $y_j$ as its center.

### Soundness
3

### Presentation
3

### Contribution
2
