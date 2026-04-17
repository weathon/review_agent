# Reference Grounded Skill Discovery

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 2

## Abstract
Scaling unsupervised skill discovery algorithms to high-DoF agents remains challenging. As dimensionality increases, the exploration space grows exponentially, while the manifold of meaningful skills remains limited. Therefore, semantic meaningfulness becomes essential to effectively guide exploration in high-dimensional spaces. In this work, we present **Reference-Grounded Skill Discovery (RGSD)**, a novel algorithm that grounds skill discovery in a semantically meaningful latent space using reference data. RGSD first performs contrastive pretraining to embed motions on a unit hypersphere, clustering each reference trajectory into a distinct direction. This grounding enables skill discovery to simultaneously involve both imitation of reference behaviors and the discovery of semantically related diverse behaviors.
On a simulated SMPL humanoid with $359$-D observations and $69$-D actions, RGSD successfully imitates skills such as walking, running, punching, and sidestepping, while also discover variations of these behaviors. In downstream locomotion tasks, RGSD leverages the discovered skills to faithfully satisfy user-specified style commands and outperforms imitation-learning baselines, which often fail to maintain the commanded style.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a way to do unsupervised skill discovery based on reference motion data. The authors first learn a latent mapping function (from $\mathcal{S}$ to $\mathcal{Z}$) with contrastive learning. In the next phase, they train a DIAYN-like objective with a KL constraint to train a skill policy. As a result, they get a set of skills that are diverse while looking natural. They compare their method with two unsupervised skill discovery baselines (DIAYN and METRA) and two imitation learning baselines (ASE and CALM) in terms of $\ell^2$ distance and FID, showing that their method produces better behaviors.

### Strengths
The paper is well-motivated and generally well-written. The problem setting in this paper (i.e., how to do skill discovery based on some prior knowledge -- in this case, a reference motion dataset) is important and timely. While the individual components of the proposed method are not necessarily novel, the combination of these components has (to my knowledge) not been previously done in the literature. I also like that the proposed method is straightforward and relatively easy to implement. The authors convincingly demonstrate that RGSD is more effective than ASE and CALM.

### Weaknesses
The main weakness of this work is in the empirical comparison. While the authors compare their method against ASE and CALM, there are many more relevant previous works (essentially the whole literature on "data-driven/offline skill discovery") that are closely related to the proposed method but are not properly discussed/compared against. For example, OPAL and SPiRL also do the conceptually similar thing as this method -- they first learn a latent space (via a VAE) and then do policy learning based on the learned latent space. I know these methods are not exactly the same as the proposed one (and the problem settings are also not identical), but I think they are still close enough, to the degree that I would have expected to see comparisons with prior works in data-driven skill discovery. I also would have expected more extensive discussions of this line of research.

The paper distinguishes its method from Motivo by saying that "it does not explicitly aim to discover new skills, which distinguishes our contribution," which I find unconvincing. I agree that the latent space is not densely covered by $z$'s from the dataset trajectories, but given that the distribution of latent vectors tends to be uniform with contrastive learning (https://proceedings.mlr.press/v119/wang20k/wang20k.pdf), interpolated latents are still more or less in-distribution. Moreover, even if one treats these interpolated vectors as "new skills," the same argument could apply to other works including Motivo, FB, etc. as well, where one samples $z$'s from the prior distribution (typically a (unit) Gaussian) to train a policy. Hence, I think the authors should either provide empirical quantitative evidence that shows RGSD indeed learns more novel skills, or tone down the claim about the ability to discover "new" skills. I also think the paper could have been much stronger with comparisons with Motivo, as it is one of the most closely related works both algorithmically and empirically.

I'd be happy to adjust my score if the authors address these points.

### Questions
I don't have particular questions about this paper as of now. My main issue is the lack of discussion and empirical comparison with related work (see the weaknesses section above), which I hope the authors can address during the rebuttal period.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces Reference-Guided Skill Discovery (RGSD), a framework for reproducing and controlling diverse motions using reference data, outperforming prior imitation and unsupervised skill discovery methods. RGSD first learns a latent skill space by contrastively embedding reference trajectories, clustering each motion around a unique latent direction. It then jointly trains a skill-conditioned policy to imitate reference motions and discover novel skills by conditioning on both reference and interpolated latents. In experiments, RGSD consistently outperforms prior approaches by achieving lower Cartesian errors and maintaining high trajectory fidelity across reference trajectories, demonstrating strong improvements over both imitation-based and unsupervised skill discovery baselines in high-DoF SMPL humanoid.

### Strengths
**Strengths:**

- Clear, easy to read
- Demonstrates strong, consistent improvements over established baselines
- Provides a principled advancement over both imitation-based and unsupervised skill discovery approaches

### Weaknesses
**Weaknesses:**

- Downstream task evaluation is limited to a single, relatively simple task
- Assumes coverage of skill in the pretraining reference dataset
- Limited exploration of performance with diverse or noisy reference data

### Questions
- How does RGSD perform when reference trajectories are noisy or highly variable, as in real-world human reference trajectories due to inaccuracies in sensing or mapping?

- Can RGSD effectively scale and generalize as the number and diversity of reference trajectories increase?

- Is the set of discovered skills strictly limited to local variants of those present in the reference dataset? If yes, I would suggest clarifying this in the paper

- How does RGSD compare to frameworks like Meta-Motivo, which also address a similar problem of acquiring skills from a reference dataset, despite not focusing exclusively on skill discovery? 

- How does RGSD offer advantages over CALM? Both methods achieve similar downstream task performance. Are there settings where RGSD’s skill interpolation provides unique benefits?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
* The main contribution of this paper is a framework that improves the initialization of skill discovery by using a seed set of reference motions that the discovery algorithm first imitates, followed by an unsupervised phase that introduces novelty and diversity without straying too far from the reference behaviors. The approach relies on constructing and exploiting a semantically grounded latent space, obtained through a contrastively pretrained encoder that maps states onto a unit-sphere latent representation via a von Mises–Fisher (vMF) distribution.
In practice, the imitation and discovery processes are performed in parallel: the pretrained encoder is frozen for reward computation, while a second encoder is trained through reinforcement learning to track it under a KL-divergence penalty.

* Experimentally, on a 69-DoF SMPL humanoid with a 359-dimensional observation space, the proposed method successfully imitates the reference motions (walk, run, sidestep, backward, and punch) with low Cartesian error and achieves competitive FID metrics compared to DIAYN, METRA, ASE, and CALM. For the downstream “GoalReaching-Sidestepping” task, a high-level policy based on RGSD attains a success rate comparable to CALM, while achieving a better FID score.

### Strengths
* The paper is clearly written and well structured. The proposed method is well motivated and exhibits an elegant design, both in its overall pipeline and in the chosen geometry of the latent space. 
* The experimental section provides clear and convincing qualitative evidence supporting the effectiveness of the approach.

### Weaknesses
* Several claims in the paper appear overstated. In particular, the theoretical analysis relies on the assumption of perfect within-motion alignment, which is unlikely to hold in practice. The encoder has finite capacity, and although the InfoNCE objective promotes alignment, it does not guarantee it. Moreover, the authors introduce several heuristic mechanisms—such as artificially terminating episodes when deviations from alignment become too large—to make the method work in practice. Unfortunately, all theoretical developments are built upon this idealized assumption, and no convergence or stability bounds are provided to quantify the impact of suboptimal alignment.

* The comparisons with baselines are also somewhat unfair. For example, ASE and CALM were originally proposed with explicit diversity components, yet these elements were omitted in the reported baseline implementations. This undermines the strength of the comparative results.

* The paper further relies heavily on the Fréchet Inception Distance (FID) as a key evaluation metric, but the justification for its use in this context is insufficient. Moreover, visual inspection of the figures suggests that small FID differences sometimes contradict qualitative impressions, casting doubt on its reliability as the primary measure of performance.

* Finally, the choice of downstream task appears somewhat ad hoc. The corresponding reward function is explicitly designed to favor a latent skill close to the “sidestep” motion encoding, introducing a bias that aligns the task with the model’s strengths. In more general settings, downstream tasks will not necessarily be directional in this sense. The paper implicitly assumes that “motions” correspond directly to “directions” in latent space, which limits the generality of the approach.

### Questions
See suggestions of improvement in the "Weaknesses"  Section.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Reference-Guided Skill Discovery (RGSD), a method that enables unsupervised skill discovery to scale to high degree-of-freedom robots by grounding the exploration process in a semantically meaningful latent space derived from reference motion data. The key insight is to first use contrastive learning to embed reference motions onto a unit hypersphere, then leverage this pre-structured latent space to simultaneously imitate reference behaviors (by sampling latent vectors along reference directions) and discover novel related skills (by sampling novel directions). Applied to the SMPL humanoid, RGSD successfully learns both to reproduce complex motions like walking, running, and punching with high fidelity, and to discover semantically coherent variations of these behaviors-addressing the fundamental challenge that pure exploration in high-dimensional spaces tends to produce meaningless, unstructured behaviors while pure imitation fails to generalize beyond the reference dataset.

### Strengths
The paper provides rigorous theoretical justification for its core technical claims (Sec.A, Sec.B, Sec.C.1, etc.).

### Weaknesses
- **Unclear scope of "discovery" and potential overselling.** The paper's motivation (first version, lines 44-53) suggests learning skills beyond the reference data that are still "semantically meaningful," using the example of a manipulator learning diverse skills like pushing and grasping. However, the experimental results demonstrate a more limited form of discovery: variations of existing reference behaviors (e.g., punching in different directions, sidestepping with varying turns) rather than genuinely novel skill categories. While this is still valuable, it conflicts with the initial framing. The paper would benefit from: (a) explicitly clarifying that discovered skills remain within the semantic manifold spanned by references, or (b) experiments sampling latent vectors farther from reference embeddings to empirically characterize what "novel" behaviors emerge and whether they remain meaningful or degenerate into unstructured motion.

- **Strong assumptions on data segmentation limit practical applicability.** The method assumes each reference motion corresponds to a single, consistent skill throughout its duration. While this holds for the curated dataset used in experiments, most real-world video or robot demonstration data contains multiple skills within single trajectories and would require extensive manual segmentation and annotation. The paper provides no discussion of: (a) how to handle multi-skill demonstrations, (b) robustness to segmentation errors, or (c) potential extensions to learn from unsegmented data. This significantly limits the method's applicability beyond carefully curated motion capture datasets.

- **Insufficient downstream task evaluation.** Sec.5.3 presents only a single downstream task (goal-reaching with sidestepping), which is insufficient to validate the learned skills' general utility. Additionally, critical experimental details are missing, e.g., how to obtain the frozen $\pi_\text{low}$

- No supplementary material (e.g., videos of rollout examples) is provided.

### Questions
- **Validation of within-motion alignment.** While Appendix B proves that contrastive pretraining achieves within-motion alignment at optimality, can you provide empirical evidence that this property is achieved in practice? Specifically: (a) quantitative metrics showing the variance of embeddings $\mu_\phi(s)$ for states $s$ within the same trajectory, (b) visualization of how embeddings cluster on the unit hypersphere for different motions, and (c) analysis of whether this alignment quality correlates with downstream imitation performance. Additionally, does this alignment property degrade when sampling latent vectors far from reference embeddings during discovery?

- **Sensitivity to reference dataset composition.** The experiments use a fixed set of 20 reference motions (first version, lines 336-342). How does the method's performance scale with: (a) the number of reference motions per category (e.g., 1 vs. 5 walking motions), (b) the total number of skill categories, and (c) the diversity within each category (e.g., straight walking vs. walking with various turns)?

- **Qualitative comparison on downstream task behaviors.** While Sec.5.3 reports FID scores showing RGSD (34.3) outperforms CALM (46.7) on the goal-reaching task, could you provide visual comparisons showing the actual behavioral differences? Given that both methods achieve similar success rates, understanding how they differ qualitatively would better support the claim of learning "semantically meaningful" skills.

### Soundness
2

### Presentation
3

### Contribution
2
