# Learning Riemannian Metrics for Interpolating Animations

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 1, 3, 5

## Abstract
We propose a new family of geodesic interpolation techniques to perform upsampling of low frame rate animations to high frame rates. This approach has important applications for: (i) creative design, as it provides a diversity of interpolation methods for digital animators; and (ii) compression, as an original high frame rate animation can be recovered with high accuracy from its subsampled version. Specifically, we upsample low frame rate animations by interpolating the rotations of an animated character's bones along geodesics in the Lie group $SO(3)$ for different invariant Riemannian metrics. For compression, we propose an optimization technique that selects the Riemannian metric whose geodesic most faithfully represent the original animation. We demonstrate the advantages of our approach compared to existing interpolation techniques in digital animation.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper has presented an interesting and inspiring study. It presented a method for animation interpolation using geodesics on Riemannian manifolds. This is the first time that Riemannian metric learning has been proposed for computer graphics. This method has shown very good qualitative results and good quantitative results.

### Strengths
The method is interesting, I like the math behind it. The results especially the video look quite nice. The amount of parameters need to be learnt is limited thus it should be much easier to interpret.

### Weaknesses
I didn't see the code, so I am not able to fully understand the full math behind it and how they actually implemented it. 
I am not sure why the quantitative results are not that outstanding. From the videos, the proposed new method seems to be a much better method or kind of revolutionary method, however, the quantitative results do not show this when compared with the Slerp method. And to my knowledge Slerp is based on the special Riemannian manifold of a sphere on which the SO(3) naturally lay. Thus the metric used in Slerp is naturally based on the angles, it will be much more obvious the effect about the Riemannian metric learning, if the learnt metrics are written out and the difference between them compared to the one used in Slerp is discussed. So maybe we can see why the quantitative results are so similar. If the learnt metric is same or very similar to the Slerp one then maybe it is not that useful.

### Questions
1. I think more explanations and analysis of why the criterion of Eq. 6 is not a closed form is preferred in the appendix. Since I didn't see the code, in my naive thought, it is just matrices multiplications, thus we shall be able to use gradient optimization. Or actually, the computation of the initial tangent vector u0 = γ ̇(0) is involved in the whole optimization process, so it is not possible to use gradient-based optimization?

2. I would really prefer a runnable code with all the environment and data needed to fully understand the paper. With them, I think maybe this paper can have even higher scores. 

3. The quantitative analysis is not sound, it might be rooted in the way of the quantitative measurement. I suggest try other measurements as well, since the qualitative results seem to be quite good.

4. I want to see the effect about the Riemannian metric learning by comparing the one used in Slerp and the learnt ones.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The authors present a new technique of SO(3) interpolation parameterized by a family of Riemannian metrics. The method learns the Riemannian metric that best describes a given ground truth animation, which is a sequence of SO(3) associated with a character's rig. The metric has three parameters, α and β for the diagonal matrices used for the metric, and a categorical parameter "inv" for invariance.
The authors adopt the Tree-Structured Parzen Estimator as a gradient-free optimization method.
The authors use a public data from Mixamo animations to compare the method with the piecewise constant, Cartesian linear interpolation, and spherical linear interpolation (slerp).

### Strengths
- The authors present a simple geodesic of interpolation method that can be addressed as a useful extension of Slerp.
- The mathematics behind the proposed method is well explained.

### Weaknesses
The second contribution is not well presented. The paper significantly lacks experimental analysis to support the effect of α and β. The guideline is also unclear.
As far as I understand, the proposed method requires the ground truth animation to learn the parameters, so the method is usable for compression but not for creative design. Even as a compression method, there is no experiment and discussion about the relationship between compression rate and accuracy degradation.
The authors raised two important points as future work: further analysis of the metric parameters and full control over the parameters. These points are definitely crucial and necessary for acceptance.
According to the above issues, section 2 about related work is not appropriate in its current form.
Since many methods are not appropriately addressed, some of them should still be comparable in the given problem setting. 


Detailed comments.
- There is an unnecessary term in the caption of Figure 3.
- "whether whether" at the bottom of page 6.
- The authors introduced "weight" in equation 9, but did not evaluate it.
- The authors simply took a weighted metric $Q_{hyb}$ as 0.5 $Q_{loc}$ + 0.5 $Q_{rot}$, but the former depends on the scaling of the scene and the latter is scale-invariant. The current result should also depend on the scale of the scene, which is inappropriate.
- Although the authors show a quantitative result, it seems inappropriate. The authors' statement means that the quantitative result does not reflect the quality: "Despite the seemingly small quantitative difference between these two, we note that Fig. 5d shows significant perceptually differences." 
- The statement lacks support: "we are able to accurately represent a high frame rate animation with very few frames, we achieve a compression rate that requires digital animators to pose fewer keyframes during".

### Questions
The reviewer also has a question about the third contribution. The authors claim the need for gradient-free optimization as "This criterion does not have a closed form as a function of α, β and inv. Thus, we cannot compute its gradient, nor leverage any gradient-based optimization methods". I'm not confident since the formulation about A_{U}( α, β, inv ) is not clearly given, however, in the reviewers' understanding, the formulation is differentiable with α, β (using autograd techniques). Please clarify how it is not differentiable. As for inv, it would clearly be indifferentiable, but just solving both cases and choosing the better one is enough.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes to learn a metric to interpolate poses represented by the tree structure of joint rotations as used in computer animation. Specifically, the metric is defined as the product of the geodesics over the scaled Riemannian manifold of individual joint rotations. The authors propose to use TPE to optimize the scaling parameters (alpha and beta per joint) and the left and right invariance. The paper shows the results of the pose interpolations compared to other basic interpolation strategies (piecewise constant, linear cartesian, and spherical interpolation).

### Strengths
The metric learning of the poses in the skeletal structures as used in the digital humans and character representations is an important problem.

### Weaknesses
The interpolation of poses defined by the tree of SO(3)s using the Riemannian manifold is a well-studied topic, especially in computer graphics. "Practical Parameterization of Rotations Using the Exponential Map" [Grassia 1998] is one such paper that became the foundation of the pose representation for many papers. For example, "Geostatistical Motion Interpolation" [Mukai and Kuriyama 2005], which the authors have cited but simply disregard because it is not implemented in the animation software, uses the concatenation of the exponential maps of the joint rotations as the pose representation. "SMPL: A Skinned Multi-Person Linear Model" [Loper et al. 2015] also uses the rotation vector (the axis vector divided by the angle component in the axis-angle representation) per joint, which is equivalent to the exponential map (the Riemannian manifold) of rotations. I wish the authors had paid attention to how these previous papers treated this problem in more depth.

More specifically, the authors' proposal of collapsing the pose representation by taking the product of the joint rotations (pose Lie group in definition 3.4) does not make sense. Intuitively, if this is a single chain of joints, the product is the rotation of the last joint relative to the coordinate frame of the first joint. Even if we assume the single chain structure, the product represents only the last joint's rotation relative to some coordinate frame and is not representative of the full pose. Even worse, because the joints are in the tree structure, taking the product of all joint rotations does not make sense at all (e.g., the left foot is not chained from the left wrist.)

Therefore, it makes much more sense to treat the poses in high-dimensional space of the concatenation of the joint transform representations (e.g. exponential maps) and optimize the interpolation parameters in this space, which is essentially [Mukai and Kuriyama 2005] and many data-driven motion papers do. The previous approaches, e.g. [Mukai and Kuriyama 2005], can be seen as learning the scaling parameters of this paper (alphas and betas) but keeping the individual components.

If authors argue that taking the product of all joint rotations makes sense, this has to be shown in numbers and visuals, compared to methods such as [Mukai and Kuriyama 2005] with more examples.

### Questions
N/A

### Soundness
1 poor

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a method for interpolating between character animation poses (i.e. keyframes).  Given a ground truth animation sequence, they propose to optimize for a Riemannian metric on the space of poses (the Lie group $SO(3)^B$), which minimizes a quality metric between the ground truth animation sequence, and one that has been first downsampled then upsampled by interpolating between keyframes along a geodesic path on a manifold equipped with the metric.  They provide comparisons between their interpolation scheme, and other well-known methods, and assert that their interpolation scheme can also be applied toward animation sequence compression.

### Strengths
The paper is well-written and offers strong exposition containing relevant mathematical background.  Building off of that groundwork, the ideas presented in the paper are clear and well-formulated.  The established problem space of animation interpolation is interesting (of particular interest to a computer graphics audience).  Their proposed method is concise, and addresses a targeted, clearly-identified problem.

### Weaknesses
The paper claims that their method may provide a "novel diversity of inbetweening methods for digital creators", yet it is not clear to me that their method has actually come close to achieving this.  No user studies are provided to support the claim, and no comparison _within_ their method between different choices of metrics is provided to illustrate how different metrics may result in different creative outcomes.  Further, I am skeptical that selecting a particular Riemannian metric would be an ideal interface for an artist to use in their workflow.

Evaluation is fairly limited, showing results on only a small set of motions.  Qualitative results are limited and not validated for statistical significance (e.g. via a user study).  I personally find that overlaying the original animation on top of the upsampled animation makes it difficult to appreciate the quality of the result.  Also, quantitative results (e.g. those shown in Figure 6 in the main paper) suggest that their method is only incrementally preferable to the leading alternative (slerp).

Though I appreciate that the authors convey that their proposed approach is compatible with a variety of quality metrics, and that they have presented multiple such metrics, more comparison between the metrics would be helpful, so that the reader may know the authors' decisive/data-driven opinion on which metric is preferable.

### Questions
Regarding the optimization method chosen for this particular problem, are there any limitations, such as defining the search space for initial values of the parameters?

It seems that the learned metric would be dependent on the specific character rig as well as which set of animations are included in the optimization.  Could you comment on how this learned metric would generalize to other rigs and/or animations?

Given the memory requirements to store character animation data, it's not clear to me that compression is a compelling motivation.  That is to say, keyframe animation data seems to be on a MUCH smaller scale than other forms of data e.g. video or web-search indexes.  Could you elaborate upon whether compression of keyframe animation data is an established problem for real-world practitioners?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
