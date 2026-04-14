# Flow Matching with General Discrete Paths: A Kinetic-Optimal Perspective

- Decision: Accept (Oral)
- Scores: 6, 10, 8, 6, 8

## Abstract
The design space of discrete-space diffusion or flow generative models are significantly less well-understood than their continuous-space counterparts, with many works focusing only on a simple masked construction.
In this work, we aim to take a holistic approach to the construction of discrete generative models based on continuous-time Markov chains, and for the first time, allow the use of arbitrary discrete probability paths, or colloquially, corruption processes. 
Through the lens of optimizing the symmetric kinetic energy, we propose velocity formulas that can be applied to any given probability path, completely decoupling the probability and velocity, and giving the user the freedom to specify any desirable probability path based on expert knowledge specific to the data domain. 
Furthermore, we find that a special construction of mixture probability paths optimizes the symmetric kinetic energy for the discrete case.
We empirically validate the usefulness of this new design space across multiple modalities: text generation, inorganic material generation, and image generation. We find that we can outperform the mask construction even in text with kinetic-optimal mixture paths, while we can make use of domain-specific constructions of the probability path over the visual domain.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes extending discrete flow matching paths from a kinetic optimal perspective. Previous research in discrete flow matching has primarily focused on masking-related corruption paths. In contrast, this paper demonstrates that the design space can be broadened to include arbitrary probability paths. The authors explore two types of velocities: probability-advancing and probability-preserving. They also derive a closed-form velocity for the probability-advancing type and the ELBO for general discrete flow matching models.

### Strengths
1. This paper extends the commonly used probability paths in discrete diffusion to a more general class of paths, providing a closed-form velocity and ELBO. This approach has the potential to inspire further research and improve a broad range of discrete flow matching models.

2. The proposed method is general and can recover previously proposed methods, such as Gat et al. (2024).

3. The experiments are comprehensive, covering image, text, and material generation. Results show that the proposed path achieves better performance compared to previous mixture paths.

### Weaknesses
I have some concerns regarding the experimental results:

1. The results on ImageNet 256 fall short of state-of-the-art and even popular baselines, including both the autoregressive baseline and the proposed method. The network structure follows MaskGit; however, MaskGit reports an FID of 6, which is only half of the number reported in Table 3. What accounts for this discrepancy?

2. Why did the authors train the VQ-VAE for ImageNet rather than using off-the-shelf pretrained weights?

### Questions
Please see the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
2

### Summary
This paper takes a holistic approach on discrete generative models. It pioneers the use of arbitrary discrete probability paths. The methods are validated on text generation, inorganic material generation and image generation.

### Strengths
1. This paper is very well written and succeeds in explaining concisely yet thoroughly the theory and background. It is well positioned in existing literature.

2. The authors propose holistic, novel methodologies for flow matching in discrete spaces.

3. The methods are experimentally validated on a broad set of modalities.

### Weaknesses
None.

### Questions
Is the use of the concept *flux* in this context novel, or already described in literature? If so, please provide a citation.

## Minor remarks
- There is a weird hyperlink at line 77 from the superscript 1 in $x^1$ to an equation in the appendix. This is also present in equation 6.
- line 174 'abusing a bit *the* previous ..'
- lines 202 - 206: it would be nice if the black triangles where vertically aligned

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper proposes to broaden the design space for constructing better discrete flow generative models. To this end, the authors first decouple the probability paths from the velocity, effectively allowing to choose the velocity independently of the target probability path beyond the traditional masking method. Through a kinetic-optimal perspective, they propose closed-form velocities for a certain subset of symmetric kinetic energy, and further show that the mixture path model from previous work is kinetic-optimal when used with a specific scheduler. Additionally, a new tractable ELBO for mixture paths is introduced, which generalizes previous formulations. Experiments across modalities such as text, materials, and image generation demonstrate that the proposed framework outperforms existing methods.

### Strengths
The generality of the proposed design space is definitely a significant contribution of the paper. Indeed, the new framework supports finding symmetric kinetic-optimal velocities for arbitrary discrete probability paths, expanding design flexibility over traditional masking. 

The paper has a strong theoretical foundation, and lays the ground for a more flexible approach to discrete generative modeling, which improves over state of the art, as highlighted by the experiments.

### Weaknesses
While the kinetic energy framework (Eq. 18) allows injecting problem dependent weighting $w_t(x,z)$ in theory, solving for arbitrary weights requires numerical approximation in practice, which can be challenging as mentioned by the authors. This can limit the practical potential of the design space.

### Questions
I would be interested to know how do DFM approaches compare with FM approaches in the current state of the art. For example, with equal training time and computational resources, how does discrete flow matching compare with continuous flow matching in the task of pixel space image generation?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Most existing methods rely on masking for the corruption process, which leaves the design space of discrete diffusion and flows underexplored. This paper introduces a new design space by optimizing symmetric kinetic energy, allowing the proposed velocities to apply across multiple probability paths. Thus, the proposed discrete flow matching method enables desired probability paths for domain-specific applications. The effectiveness of this approach is validated across various modalities — including text generation, material generation, and image generation — where it outperforms mask-based methods, especially in language tasks, and achieves a new state-of-the-art in crystalline material generation.

### Strengths
1. The proposed method, Discrete Flow Matching (DFM), is theoretically well-justified through the use of kinetic optimal velocities and probability paths.

2. Extensive evaluations on text, crystalline material, and image generation tasks demonstrate the effectiveness of the proposed velocity formulation, achieving state-of-the-art results in material generation.

3. The construction of an infinite number of velocities for any chosen probability path broadens the design space for probability paths, enhancing applications for downstream tasks.

### Weaknesses
1. Typo in Line 50: “misleadingly actually” --> "actually misleading"

2. The velocity $u_t(x,z)$ is only defined for $x\neq z$ in Eq. (5).  The authors need to explicitly define $u_t(x,z)$ for the case where $x=z$, or explain why this case does not need to be defined separately.

3. What is $\bar{i}$ in Line 117? Define this when it is first introduced in the paper.

4. Line 227: Explicitly explain the relationship between the symmetric condition mentioned and the detailed balance equation in Markov Chains, if any. This would help clarify the theoretical foundations of the proposed method.

5. The authors are requested to provide a brief comparison of the key mathematical or conceptual differences between their kinetic optimal velocity formulation and the approach used in SEDD [1].

6. Typo in Line 511: “can be provide” --> "can provide"

7. Typo in Line 528: “We additional show generated” --> “We additionally show generated”

Overall, this paper is well-written with sound theoretical justification and experimental results supporting its main claims. The reviewer is unfamiliar with some related work (e.g. crystalline material generation) and is open to discussion with other reviewers/authors for a more comprehensive assessment.

### Questions
Please see the weaknesses section above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper broadens discrete generative modeling by introducing discrete Flow Matching models with flexible, iterative probability paths using continuous-time Markov chains (CTMC). Unlike conventional masked models, this approach enables optimized, domain-specific probability paths.
Key contributions:
A decomposition of path velocities, making generation simpler.
Closed-form velocities that allow any probability path, unifying and expanding prior methods.
A kinetic energy-based optimization of paths, leading to new, source-dependent schedulers.
A general ELBO that improves performance on mixture paths, surpassing masked models.

### Strengths
I generally like this paper; it establishes a new framework for discrete flow matching and provides a method for constructing kinetic-optimal probability paths. The experiments across three downstream applications effectively demonstrate its real-world utility.

### Weaknesses
1. Is the construction in Section 4.2 unique, and is it general enough?
2. How does the proposed method relate to discrete rectified flow, and how do their equilibrium probability paths differ when using the same source and target domains?

### Questions
Could you provide more background on the mixture path and explain how it connects to the claim in Section 4.2?

### Soundness
3

### Presentation
3

### Contribution
3
