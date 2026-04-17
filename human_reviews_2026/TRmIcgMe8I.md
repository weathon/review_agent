# Concept-TRAK: Understanding how diffusion models learn concepts through concept attribution

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
While diffusion models excel at image generation, their growing adoption raises critical concerns about copyright issues and model transparency. Existing attribution methods identify training examples influencing an entire image, but fall short in isolating contributions to specific elements, such as styles or objects, that are of primary concern to stakeholders. To address this gap, we introduce _concept-level attribution_ through a novel method called _Concept-TRAK_, which extends influence functions with a key innovation: specialized training and utility loss functions designed to isolate concept-specific influences rather than overall reconstruction quality. We evaluate Concept-TRAK on novel concept attribution benchmarks using Synthetic and CelebA-HQ datasets, as well as the established AbC benchmark, showing substantial improvements over prior methods in concept-level attribution scenarios. We further demonstrate its versatility on real-world text-to-image generation with compositional and multi-concept prompts.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces Concept-TRAK, which attributes training samples that taught a specific concept in diffusion models by aligning a per-sample training gradient with a concept gradient. They first introduce losses that operate through reward gradients in the tangent space and construct an influence function framework based on these losses. Evaluations are performed on controlled synthetic datasets as well as CelebA dataset and AbC benchmark for data attribution evaluations.

### Strengths
The paper proposes a novel problem where the focus is to pinpoint which training images taught a specific concept rather than just influencing an entire image.

### Weaknesses
1. The evaluations are very narrow and limited with respect to the concepts used. In real-world settings, the concepts can be highly compositional and overlapping, whereas the concepts used in the paper are simple and clean, which doesn’t reflect the real-world setting.

2. The paper lacks a human evaluation for the applications shown in Fig. 8. It is important to analyse if the retrieved samples actually reflect the target concept in these cases. Adding a human judgment study would strengthen the claims.

### Questions
1. I did not see any experiments that show the computational effectiveness of the approach. How much time does the approach take for a single concept?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces Concept-TRAK, a novel framework for concept-level attribution in diffusion models. The method reformulates influence functions through a reward optimization framework, where a concept-aware loss is defined using gradients along the tangent space of the diffusion manifold. The authors leverage Explicit Score Matching to approximate concept-relevant gradients and compute per-sample attributions without re-training the diffusion model. Empirical studies on synthetic datasets, CelebA-HQ, and the AbC benchmark demonstrate that Concept-TRAK outperforms baselines (TRAK, D-TRAK, DAS) both in in-distribution and out-of-distribution concept attribution, achieving substantially higher performance.

### Strengths
1. The paper moves beyond image-level attribution toward concept-level interpretability, a direction of genuine novelty and relevance for auditing large-scale generative models.
2. The integration of reward optimization and influence functions is conceptually elegant.
3. The method has potential applications in copyright auditing and transparency for generative AI, aligning with broader research interests in responsible model development.

### Weaknesses
1. The approach likely has high computational complexity due to Hessian approximations and multiple gradient evaluations per sample.
2. The synthetic dataset uses only a 2×2 factor combination; CelebA-HQ omits a single triple attribute; AbC relies on textual inversion with a frozen base model. These experiments demonstrate the idea but fall short of realistic large-scale diffusion scenarios involving diverse prompts and complex concept interactions.
3. The method is described as “training a diffusion model on a reward-shaped distribution,” but in reality, Concept-TRAK operates post-hoc on pretrained models without updating parameters. This wording could mislead readers regarding optimization vs. analysis.

### Questions
1. What is the time and memory cost of running Concept-TRAK on a large model like Stable Diffusion XL or SD 2.1?
2. How does attribution behave when concepts overlap semantically?
3. In prompts with multiple intertwined concepts (e.g., “a cat in Van Gogh style”), how is the negative concept constructed, and how does Concept-TRAK avoid misattributing correlated concepts (cat vs. tiger, style vs. object)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel data attribution method for diffusion models designed to identify which training samples influenced a specific concept within a generated image. The key innovation is the use of specialized, reward-based loss functions for both the training sample and the target concept, which are designed to capture concept-relevant directions in the model's tangent space. Experiments show that Concept-TRAK substantially outperforms prior methods in these concept-level attribution scenarios.

### Strengths
- The paper addresses a significant and overlooked limitation in existing data attribution for diffusion models, which is attributing influence for a *specific* concept.
- The paper conducts controlled experiments and shows that Concept-TRAK performs well in both ID and OOD settings.

### Weaknesses
- If I understand correctly, other methods only allow attribution to a specific generated image. How do you evaluate other methods in the experiments? A naive way is to find influential training data to a set of generated images containing the target concept. Is this how you conduct the experiment? If not, the paper should include that as well.
- The paper does not specify which model version is evaluated in the paper. A thorough experiment with many model types, such as SD 1.4, SD v2, SD XL, Flux, DeepFloyd, etc. would strengthen the claim

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3
