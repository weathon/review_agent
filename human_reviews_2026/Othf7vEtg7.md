# MoCA: Mixture-of-Components Attention for Scalable Compositional 3D Generation

- Decision: Reject
- Scores: 6, 2, 8, 4

## Abstract
Compositionality is critical for 3D object and scene generation, but existing part-aware 3D generation methods suffer from poor scalability due to quadratic global attention costs when increasing the number of components. In this work, we present MoCA, a compositional 3D generative model with two key designs: 1) importance-based component routing that selects top-k relevant components for sparse global attention, and 2) unimportant components compression that preserve contextual priors of unselected components while reducing computational complexity of global attention. With these designs, MoCA enables efficient, fine-grained compositional 3D asset creation with scalable number of components. Extensive experiments show MoCA outperforms baselines on both compositional object and scene generation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a new method for compositional 3D generation. The authors identity that part-aware 3D generation suffers from poor scalability due to the quadratic complexity of attention-based models. The authors therefore proposed to use mixture-of-experts to route the most important components and compress distant components to achieve linear complexity. The authors evaluated their method on both object-level and scene-level 3D generation tasks to demonstrate the effectiveness of their methods.

### Strengths
- The paper is well-motivated and technically sound.
- The use of MoE in 3D generation is interesting and effective.

### Weaknesses
- The generated results in Figure 3 are less detailed compared to baselines like PartCrafter. The surface is much smoother.
- The comparison is relatively limited compared to papers like PartCrafter, which compared on different datasets like Objaverse, ABO etc.
- Visual results and 3D results are fairly limited. Would love to see more 3D rendering results.

### Questions
I would love to see the authors provide more comparison and video results. I would also like to see the explanations of loss of details in reconstruction.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates part-composed 3D object and scene generation, an important and timely problem in 3D vision. The approach leverages the availability of high-quality modern 3D datasets. The authors propose a transformer-based model (arguably over-complicated) trained with flow matching to learn the distribution of shape latents. Each part is encoded and decoded in SDF representation, while image and text conditions can be incorporated into the diffusion transformer. Only a few examples are shown, also demonstrating synthetic image to compositional 3D generation.

### Strengths
- Although the method itself is standard, the results appear strong, largely due to the high quality of current datasets rather than novel modeling contributions.
- The focus on structured, compositional 3D generation is a meaningful and valuable research direction that deserves further attention.

### Weaknesses
- The transformer design lacks novelty and is quite widely used and studied in the literature of scene generation and part-based 3D object generation. Given the small latent space (fewer than 100 parts), any sufficiently large transformer could model the distribution; the architectural choices do not seem critical. I would believe this is a "fake" contribution.
- The reported results are not diverse, raising concerns about overfitting to the dataset.
- The paper only evaluates on synthetic data and generated images no real image experiments are provided.

### Questions
- Main question: What is the real contribution of this paper to the community?
- Secondary: How does the method ensure diversity and handle real data?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces MoCA, a compositional 3D generative model for efficient, scalable, and accurate compositional modeling of 3D objects and scenes.

### Strengths
1. The proposed Mixture-of-Components Attention is well motivated for addressing the quadratic global attention cost.

2. Complete ablations: All design choices (compression, gating, activation, multi-head routing) are completely ablated.

3. Strong performance: better experimental results have been observed against baselines like PartPacker, PartCrafter, MIDI

### Weaknesses
1. No apperance: It seems that all methods, including baselines, only generate meshes without textures, which might limit the real-world applications. Can the authors provide more details about this?

### Questions
Runtime analysis: It would be better to include a breakdown of runtime about different procedures of the proposed pipeline and compare against other baselines to demonstrate the efficiency.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
TLDR: using routers to select top relevant tokens between parts to avoid full attention, enables larger number of parts generation.

This paper proposes a new approach to the image to 3D parts problem. It tries to generate more parts than previous work by using routers to select top relevant tokens between parts to avoid full attention, which is inspired by the router mechanism in the Mixture-of-Experts (MoE) methods.

This can effectively reduce the memory requirement and hence support up to 30 components generation.

### Strengths
- Good motivation: The most challenging aspect of part generation is having a large number of parts. This works tries to improve performance on such important task.
- Good results. The method can generate more parts than previous work.
- Sound technical approach. The general design of the method is sound. And the idea of using routers is interesting.

### Weaknesses
- Complicated system. The system seems to be very complicated, and the technical details are hard to read. Method pipeline figure is challenging to understand - Maybe a better way is to decompose the figure into multiple figures so it is easier to understand part by part.
- Limited insight. Although the Routing mechanism seems valid, it is a general method and the author does not further utilize properties that are unique to 3D part structures.

### Questions
- Why the part geometry quality degrade significantly when the number of parts are increased? In figure 6, shapes start to be scattered and broken for part number to be 30.
- Also in figure 6, we show the parts clearly get disconnected from each other (the feet and the legs), please provide more analysis on this limitation and give potential ideas to solve
- What if you don't use routing scheme at all? How do you compare your model removed the routing mechanism. The results should be better since it will use full attention, but what will be the difference, and how do you compare the results when you have same number of parts, like 30, but with smaller number of tokens per part so the parts can fit in the memory, will it have similar results as the scattered ones you have shown in figure 6?

If the author can better understand how routing contributes to the existing approach, I would consider to raise my score.

### Soundness
3

### Presentation
3

### Contribution
2
