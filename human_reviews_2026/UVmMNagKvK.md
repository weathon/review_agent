# StructEval: A Benchmark for Evaluating Generation, Inference, and Reconstruction in Atomic and Crystalline Structures

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Crystalline materials power technologies from solar conversion to catalysis, yet current machine learning evaluations artificially divide infinite lattices and finite nanoclusters into separate domains. StructEval unifies these regimes with a symmetry-aware, radius-resolved framework that systematically links primitive unit cells to nanoparticles across ten industrially critical compounds. Each material includes 20+ radii configurations between 0.6--3.0 nm, sampled at 780 quasi-uniform orientations, producing nearly $200,000$ structures spanning $55-11,298$ atoms. StructEval defines three rigorous challenges--unit cell $\rightarrow$ nanoparticle generation, nanoparticle $\rightarrow$ unit cell resolution, and lattice reconstruction--supported by a leakage-free split that isolates orientation interpolation from radius extrapolation. This design enables precise measurement of generalization across scale and symmetry. Benchmarking leading generative models reveals severe breakdowns under out-of-distribution conditions, exposing a fundamental gap in current architectures. By providing a reproducible, geometry-grounded testbed, StructEval establishes the foundation for next-generation generative, inference, and reconstruction models in crystalline systems. Data and implementations are released at https://anonymous.4open.science/r/StructEval-ANONYMOUS.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper provides a benchmark suite for nanostructure-related generative modeling/prediction tasks. In particular, it evaluates the ability of models to relate unit cell structure to varying nanostructure configurations. It also carefully considers the construction of nanostructures by varying the rotation and radius size. It also takes care to provide benchmark evaluations that assess OOD adaptation, finding that the popular models have issues there.

### Strengths
It is important to analyze and generate nanostructures. In general, most papers in the ML for materials community focus on periodic crystals, although many devices and systems of interest do not satisfy these ideal periodic crystal constraints. Rather, real-world systems involve nanostructures, so I like that this paper is tilting the focus of the community towards tasks involving nanostructures.

I also like that the construction of nanostructures is carefully considered by varying the central atom, radius, orientation. It is also good that the paper points out a shortcoming of existing generative models in handling OOD data.

### Weaknesses
- Why do you only do ten materials? Surely the pipeline makes it easy(-ish) to simulate more materials.
- Have you considered XRD related tasks? [1]
- Why do you not have property prediction from nanostructure as part of the benchmark suite?
- Please do a more extensive literature search for recent ML papers that address nanostructures, such as [1].
- Please fix line spacing.

[1] Guo, G., Saidi, T.L., Terban, M.W. et al. Ab initio structure solutions from nanocrystalline powder diffraction data via diffusion models. Nat. Mater. (2025). https://doi.org/10.1038/s41563-025-02220-y

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a benchmark for evaluating generative models of crystals across different scales and orientations. The benchmark consists of creating nanoparticles and three challenging tasks: predicting the lattice from nanoparticles, predicting nanoparticles from lattices, and predicting atom positions within masked regions of nanoparticles. Finally, the paper shows that current models perform worse on challenging, out-of-distribution (OOD) tasks.

### Strengths
- The paper is well-structured, well-written, and the motivation is clear.
- The benchmark contains a train, validation, in-distribution test, and an out-of-distribution test set, which is rare in current benchmarks, and quite helpful in analyzing the behaviours of models.
- The created benchmark is quite large and diverse, and the construction method can also be easily extended to more samples.
- The paper benchmarks a wide range of existing generative models and highlights their inability to perform well in the proposed tasks.

### Weaknesses
I have some moderate concerns with the submission:
- missing important existing symmetry-aware generative models
- some conceptual questions about a few design choices
- formatting concerns

These weaknesses are supported by the questions below. I am willing to increase the scores during the discussion period if these are adequately addressed.

### Questions
- **Formatting**: I believe that the gap between paragraphs (for instance, between the first two paragraphs in the Introduction section) has been reduced, which goes against the standard formatting requirements. Can you modify it to adhere to the original instructions?
- The benchmarking suggests that it is crucial to design symmetry-aware generative models. However, many such bodies of work exist [1,2,3]. I would recommend benchmarking these methods to have a holistic overview of the current models and their performance on your proposed tasks.
- What is the importance of section 2.2?
- Given the poor performance across all the tasks and models, is it possible that the proposed tasks are extremely challenging?
  - For instance, the models use the SchNet encoder/model at the core; can it handle such large atomic systems? Maybe the poor performance is due to the underlying architectures rather than the training paradigms? 
  - Is there a solvability study of the task, i.e., is there an intuition about whether the proposed tasks can indeed be solved?
  - How real (or close to ground truth) are the generated nanoparticles (or the configurations)? Is it possible to show some distribution metrics with existing nanoparticles?
  - Can there be a simpler version of the proposed tasks, i.e., maybe nanoparticle $\rightarrow$ lattice inference be posed as a classification task (since we know the underlying 10 lattices and their space groups)?
- How were FlowLLM models trained, given that it is significantly more time-consuming than other models? How was the training budget for each model decided, given the differences in training time (s/epoch) and compute requirement?
- How was the training set adapted for each downstream task, i.e., what were the training objectives? 

1. Space Group Constrained Crystal Generation. Jiao et al., ICLR 2024
2. SymmCD: Symmetry-Preserving Crystal Generation with Diffusion Models. Levy et al., ICLR 2025.
3. WyckoffDiff--A Generative Diffusion Model for Crystal Symmetry. Kelvinius et al., ICML 2025.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a new benchmark which targets the domain of nanoparticles. They define three different tasks and benchmark on these tasks. 

While this could potentially be an interesting application, I find the paper difficult to read and follow with limited descriptions, inconsistent notation, and many design choices not being described. Additionally, the actual results are a bit suspicious, and I feel the authors are not presenting compelling analysis and explanation for the numbers. More details can be found in “Weaknesses” and “Questions”. 

Due to the difficulties of following what is being done and the evaluation, I have rated this paper as a rejection. I also think that this paper could be better suited for a journal (potentially in materials science) which allows for more space (and potential a different target audience).

### Strengths
An application which seems to be new and unexplored, and which could provide interesting development of machine learning methods, suited for the task

### Weaknesses
**The paper is breaking the ICLR formatting, which requires paragraphs being separated by 1/2 line space**.

The paper is difficult to understand. For example, I think sections 2.1 and 2.2 are not written for an ML conference, but a materials science journal. In section 4, it is often talked about results and “the model”, but is unclear which model is talked about, and I cannot seem to find the mentioned results in the appendix. I have given examples of questions in “Questions”. I also think that the figures have way to small text.

The evaluation seems a bit suspicious, and I would be careful in interpreting the results. In general, in table 1-3 in appendix, exact numbers are repeating, which to me indicates that the models haven’t learned anything but just output randomness, leading to the exact same numbers in the evaluation. I think this is important to address and find an explanation before drawing any conclusions. In Line 442 the authors say “and the near-identical RMSE shows lattice-parameter recovery generalizes even to larger clusters”, but I would say that it could also be an indication that it is just pure guesses and randomness that is coming out of the model. The 0.0 % accuracy could further hint at it being the latter, that it is just pure randomness and therefore the accuracy is 0.0.

### Questions
Line 227: After reading through the paragraph I understand what N is, but when starting reading it is not clear what N is. 
Regarding RMSD: how do they get the indices of the points in the two different materials? I.e., how do they know that a specific point in P corresponds to a specific point in P\*? 

Section 3.2. I am still not sure why they need the rotations. How are the rotations used in practice? Isn’t there a way of finding a rotation such that two point clouds are as closely aligned as possible, and therefore make the evaluation invariant to rotations?

Section 3.3: In section 3.1 the notation R is a radius, in section 3.2 it is a rotation, and in section 3.3 I think it is a radius, as you use $R\leq 24$, indicating a single value which supports $\leq$. But I don’t understand the notation R6, R24 etc…? Also, what is b?

Line 323: is the “deterministic spherical truncation pipeline” explained somewhere?

General question: the chosen baselines methods are designed for generation of periodic materials, i.e., unit cells. First off, how do they adapt them to this task? I.e., how is the generation performed in practice? Second question, is it reasonable to expect the models to perform well?

How is “nanoparticle to lattice” carried out in practice? If I use for example DiffCSP for this task, how is it used? I.e., how is the function $f_2$ constructed?

Line 363: how is lattice reconstruction carried out in practice? This looks like an “inpainting” task, which a lot of research has been put into in other domains. Which method is used here?

Section 4.1: this is very difficult to understand. When it is talked about degradation: any specific model that has this degradation, is it in general or average, or something else? Also, I cannot find the numbers in the appendix. Where can I see that the “convex hull deviation” goes from 31.15 to 178.6 when going from ID to OOD?

Section 4.2 and 4.3 “the model” is mentioned, but I don’t understand which model this refers to, and on line 441, “the classifier” is mentioned, but I don’t see where this has been described earlier.

Minor: 
Line 165: Matbench not correctly cited: Dunn, A., Wang, Q., Ganose, A., Dopp, D., Jain, A. Benchmarking Materials Property Prediction Methods: The Matbench Test Set and Automatminer Reference Algorithm. npj Computational Materials 6, 138 (2020). https://doi.org/10.1038/s41524-020-00406-3

### Soundness
1

### Presentation
1

### Contribution
1
