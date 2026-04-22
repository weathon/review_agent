# ARC-AGI Without Pretraining

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 2, 4, 2

## Abstract
Conventional wisdom in the age of LLMs dictates that solving IQ-test-like visual puzzles from the ARC-AGI-1 benchmark requires capabilities derived from massive pretraining. To counter this, we introduce *CompressARC*, a 76K parameter model without any pretraining that solves 20\% of evaluation puzzles by minimizing the description length (MDL) of the target puzzle purely during inference time. The MDL endows CompressARC with extreme generalization abilities typically unheard of in deep learning. To our knowledge, CompressARC is the only deep learning method for ARC-AGI where training happens only on one sample: the target inference puzzle itself, with the final solution information removed. Moreover, CompressARC does not train on the pre-provided ARC-AGI "training set". Under these extremely data-limited conditions, we do not ordinarily expect any puzzles to be solvable at all. Yet CompressARC still solves a diverse distribution of creative ARC-AGI puzzles, suggesting MDL to be an alternative, highly feasible way to produce intelligence, besides conventional massive pretraining.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work introduces a method for solving ARC-AGI-1 evaluation puzzles, based on minimum description length approach (i.e., the so-called code golf). It is worthy of noting that this work does not requite any pretraining, and only trains a rather small neural network during inference time, showing potentials for efficiency and the rid of massive LLMs training.

### Strengths
As shown in the summary:
1. A novel inference-time perspective for ARC-AGI, as the pertaining is omitted.
2. Rich discussions on the proposed schemes in the solving process and explanations.
3. Clarity for pros and cons in the work.

### Weaknesses
1. The proposed pipeline may lack formal theoretical analyses for the mathematical  fundaments (which though I think applies to other related methods in this field).
2. Is it possible to apply MDL to other tasks? How and why is possible/promising?
3. It seems that the running time for (total iterations and per iteration) remain possible to be greatly improved. How do the author explain such cost/efficiency and what can be done for improvement?
4. Despite its focus on the pretraining-free advanatage, it would still be nice to have more baselines(related work) comparisons and discussions on the main context.

### Questions
See the weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces a lightweight CompressARC to solve puzzles from the ARC-AGI-1 benchmark without any pretraining. By applying the Minimum Description Length (MDL) principle during inference, the model learns purely from the target puzzle itself, without using any training set. The results suggest MDL as a data-efficient solution for solving puzzles.

### Strengths
1.	The research addresses a compelling and timely question on data-efficient methods for solving ARC-AGI puzzles, offering a novel alternative to large-scale pretraining.
2.	The manuscript and supplementary materials provide comprehensive implementation details, ensuring reproducibility and methodological clarity.
3.	The experimental results are supported by in-depth analysis of the proposed method.

### Weaknesses
1.	The emphasis on "no pretraining" and single-puzzle focus does not sufficiently establish methodological novelty; it remains unclear whether performance stems from genuine innovation or inherent dataset shortcuts.
2.	Claims of novelty are inadequately supported by architectural or algorithmic innovation, as the model relies on established components without clear differentiation.
3.	Section 3 lacks intuitive motivation, proceeding directly into technical details without conceptual justification, hindering reader comprehension.
4.	The three core algorithms appear to lack substantive novelty, with insufficient emphasis on what constitutes a technical advance.
5.	The network architecture is relatively conventional, predominantly building on widely adopted components.
6.	Comparative evaluation with state-of-the-art methods is absent, limiting validation of claimed advantages.
7.	Generalization claims are not fully convincing, as the method may exploit dataset-specific shortcuts rather than learning generalizable reasoning, limiting applicability to tasks like RAVENs.

### Questions
1.	What specifically constitutes the novel component(s) in the proposed architecture or algorithm?
2.	Appendix K offers supplementary dataset details but lacks illustrative puzzle examples such as the structure of the question panel, the format of a correct solution, and the criteria for matching a proposed answer to the ground truth. Including such examples would significantly improve readers' understanding of the task and the proposed method's problem-solving process.
3.	During inference-time learning, are solutions to training/test samples used to fine-tune the model parameters?
4.	How are "steps" and "attempts" formally defined? What operations occur per step, and what constitutes an attempt?
5.	How does the method compare to existing few-shot or zero-shot learning approaches on comparable tasks?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes CompressARC, a 76K-parameter neural model that tackles the ARC-AGI-1 benchmark without any pretraining or usage of the provided training set. The model applies the Minimum Description Length (MDL) principle to infer solutions purely at inference time, treating ARC puzzles as a code-golfing problem—that is, searching for the shortest program capable of reproducing the dataset. Despite its simplicity, CompressARC achieves 20% accuracy on the evaluation set and 34.75% on the training set, which is remarkable given that no pretraining or external data are used.

### Strengths
1. The framing of ARC-AGI solving as a code-golfing / MDL minimization problem is deeply original.
2. The implementation of inference-time learning via MDL provides a clean and theoretically motivated path to “training-free” intelligence.

### Weaknesses
1. Although the 20% accuracy result is interesting, it remains far from state-of-the-art (50%+).
2. Each puzzle takes 20 minutes and 2000 steps of inference-time optimization, which raises scalability and practicality concerns.
3. The discussion contrasts CompressARC mainly with large pretrained models (LLMs) but does not include comparisons to smaller ARC solvers or neuro-symbolic baselines.

### Questions
1. Compare with baseline models (random, heuristic, small CNN/VAE) under identical inference-time constraints.
2. Explicitly mention the restricted scalability and the dependency on ARC’s small grid size; discuss whether CompressARC could generalize to larger, more open-ended domains.

### Soundness
2

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
2

### Summary
The paper poses solving ARC-AGI problems as a search for a minimal program that reproduces the example outputs given the example inputs. The program found can then be applied to new inputs and its output be the solution to those new inputs. The benefit of this approach is it does not require one to train on a large dataset beforehand as training is done on a per-instance basis.

### Strengths
The justification for the approach using Occam’s razor is convincing. The ability to solve problems without pre-training is also appealing. Informative case studies are performed on success and failure cases.

### Weaknesses
The paper describes their approach in Algorithms 1, 2, and 3. There is a lot that is not clear in these algorithms. 

-	These algorithms seek to minimize the seed length used. What is the relation between seed length and program complexity?

-	There are three seeds mentioned in Algorithm1 (seed_z1, seed_error, and seed_z2). Which seed length is the one that is to be minimized?

-	Algorithm 2 says “Measure n_exmpl,n_colors,width,height from P to initialize equivariant_NN”. What is meant by “measure”? How is this to be used to initialize the neural network? What is the initialization procedure?

-	How are \mu and \Sigma initialized?

The paper also does not contextualize the results in the broader research landscape. For someome unfamiliar with the ARC-AGI benchmark, one cannot know how these results compare to other existing results, especially for those that do not rely on pre-training. Is this method the only one that does not rely on pretraining?

### Questions
How does this approach compare to other methods that use pretraining and those that do not use pretraining?

See other questions in Weaknesses section.

### Soundness
2

### Presentation
1

### Contribution
2
