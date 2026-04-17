# Visual Perceptual to Conceptual First-Order Rule Learning Networks

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Learning rules plays a crucial role in deep learning, particularly in explainable artificial intelligence and enhancing the reasoning capabilities of large language models. While existing rule learning methods are primarily designed for symbolic data, learning rules from image data without supporting image labels and automatically inventing predicates remains a challenge. In this paper, we tackle these inductive rule learning problems from images with a framework called γILP, which provides a fully differentiable pipeline from image constant substitution to rule structure induction. Extensive experiments demonstrate that γILP achieves strong performance not only on classical symbolic relational datasets but also on relational image data and pure image datasets, such as Kandinsky patterns.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the problem of learning explainable, first-order logical rules directly from raw images without relying on any image labels. The proposed method, γILP, is a differentiable framework that trains a deep clustering module and a differentiable logic learner, forcing the system to discover concepts that are both visually and logically consistent. The main results show that γILP can successfully learn rules from unlabeled images competitive with or better than certain LLMs, and can match the accuracy of black-box classifiers (like ResNet) on abstract tasks while also inventing the correct, human-readable rules for its decisions.

### Strengths
Learning interpretable rules directly from perceptual data is a hard problem and this paper proposes a valid solution for it.

Interpretebility has been a limitation for deep learning methods, this paper addresses this by combining symbolic approaches with deep learning to learn interpretable rules.

The interplay between symbolic and deep learning based methods is well done and leads to human interpretable rules.

### Weaknesses
I believe one limitation is its not clear to me how this approach can scale to real-world images where 
1. Clustering may not work well
2. rules may not be very clear and interpretable

Also, the number of cluster should be known beforehand which seems like a limitation

### Questions
A clarification on the LLM baseline experiments How many example images from a latent cluster were typically fed into the LLM? Given that a large number of images could exhaust the context window, how was this image sampling handled? Did the authors investigate the sensitivity of the LLM to this?

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
3

### Summary
The paper proposes a neuro‑symbolic framework for learning first‑order logic rules directly from images without explicit labels. The pipeline couples (1) a perceptual front‑end (pre‑trained encoders), (2) a differentiable clustering module that induces latent “symbolic” entities, (3) a differentiable substitution mechanism that builds training examples for (4) a differentiable rule‑learning network, and (5) an LLM‑based predicate invention step that maps placeholder predicates to human‑readable semantics. Experiments span classical ILP (Inductive logic programming) tasks, relational MNIST variants, and Kandinsky patterns. The system shows encouraging results on visual reasoning benchmarks, with qualitative examples of invented predicates.

### Strengths
- A coherent architecture that connects perception, clustering as latent symbol induction, and differentiable rule learning, culminating in interpretable rules. The overall design targets a long‑standing neuro‑symbolic goal—bridging pixels to logic. 

- On compositional visual reasoning (e.g., two‑pair), the method outperforms standard vision baselines and illustrates the value of explicit relational structure.  

- The paper is well‑structured, with helpful figures/tables and algorithm boxes that make the pipeline understandable to readers outside ILP.

### Weaknesses
- I believe paper's central claim of a "fully differentiable" pipeline relies on a non-differentiable. The predicate invention step, as detailed in Algorithm 2, relies on a non-differentiable, hard query to a frozen LLM. This is the step where the actual "conceptual" label (e.g., "triangle") is introduced, and it lies completely outside the end-to-end optimization loop. This architectural choice has implications -- the model does not learn concepts like "red" or "triangle" through backpropagation. Instead, it learns to form a cluster of entities that happen to be red triangles, and then it asks an external oracle (the LLM) for the appropriate human-readable label. This should be stated explicitly and transparently if my interpretation is correct as it reframes the contribution. 

- The reported runtimes (Table 4, Appendix) focus on γILP's own training steps but do not discuss the immense computational resources required by ViT, BERT, and especially LLM queries. Training and inferring with these foundation models comes with high cost in time, compute, and financial resources. There is no holistic accounting of these requirements, nor is the LLM inference cost (token count, latency, etc.) analyzed. 

- Some experiments appear to encode label cardinality via the number of clusters and rely on pre‑trained encoders with class‑discriminative structure, blurring the “no label leakage” claim.

- Failures on classical tasks (Fizz/Buzz) signal a combinatorial bottleneck in bottom‑up propositionalization with many variables/longer chains.  

- Several results are reported as “best of N runs,” with substantial variance on harder tasks. It will be helpful to see mean±std over fixed seeds and confidence intervals.

### Questions
- How does semantic accuracy change with 10–20% noisy cluster members? Please include a prompt‑sensitivity table across 3–4 prompt families and 2–3 LLMs if you have any.  

- What concrete search constraints or hierarchical strategies can unlock 4+ variable tasks (Fizz/Buzz)?  

- What is the end‑to‑end costing (encoder inference + LLM queries): tokens per predicate, latency, and monetary cost per dataset.  

- What do you think of learnable truth‑scoring for placeholder atoms instead of a strict logical AND over entity presence?

### Soundness
2

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
The paper proposes $\gamma$ILP, a framework designed to learn first-order logic rules from visual data without requiring explicit image labels. The authors claim this approach tackles the symbol grounding problem without "label leakage", enables "predicate invention" for undefined relations, and the proposed pipeline is end-to-end differentiable. The framework is evaluated on classical symbolic ILP datasets, relational image datasets using MNIST, and pure image datasets (Kandinsky patterns).

### Strengths
1. The framework uses LLMs to automatically assign semantic meaning to newly discovered relations, a step toward automated knowledge discovery.
2. The authors claim that $\gamma$ILP successfully integrates modern deep learning with a symbolic rule-learning network in a fully differentiable pipeline, enabling joint training.
3. The authors test their method across a diverse set of tasks, from classical symbolic datasets to complex visual reasoning with Kandinsky patterns, demonstrating its versatility.

### Weaknesses
1. The core claim of automated predicate invention depends entirely on pre-trained LLMs acting as an oracle. This reliance is unrigorous. Moreover, it essentially reduces to prompt engineering, which is highly sensitive to phrasing, model choice, and randomness.
2. The key generalizations from instance-specific outputs to abstract predicates appear to be manually derived by the authors, wich contradicts the claim of full automation. The LLM contributes only loosely interpreted semantics, introducing subjective human input. This undermines the claimed autonomy of the system.
3. Using an LLM for semantic interpretation introduces non-transparent.  Positioning such a unverifiable framework within explainable AI is conceptually inconsistent.
4. The experiments fail to convincingly validate novelty or effectiveness. Specifically, the baselines are poorly chosen, which compare relational reasoning tasks to non-relational models. The ablations are missing, leaving component contributions unverified.
5. The “differentiable substitution” mechanism is poorly explained and insufficiently contrasted with prior work.

### Questions
Does the claim of a “fully differentiable pipeline” hold, given that predicate invention depends on a non-differentiable LLM query?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes γILP, a fully differentiable inductive logic programming (ILP) framework that provides a fully differentiable pipeline from entity substitution to rule structure induction. This method consists of a deep clustering module serving as a generalization function, a latent knowledge base generator, a rule learning neural network with a novel differentiable substitution method, and LLMs acting as concept learners. Also,It is evaluated on classical symbolic ILP tasks, relational image datasets (e.g., MNIST-based relational reasoning), and pure image tasks (Kandinsky patterns), showing competitive performance and interpretative rule induction.

### Strengths
1.	No label leakage: the first model to learn rules from relational image datasets without label leakage which is a key limitation in prior ILP work.
2.	Broad empirical validation: across diverse tasks (symbolic, relational images, Kandinsky) shows robustness.
3.	Interpretative predicate invention: Predicate invention via LLMs is a practical and interpretative solution to a classic ILP challenge.

### Weaknesses
1.	LLM dependency: Predicate invention is outsourced to black-box LLMs; no analysis of prompt sensitivity or LLM choice.
2.	Task limits: The method struggles with tasks (e.g., Fizz/Buzz), “In γILP, the learned rule can only describe the entity zero, but fails to capture numbers divisible by three and five in the Fizz and Buzz datasets”, “the search space for γILP and DFORL becomes enormous without constraints such as the logical template used in ∂ILP”, suggesting limited scalability in complex relational reasoning without strong inductive biases or templates.

### Questions
1.	The paper emphasizes that γILP is “the first model to learn rules from relational image datasets without label leakage”. However, it relies on pre-trained ViT and BERT encoders that were trained on large labeled datasets. Given that these encoders may already embed semantic distinctions between colors and shapes—even without task-specific labels—could this constitute a form of implicit label leakage through the perceptual backbone?
2.  Recently, there is an increasing number of  studies addressing relational imaging learning such as, raven progressive matrices. I am not sure how this model can learned abstract rules for solving raven reasoning tasks.
3.	The paper mentions in Section 5.2 (p. 8, lines 456–458) that the learned rule for the two-pair Kandinsky task achieves recall = 1 but precision < 1 because it fails to account for another pair of entities with the same color and shape. Since γILP fixes the number of logic variables to the number of clusters (Section 4.3, p. 6), might this design choice restrict the model’s performance for tasks that require reasoning over multiple relational pairs simultaneously


References:
1. https://arxiv.org/abs/1903.02741

### Soundness
3

### Presentation
3

### Contribution
3
