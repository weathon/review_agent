# Model Diffusion for Certifiable Few-shot Transfer Learning

- Decision: Reject
- Scores: 4, 4, 8, 6

## Abstract
In contemporary deep learning, a prevalent and effective workflow for solving low-data problems is adapting powerful pre-trained foundation models (FMs) to new tasks via parameter-efficient fine-tuning (PEFT). However, while empirically effective, the resulting solutions lack generalisation guarantees to certify their accuracy - which may be required for ethical or legal reasons prior to deployment in high-importance applications. In this paper we develop a novel transfer learning approach that is designed to facilitate non-vacuous learning theoretic generalisation guarantees for downstream tasks, even in the low-shot regime. Specifically, we first use upstream tasks to train a {\em  distribution over PEFT parameters}. We then learn the downstream task by a {\em sample-and-evaluate} procedure -- sampling plausible PEFTs from the trained diffusion model and selecting the one with the highest likelihood on the downstream data. Crucially, this confines our model hypothesis to a {\em finite} set of PEFT samples. In contrast to the typical continuous hypothesis spaces of neural network weights, this facilitates tighter risk certificates. We instantiate our bound and show non-trivial generalization guarantees compared to existing learning approaches which lead to vacuous bounds in the low-shot regime.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This study try to use diffusion model to learn the distribution of various tasks under the brief that the any task can be identified by the model's parameters. Based on the learned diffusion model, the authors claimed that this distribution can reduced the complexity of hypothesis space from infinite to finite, thereby resulting in a tighter generalization bound, especially under the scenario of a few samples. The authors subsequently conducted abundant experiments to verified their idea.

### Strengths
1. The idea of using a diffusion model to learn the distribution of tasks and then applying it to transfer learning sounds like a interesting idea. Even though it requires a strong assumption that the task can be totally identified by model parameters with little or no ambiguity.
2. The performance of STEEL is powerful across various few-shot datasets compared to the other methods.

### Weaknesses
I have several concerns about STEEL
1. The authors use a diffusion model to learn the potential distribution of "tasks" and then sample a collection of candidate $\theta_i$ from it. However, this sampling process does not appear to leverage any information specific to the target task. As a result, STEEL essentially employs a diffusion-based method to reduce the hypothesis space, rather than directly imposing a meaningful prior constraint informed by the target task itself. While reducing the size of the hypothesis space can indeed tighten the generalization bound, it may also diminish the expressive power of the model. There is no evidence provided that the optimal parameter for the target task remains within the reduced hypothesis space, especially in the absence of any target-specific information. From this perspective, I am concerned that the motivation of this paper may be stereotype. In other words, if the authors focus solely on achieving a tighter generalization bound without considering the impact of hypothesis space richness, it becomes trivial to obtain a minimal upper bound simply by excessively restricting the hypothesis space—for example, to a linear or even a constant model. However, such an approach would also severely compromise the expressive capacity of the model. 

2. In the initial version, the author didn't compare the cost during both training process and testing process with other methods. I suppose it will enhance the solidness of this paper but doesn't influence the core impact of this study.

### Questions
Pls see Weaknesses

**Summary of Review**: As elaborated in the first item of Weaknesses, I do not think the authors' motivation is sufficiently reasonable, as the approximation ability of the hypothesis space is overlooked. While this omission is sometimes accepted in learning theory, where the focus is primarily on statistical aspects, it is not appropriate when discussing the size of the hypothesis space, given the strong connection between model performance and the richness of the hypothesis space. Nevertheless, considering that this study is not in the theory track and the authors provide improved practical results, I am inclined to give a score of 4.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies low-shot cross-task transfer learning. In the upstream phase, the method fits trainable modules to available source tasks and then trains a parameter diffusion model to generate models according to the task distribution. In the downstream phase, it follows a sample-then-evaluate paradigm. However, the architecture of the diffusion model and its training procedure are not clearly described. Experimental results show improved bounds and non-trivial generalization guarantees.

### Strengths
The paper provides some theoretical analysis, and the experiments indicate improved bounds and non-trivial generalization guarantees.

### Weaknesses
1. The title in the submitted PDF differs from the one on OpenReview, which raises concerns.
2. The main contribution appears to be learning a parameter diffusion model to generate PEFTs according to the task distribution. However, the architecture and training of the diffusion model are unclear. Section 3.3 discusses its role only at a high level, stating it is trained on PEFT parameters {θi}, but omits key architectural details and training objectives. The paper seems to assume readers are already familiar with parameter diffusion models, which may not be the case. While I am familiar with generalization theory and image/language diffusion models, I could not find sufficient background or specific descriptions of the parameter diffusion model used here.
3. In terms of task performance, the method does not consistently or significantly outperform baselines across Tables 1 and 2, even though it achieves better bounds and non-trivial generalization guarantees.
4. Efficiency concerns: The method requires evaluating thousands (sometimes tens of thousands) of candidate hypotheses per support set, which is computationally expensive and may be impractical outside research settings. Although Section 3.4 mentions hierarchical search and clustering, there is no detailed comparison between exhaustive and approximate search strategies.

### Questions
See weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes STEEL: learn a diffusion model over weight-space PEFT parameters $p(\theta)$ from a model-zoo of upstream adapters, and at test time sample a finite set $\Theta$ and evaluate–then–select the adapter with the lowest support loss. Casting few-shot adaptation as selection over a finite hypothesis set allows PAC/PAC-Bayes–style non-vacuous risk certificates, while keeping test accuracy broadly comparable to standard learners.

### Strengths
- A neat combination of weight-space generative modeling (DDPM) to form a finite hypothesis set and evaluate–then–select to keep the complexity term fixed while minimizing empirical risk—yielding non-vacuous certificates in few-shot regimes.
- Experimental breadth & rigor. Evaluations span multiple LaMP tasks and vision datasets under standard meta-learning protocols, reporting not only accuracy but also % non-vacuous, bound statistics, and gap.
- The paper visualizes how certification varies with $|\Theta|$ and demonstrates non-vacuous certificates from low-shot settings onward.
- The training of $p(\theta)$, sampling of $\Theta$, and the hierarchical search details are described clearly enough to reproduce.

### Weaknesses
(W1) The paper fixes LoRA-XS (~2.6K params) and CoOp (1,024-dim prompt) without varying LoRA rank or token count. Given both diffusion learnability and certification can depend on $\dim(\theta)$, a $\theta$-size sweep would strengthen the claims.

(W2) Fairness to model-zoo under matched search. Hierarchical search is an inference strategy, not unique to STEEL. A controlled comparison where model-zoo also uses the same k-means/medoid + top-15 pipeline would isolate the benefit of diffusion sampling vs search.

(W3) Choice of $|\Theta|$ vs. zoo size not probed. In vision, STEEL uses 20K samples while the zoo size is 10K. Results for $|\Theta|=10K$ (matched to the size of model-zoo) are not reported, leaving unclear whether gains come from diffusion coverage or simply larger candidate pools.

(W4) Missing reporting of the chosen cluster count $N$. For LaMP, $N$ is selected via minimum silhouette over a range, but the actual values (or distribution) and typical cluster sizes re-evaluated are not reported, making it hard to assess how much improvement stems from search granularity.

### Questions
Please provide responses to the reviews mentioned in the Weaknesses section.

### Soundness
3

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
This paper introduces a STEEL - a diffusion hypernetwork to produce PEFT weight candidates at test time in order to provide stronger / tighter generalization bounds. By sampling weights from the diffusion model, and selecting via sample-and-evaluate, risk certificates can be made tighter. On both LLMs and several few-shot vision tasks, the authors offer supporting practical evidence on a smaller scale, mostly matching normal PEFT performance while providing much better generalization bounds,

### Strengths
The paper is well written and straightforwward to follow; while STEEL itself is sensible - you utilize a diffusion model as a weight candidate generator for your test-time PEFT in order to provide tighter (and actual) generalization bounds. The combination of PEFT hypernetwork and generalization bounds is, to the best of my knowledge, novel - and sensible, with convincing results in 5.1 and 5.2 on both LLM and Vision-model adaptation.

### Weaknesses
I do think this paper provides a very interesting use of PEFT hypernetworks for generalization bounds, which to me comes with one major question / issue:

The whole approach hinges on L. 193: "We expect that Θ is rich enough to represent the true task distribution ptrue(T) faithfully, and the
adapted (“selected”) θ will generalize well on unseen samples from T", which is a very, very strong statement to make. By default, STEEL is likely much more limited when it comes to adapting to larger distribution shifts, as no weights are actually learned on support data (i.e. downstream data). This means that generalization bounds really only hold when the downstream data is much more i.i.d. with respect to the upstream data compared to normal PEFT, no? I.e. what does the performance look like as the downstream data moves progressively further away from the upstream data? At what point would be we a significant drop in performance of STEEL?
Or, from the opposite perspective - how well do these generalization bounds and comparable performances hold as the underlying mode is scaled up alongside the upstream data and the diffusion candidate generator?


Nitpick: Submission title does not match paper title

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
