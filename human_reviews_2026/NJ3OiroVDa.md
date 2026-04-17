# Entropy-Reservoir Bregman Projection: An Information-Geometric Unification of Model Collapse

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Self-referential learning---training a model on data it generated itself---promises
		boundless scalability but chronically suffers from \emph{model collapse}: language
		models degenerate into repetitive text, GANs drop modes, and reinforcement-learning
		policies over-exploit. Although practitioners employ ad~hoc fixes such as real-data
		mixing, entropy bonuses, knowledge distillation, or retrieval-augmented generation,
		a single principle that explains both the failure mode and the success of these
		fixes has remained elusive.
		
		We present \textbf{Entropy-Reservoir Bregman Projection} (ERBP), an
		information-geometric framework that unifies these phenomena. We model the closed
		loop as a stochastic Bregman projection sequence in distribution space. Without
		external coupling, finite-sample noise forces the system to project onto an
		ever-shrinking empirical support, causing exponential entropy decay and eventual
		collapse. Introducing an \emph{Entropy Reservoir}---a high-entropy distribution
		mixed into each projection---injects a controllable entropy flux that provably
		stabilises the dynamics.
		
		Our theory yields (i) a necessary condition for collapse, (ii) a sufficient
		condition that guarantees a non-trivial entropy floor, and (iii) closed-form rates
		that depend only on sample size and the strong-convexity/Lipschitz constants of
		the Bregman generator. Experiments on large-language-model self-training, Soft
		Actor-Critic in reinforcement learning, and GAN optimisation validate our
		predictions and show that disparate stabilisation heuristics correspond to
		specific reservoir choices and coupling coefficients. ERBP thus transforms a
		collection of folk remedies into a single, quantitative design rule: monitor and
		budget your entropy flux.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Entropy-Reservoir Bregman Projection (ERBP) that derive the necessary condition for collapse, and a sufficient condition for a non-trivial entropy floor.

### Strengths
- This paper analyze the dynamics of self-referential learning systems with Entropy-Reservoir Bregman Projection framework
- Rigorous proofs and empirical experiments are provided

### Weaknesses
- Some notations appear before being properly defined, such as $P_{res,t}$ and $\lambda_t$ in lines 64–65. Providing brief explanations when first introduced would improve readability.
- The presentation of Sections 4.2 and 4.3 could be strengthened by adding more narrative around the proof logic and the connections between theorems, rather than only listing results. This would help readers follow the reasoning flow more clearly.
- The experimental section is relatively weak compared to the theoretical part. It would be beneficial to include additional comparisons with related methods mentioned in the related work to better demonstrate the advantages of the proposed approach.

### Questions
- The abstract should be presented as a single coherent paragraph rather than being split into multiple short ones. Merging them would improve readability and make the summary flow more naturally.

### Soundness
2

### Presentation
1

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
The paper proposes an information-geometric framework, called Entropy-Reservoir Bregman Projection (ERBP), to unify different phenomena of model collapse across LLMs, GANs, and RL agents. The authors model self-referential training loops as stochastic Bregman projection sequences in distribution space, where entropy tends to shrink without external coupling. They introduce the concept of an Entropy Reservoir to counteract entropy decay.

### Strengths
1. The novelty is ok. Recasting self-training dynamics as Bregman projection processes is novel and potentially unifying for understanding entropy decay across domains.

2. This paper is clearly organized, with some tables summarizing conceptual mappings.

### Weaknesses
1. Mathematical inconsistency / over-claim in theoretical findings, e.g., thm 1.

2. Lack of proof detail. For example, in the proof of thm 1,  what does ``martingale convergence plus the support argument of the main text finishes the proof.'' mean?

### Questions
1. In Theorem 1, how can the authors claim that full collapse to a single mode is inevitable when inequality (5) only upper-bounds the expected entropy by $C_F(m) + \frac{L_F \kappa}{\alpha}$ (which is strictly positive, e.g. $\log m > 0$ for the Shannon case)? What additional assumptions or steps would be required to make this conclusion mathematically valid?

2. In experimental evaluation,  what exact $\lambda$ value and reservoir sampling process were used? How was the ``entropy proxy'' (unique n-gram count) computed and normalized?

3. Does the framework hold if the model manifold M is highly non-convex and projections are approximate (large $\epsilon_{\max}$)? How sensitive are the results to $\epsilon_{\max}$ assumptions?

### Soundness
2

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
This paper proposes Entropy-Reservoir Bregman Projection (ERBP) as a unified information-geometric lens on ``model collapse'' in self-referential learning loops (LLMs trained from synthesized data, GANs, RL). They model each round as a stochastic Bregman projection onto a target formed by mixing the empirical self-samples with a high-entropy reservoir using coefficient $\lambda$. Without the reservoir ($\lambda$=0), generalized entropy contracts toward a small-support limit, but any sustained coupling guarantees a non-trivial entropy floor, with rates controlled by sample size and the generator's strong-convexity/Lipschitz constants. A small simulation with a frozen LLM shows entropy decay with $\lambda$=0 and stability with $\lambda$>0.

### Strengths
1) The paper provides a unifying perspective that cleanly ties together disparate "folk remedies" via the $\lambda$-coupled reservoir.
2) The proposed method provides simple, quantitative conditions that are easy to reason about and potentially monitor during training.
3) The results are demonstrated with a breadth across various divergences beyond just KL.

### Weaknesses
1) The use of terminology is a little bit confusing. As a researcher from the generative models community, the term that I'm more familiar with is "mode collapse" instead of "model collapse". I originally thought the authors wanted to propose a new definition that describes a different class of model failure case, but according to the paper it seems like the authors are just describing "mode collapse". Please correct me if I'm wrong.

2) While the proposed framework can be very promising, and in fact the case of RL and GAN validations may be covered too, the authors choose to leave them as planned future work and only focus on single frozen-LLM loop, which is a little bit disappointing. I recommend to at least show some rather simple cases in RL, since this can take less efforts than in GANs and will make the paper more impactful.

3) The projection-as-dynamics viewpoint is interesting, but parts echo standard entropic regularization intuitions; the delta over prior collapse analyses (e.g., recursion-curse) could be sharpened in positioning.

### Questions
1) How do you propose estimating $\epsilon_{max}$ during scaled-up settings?

2) How specifially does algorithms like top-p sampling impact the distribution under your framework? Could you extend your current analysis with a given step probability cap $p$?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces the "Entropy-Reservoir Bregman Projection" (ERBP) framework, a novel information-geometric model to unify the phenomenon of model collapse in self-referential learning. The authors model this process as a stochastic Bregman projection sequence, arguing that entropy-decay and collapse are inevitable unless coupled with a high-entropy "Entropy Reservoir". This concept is used to provide a unified explanation for various stabilization techniques, from real-data mixing to Retrieval-Augmented Generation (RAG).

### Strengths
The paper has a few strengths:
1. The core idea of modeling self-referential learning as a Bregman projection dynamical system is elegant and provides a powerful new language for analyzing these systems.
2. The "Entropy Reservoir" concept is insightful, successfully connecting disparate, seemingly ad hoc techniques (like data mixing, RLHF, and label smoothing) under a single, coherent mathematical principle.
3. The paper provides theoretical proofs for its claims, formalizing the conditions for both entropy collapse (Theorem 1) and stability (Theorem 2).

### Weaknesses
The paper's primary, and critical, weakness is a failure to substantiate its broad claims with empirical evidence. The experimental section is critically incomplete:

1. The abstract explicitly claims validation across large-language-model self-training, Soft Actor-Critic in reinforcement learning, and GAN optimisation. However, Section 6 directly contradicts this, stating that the work on "LLM fine-tuning and reinforcement learning" is "planned future work". The GAN experiment is never mentioned again. This misrepresentation of the work's completion status is a major flaw.
2. The only experiment provided (Section 6.1) is insufficient as a proof of concept. It uses a frozen distilgpt2 model and shows that feeding its own output back as a prompt (with $\lambda=0$) leads to repetition. This demonstrates context collapse in an autoregressive loop, but it does not test the paper's core theoretical claim, which is about model collapse resulting from self-referential training (i.e., an optimization and projection step that updates the model's parameters or state).

While the theoretical framework presented is promising and insightful, the paper is incomplete. The empirical validation promised in the abstract is essential for supporting the paper's unification claims, but it is explicitly admitted to be "planned future work". The single experiment that is present does not adequately test the central theory of collapse during training. Therefore, the paper should be rejected in its current form.

### Questions
Please see the Weaknesses

### Soundness
1

### Presentation
1

### Contribution
1
