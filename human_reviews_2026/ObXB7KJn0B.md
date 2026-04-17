# The Serial Scaling Hypothesis

- Decision: Accept (Poster)
- Scores: 8, 4, 8

## Abstract
While machine learning has advanced through massive parallelization, we identify a critical blind spot: some problems are fundamentally sequential. These "inherently serial" problems—from mathematical reasoning to physical simulations to sequential decision-making—require sequentially dependent computational steps that cannot be efficiently parallelized. We formalize this distinction in complexity theory, and demonstrate that current parallel-centric architectures face fundamental limitations on such tasks. Then, we show for first time that diffusion models despite their sequential nature are incapable of solving inherently serial problems. We argue that recognizing the serial nature of computation holds profound implications on machine learning, model design, and hardware development.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper examines the limitations of parallel scaling in modern machine learning algorithms, with a particular focus on diffusion models. The authors advocate the necessity of serial computation to overcome these limits, through the titular 'serial scaling hypothesis'. Through various examples they argue that many ML tasks require moving out of the $\textsf{TC}^0$ complexity class. The paper concludes with a theorem arguing that diffusion models with $\textsf{TC}^0$ backbones remain within $\textsf{TC}^0$, even with infinite sampling steps.

### Strengths
This paper is a little outside my main area of expertise. I suspect I was assigned to review this due to my experience with diffusion models, so I will focus most of my comments around that area. I have made an effort to understand other parts but it is very possible that there are aspects of the paper that I did not absorb. Overall, I noticed the following strengths:

1. The paper is well written and the authors make a compelling case in favor of serial compute capabilities in the next epoch of ML models.

2. They clearly state the limitations and assumptions behind their statements, and clarify several misconceptions that arise when talking about serial vs. parallel compute in ML. In particular they address how an architecture that is parallel, when paired with an appropriate inference procedure, can capture inherently serial computations (Sec. 2.3). I found this very helpful when trying to understand their theorem on diffusion models. I had in mind a video diffusion model which is rigged to predict the $N+1$-th frame given the first $N$ frames. Such a model can, in principle, be configured in a sliding window manner to auto regressively continue the video. If the video shows a many-body physics problem (an example from the paper) then the whole setup solves a serial problem. If I understood correctly, theorem 4.1 in the paper applies to only one of these steps, where a single frame is predicted. The latter is independent of $N$.

### Weaknesses
1. The proof of theorem 4.1 assumes that scores are known perfectly, which is never the case in a practical diffusion model. While the authors acknowledge this limitation in the comment at the end of App. F.2, and around line 485, they leave the practical case for future work. But this does undercut their contribution somewhat, since it is highlighted as the 'main theorem' in the paper, and since the limitations of approximate scores are central to all realistic diffusion models. Is there an opportunity here to add some experiments to bolster the claim that a generalized version of the main theorem applies if $f_\theta$ is good enough? At the very least, when speaking about the main theorem in the main text, it could be stated that it applies in an idealized limit.

2. The proof also relies on non-uniform circuit complexity, which is strictly stronger than the usual uniform classes most ML models implicitly correspond to. The former allows for a magic constant to be stored in each circuit. Since these advice strings need not be efficiently derivable or implementable in practice, it is not clear to what extent this result constrains the computational behavior of real diffusion models. A brief discussion clarifying the practical implications of this non-uniformity assumption would strengthen the claims.

3. The paper initially suggests diffusion is sequential, then proves that its step-structure provides no serial scaling. I think this is conflating serial in auxiliary diffusion time (the iterative denoising schedule) with serial in data-dependent logical dependencies (the kind of seriality complexity theory cares about). Although the authors eventually clarify that they are talking about the latter, it felt like they were conflating the two to setup the premise.

Overall, the conceptual contribution of the paper feels timely, and with some clarification around Theorem 4.1, I think the paper could be strengthened further.

### Questions
Small typo around line 474: 'which', not 'whiche'.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper argues that some problems are inherently serial and cannot be efficiently parallelized, identifying a conceptual blind spot in current scaling approaches in machine learning. The authors formalize this distinction through computational complexity theory and apply it to modern ML architectures, contending that the diffusion models cannot effectively handle inherently serial tasks. They support this argument by linking theoretical limits on parallelism with empirical patterns observed in reasoning, simulation, and decision-making tasks.

### Strengths
1) Raises a meaningful point: not all tasks yield to parallel scaling.
2) Connects established computational theory to machine-learning practice.
3) The theoretical framing is clear and potentially useful.
4) Adds to the discussion on the limits of diffusion and transformer architectures.
5) Writing and diagrams were very clear.

### Weaknesses
1) Novelty is limited; the serial/parallel distinction is long-established in complexity theory.
2) Lacks new experiments or empirical demonstrations.
3) The implications for model or hardware design remain abstract.
4) The critique of diffusion models is not representative of how text diffusion models are commonly trained today.

### Questions
1) Can you extend your theory to diffusion models that are trained with auto-regressive objectives or structurally autoregressive generation structures, such as block diffusion? This is a very important point to address for the serial scaling hypothesis to have practical implications for diffusion models in architectural design. 
2) Comment on the translation of the work to papers like https://github.com/DreamLM/Dream-Coder
3) What concrete architectural or hardware implications follow from your theory?

I enjoyed the paper as a whole, but I can't recommend accepting unless weakness (4) and question (1) are addressed. This is my main issue with the work.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper reveals that modern machine learning overlooks problems that are fundamentally sequential and cannot be parallelized. It formalizes this distinction in complexity theory, showing that parallel architectures face theoretical limitations on such tasks. The study further shows that diffusion models also fail on inherently serial problems.

### Strengths
- The writing is concise and sharp and the study focuses on a precise and underexplored question: the boundary between parallel and inherently serial computation.
- Examples from reasoning, physics, and decision-making vividly illustrate what “inherently serial” means. Theoretical formalization combined with diffusion model evidence creates a rigorous and convincing argument.

### Weaknesses
- The proposed hypothesis trys to introduce computational complexity of serial dependency into model performance evaluation. However, the many-body example of serial problem is a prediction task on chaotic systems, where, even if we have ground truth labels and prior knowledge to train a model, the generalization performance is doubtfully discussed by a general hypothesis and experimental supports. This stems from the nature of the problem itself. It seems not good to use a many-body case here.
- Further numerical experiments with claim-relate analyses can validate the relevant conclusions and explore future research directions, such as how to verify the proposed hypotheses. This will make the factual existence clearer to the audience.

### Questions
In reasoning QA tasks, could more ablation about CoT or not be provided? The claims here try to specialize the type of QA as inherently serial, but it's hard to say that the reasoning and knowledge are decoupled and the models are actually reasoning without ablation.

### Soundness
3

### Presentation
4

### Contribution
3
