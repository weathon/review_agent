# Learning Reactive Synthesis from Model Checking Feedback

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Deep learning applications to formal verification typically fall into one of two categories: employing reinforcement learning that suffers from slow convergence, or supervised learning that suffers from limited exploration. For reactive synthesis, the problem of automatically constructing a system that satisfies a formal specification, existing approaches fall into the latter category. In this paper, we propose a hybrid approach that only initializes the model with supervised learning and then continues training by reinforcing formally verified predictions. We show that by training the model to synthesize correct solutions rather than fixating on the supervised data, performance substantially improves. We can further utilize our approach to optimize for size without any performance degradation. Finally, we show that we can iteratively reinforce on open problems that synthesis tools are unable to solve. Our approach is demonstrated for both deep neural networks trained from scratch and pre-trained models fine-tuned on reactive synthesis, establishing new state-of-the-art results for learning reactive synthesis.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposed a new reinforcement learning framework of synthesizing hardware circuits based on the feedback from model checking results.
The experiments are based on open datasets and the results are outperform supervised learning baselines.

Pros:
1. The integration of model checking results and circuit synthesis is interesting.

Cons:
1. Using feedback from formal methods for learning is not novel, the novelty of the method is limited.
2. The experiment results are limited and not convincing.

### Strengths
1. I think it is interesting to apply feedbacks from model checking to circuit synthesis, and this integration is quite novel, at least I have never seen it before.
2. Overall the paper is well organized and it is not hard to follow.
3. The literature review is comprehensive.

### Weaknesses
1. My major concern of this work is the novelty of using model checking/formal verification feedback for learning, there are many of these works since 2023 and I'm not sure what novel or unique challenges does the circuit synthesis problem introduce. In fact, I have a feeling that this problem is well formulated and easy to solve compare to other problems. Maybe the authors can better elaborate the challenges and why it can be solved by this way.
2. My second concern is the experiments, seems like codeT5 outperforms the HT method, then why we want to have it?
3. I think the authors should clarify how the supervised part work from the problem level, currently, this part is missing in the paper.

### Questions
I have a basic question for choosing LTL, why not other logics?
For example, signal temporal logic (STL) can provide dense and continuous feedbacks from the specification satisfaction, why you choose LTL?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes an approach for synthesizing circuits from linear temporal logic (LTL) specifications using machine learning. The method builds on prior work by integrating model checker feedback and adding a search component for circuit size optimization. The approach is evaluated on several datasets.

### Strengths
The paper improves the performance of neural circuit-synthesis approaches by integrating model-checker feedback.

### Weaknesses
- **W1.** (Minor) The second paragraph of the introduction needs citations.
- **W2.** In my view, the contributions are not very strong. The paper proposes an arguably trivial combination of existing neural circuit-synthesis, feedback-based fine-tuning, and expert-iteration methods.
- **W3.** The differences between the proposed approach and prior approaches are not discussed in the related work section.
- **W4.** Although it is simpler than reactive synthesis, LTL model checking is also computationally hard. Thus, the proposed approach may introduce additional computational burden compared with existing methods, which is neither discussed nor evaluated.
- **W5.** (Minor) The algorithm fragment at the bottom of page 4 is somewhat confusing.
- **W6.** The results for SYNTCOMP in Table 1 are not very impressive.

### Questions
What are your thoughts on the weaknesses mentioned above?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the limitations of existing deep learning approaches to reactive synthesis—where supervised learning is confined to imitating synthesis tools and reinforcement learning has slow convergence. It proposes a hybrid method that initializes models via supervised learning, then refines them using model checking feedback to prioritize correct circuit synthesis over tool imitation.

Reactive synthesis, which constructs systems satisfying linear temporal logic specifications (critical for hardware design), is computationally hard (2EXPTIME-complete), leading traditional tools to timeout even for small specs. The paper’s hybrid framework first trains an initial model ($M_0$) on 200,000 Strix-generated specification-circuit pairs (supervised phase). In the second phase, it verifies the model’s predicted circuits ($\hat{C}$) with nuXmv: if $\hat{C}$ meets the spec, it reinforces the model with $(\varphi, \hat{C})$; if not, it falls back to the dataset’s correct circuit ($C$).

Three core variants extend the framework: 1) "Reinforcing Learned Semantics" boosts generalization by leveraging correct non-dataset predictions; 2) "Expert Iteration" uses beam search (top-k predictions) to improve performance and minimize circuit size with 54% smaller than Strix on average; 3) "Iterating on Open Problems" samples unsolvable Timeouts dataset to exceed tool capabilities.

Experiments on hierarchical transformers and fine-tuned CodeT5 show state-of-the-art results: CodeT5 with expert iteration hits 89.3%  on Testset and 51.9%  on Timeouts. The method advances reactive synthesis by combining efficiency, correctness, and scalability beyond traditional tools.

### Strengths
1.  The paper addresses the limitations of pure supervised or reinforcement learning in reactive synthesis by proposing a two-stage hybrid approach. It initializes models via supervised learning (imitating synthesis tool Strix) but shifts to reinforcing formally verified predictions using model checkers (nuXmv). This refocuses the training objective from "tool imitation" to "generating correct circuits," enabling better generalization.

2. Extending the core framework with expert iteration  delivers both performance and efficiency improvements. With beam size 4, CodeT5 achieves an 89.3%  on the Testset  and reduces circuit size by 54% on average compared to Strix—surpassing the 46% reduction from fine-tuning alone. Critically, optimizing for smaller circuits does not harm correctness, as semantic accuracy remains high even when syntactic alignment with training data drops to 0%.

3. The framework uniquely enables iterative improvement on "open problems" (specs Strix times out on). By sampling these specs during training (with \(p_{timeout}=0.5\)), CodeT5’s performance on the Timeouts dataset reaches 51.9%—solving over half the specs traditional tools cannot.

### Weaknesses
1.  The framework depends heavily on Strix for dataset generation and nuXmv for model checking. It does not validate performance with other synthesis tools (e.g., SYNTCOMP competitors) or model checkers (e.g., SPIN), leaving uncertainty about whether results hold across different toolchains.

2. No comparison with other state-of-the-art approaches such as NeuroSnyt. The experimental evaluation cannot reflect the advantage of the proposed approach over these existing approaches and tools without any experimental evaluation. 

3.  While expert iteration boosts performance, the paper only notes “linear scaling” of model-checking calls with beam size. It lacks quantitative data on training time/memory costs for large beam sizes (e.g., >4) or complex specs, making it hard to assess feasibility for resource-constrained scenarios.

4.  Though the proposed method solves 51.9% of Timeouts specs, the paper does not analyze if these solved “open problems” share specific patterns. It also fails to test if the method generalizes to open problems from non-Strix-generated datasets, weakening claims of transcending tool limits.

### Questions
1. The framework relies on Strix for dataset generation and nuXmv for model checking. Have you tested whether the performance gains  hold when using other synthesis tools (e.g., SYNTCOMP competitors like ltlsynt) or model checkers (e.g., SPIN)? If not, what justifies the exclusive dependence on these two tools for generalizability?  

2. When optimizing circuit size via expert iteration, syntactic accuracy drops to 0% while semantic accuracy improves—but the paper does not analyze why this decoupling occurs. Could you explain the key mechanisms that let the model diverge from the dataset’s syntactic patterns yet retain or enhance correctness, and whether this decoupling risks unexpected errors in complex, real-world specs?  

3. For open problem iteration, you sample Timeouts specs with \(p_{timeout}=0.5\) to boost performance. Have you explored higher \(p_{timeout}\) values e.g., 0.75 to see if open problem training can further improve pass rates, or does a ceiling exist due to the model’s reliance on initial supervised data from Strix?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Reactive synthesis is the problem of synthesizing finite-state models from temporal logic specifications. This paper explores if deep learning can be used to solve this problem. Compared to earlier attempts to use ML for reactive synthesis, the new ideas include use of a model checker to give feedback to update the model, use of top-k predictions for improving the quality of learnt solutions, and iterating on problems that model fails to solve. The methods are implemented and evaluated on benchmarks for synthesis competitions.

### Strengths
As authors explain, reactive synthesis has a long history of research. This is a computationally challenging problem, and thus there is a clear motivation to explore if deep learning can lead to better techniques. The 3 ideas to improve basic supervised learning are all natural, and likely not previously explored in this context.

### Weaknesses
The ideas proposed to improve supervised learning (e.g. using a verifier, model checker in this particular context, for RL feedback) are all standard from ML literature, so there is little conceptual novelty. Yet the paper will be a valuable contribution if the experimental results were strong, but that does not seem to be the case.

### Questions
1. Did you try using state-of-the-art LLMs (GPT-5 for instance with some prompting) to solve any of these problems
2. Is there more detailed analysis of computational effort needed to solve these benchmarks
3. Is there some qualitative analysis of some case study that indicates that now we can use reactive synthesis to solve a problem of practical interest

### Soundness
3

### Presentation
3

### Contribution
2
