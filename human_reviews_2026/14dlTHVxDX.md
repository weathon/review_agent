# Emergent Discrete Controller Modules for Symbolic Planning in Transformers

- Decision: Accept (Poster)
- Scores: 8, 2, 8, 6

## Abstract
Transformers struggle with tasks that require symbolic planning loops, variable updates, and conditional branching, especially under length extrapolation. We introduce discrete controller modules that insert a small set of program primitives (ASSIGN, ADD, COMPARE, BRANCH) into Transformer blocks via a Gumbel–Softmax selector over operations and a compact program state of registers, flags, and optional memory. We prove that the augmented model can simulate any bounded-step program by mapping each primitive step to one controller step, and we bound the deviation of relaxed execution from its discrete trace by $O(\tau+\kappa^{-1})$ (selection temperature $\tau$, comparison sharpness $\kappa$). Empirically, the controller-augmented Transformer achieves strong length generalization on algorithmic benchmarks (Sorting, Sum-of-List, BFS), improving longest-length accuracy by up to $20$–$40$ points over strong baselines, and yields consistent gains on symbolic QA (DROP) and program-synthesis-style tasks (RobustFill) with reduced compositionality drop-off. The learned execution is interpretable: operation traces align with ground truth, register roles are linearly decodable, and targeted knockouts cause localized accuracy losses. The approach adds only $\sim$5–7% FLOPs and can be applied sparsely (every $p$-th layer).

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper designs, controller-augmented transformer, using Gumbel-Softmax trick to integrate diccrete operations and differentiable learning. The approch in the paper is notable for its strong theoretical grounding, providing an expressivity guarantee for a class of bounded imperative programs. Empirical results confirm the effectiveness of controller-augmented transformers.

### Strengths
1. The combination of Gumbel-Softmax and transformers leverages both advantages of symbolic learning and transformers
2. The theortical proofs are elegant.
3. Empirical results shows considerable improvements compared to previous approaches.

### Weaknesses
I am unsure whether the paper have conducted enough comparsions between other approaches. CLRS-Algorithmic Transformer, the newest method in 4.2 BASELINES section, is developed in 2022, which maybe out of date. It hard to say whether the paper has really improved the margin of the state

### Questions
I know there are many approaches that combine statisical selction and differential learning. Why do you choose  to combine Gumbel-Softmax and transformers? Is it because it is easy to make theortical proofs for Gumbel-Softmax?

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
3

### Summary
The paper presents the idea of a discrete controller module. A discrete controller module adds to a transformer block the ability to learn 5 specific instructions, by using the Gumbel-softmax optimization trick to handle the discrete nature of the learning task.

An empirical evaluation is presented for 3 small benchmarks: a synthetic one, some string-to-program tasks inspired from RobustFill, and the DROP benchmark. In all of the presented results in Table 1, the method has a noticeable drop in accuracy as the input size n is scaled from 80 to 320. It is less pronounced than the baselines compared with, but it is a statistically significant drop. Table 2 and 3 reports that about upto 5% average gain in accuracy.

The added performance cost is about 5-7%.

### Strengths
- The proposed method and evaluation criteria is clearly described.

- The performance cost is modest, 3 to 10% depending on often the controller is inserted

- The traces are human interpretable, in the sense that one can inspect the opcodes printed out.

### Weaknesses
- The method is very specialized to the choice of 5 opcodes and the applications it can support. This severely limits the scientific value of the work.

- No justification is given for how the benchmarks are chosen. They appear to be chosen to fit the 5 opcodes and are rather small.

- We see a significant loss in accuracy, albeit lesser than other baselines, when n is scaled from 80 to 320. As such, the method doesn't scale well for the presented example applications, and will probably fair poorly as we increase n beyond what is reported. 

- It is difficult to interpret the alignment study results in Section 5.4. Taking the example of sorting, why are we checking trace alignment with a specific way of sorting, e.g., compare-and-swap? Doesn't the algorithm matter, eg. quicksort vs. bubble sort? 

Overall, one could perhaps see the scientific value better if the paper's method extends to a much larger set of opcodes. For example, the authors can consider taking the opcodes of a real processor (e.g. RISC-V, x86) and show if the method works generically for that. This would considerably broaden the applications for which the idea of discrete controllers can be used.

### Questions
What is the scientific contribution that extends to program reasoning, and is applicable to tasks that need more than those 5 opcodes?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes adding some programming primitives after every 2 
transformer blocks to give the neural architecture an ability to learn
fixed length programs.

The paper describes this extension in detail. Specifically, the "controller"
block that goes after a transformer block has 8 registers (each storing a 64-dim vector), 
memory (64x64) and some binary flags -- these define the state of the controller. 
The operation set consists of Assign, Add, Compare, Branch and Noop operators. 

If H_l = token representations entering controller block l
1. controller first reads a summary s_l = \phi( H_l ) 
2. s_l is used to draw an operator using Gumbel softmax using logits W_o s_l + b_o 
where o is one of the 5 operators.
3. each operator has a semantics that is used to update the registers, memory and
flags.  Assign and Add update the registers, compare updates the flags, branch 
updates the full state. The final updated state is a linear combination of the
updates from the different operators - weighted by the values from (2).
4. the output H_{l+1} of the controller block is H_l + \rho( change in state ) W_s

When this feeds into the next transformer block, we map the state-to-token 
and add to H_{l+1}, apply the transformer block on it, and feed that to the next
controller block or transformer block.

The evaluation compares the new Ctrl-Transformer with pure transformers and their
variants. The benchmarks include sorting, sum-of-list, graph traversal, along
with some program synthesis, math reasoning and symbolic question answering. 
Some of the algorithmic problems are parameterized -- so one would train on some
values of the parameters, say n = 10, 20, 40, 80 and then test on same or higher
values of the parameters. The main finding is that Ctrl-Transformer maintains
its performance for higher parameters, whereas baselines degrade in performance.
Even on other non-parametric tasks, there is a visible improvement.
The paper also has detailed ablations and error analysis to justify the choices.

### Strengths
Strengths:
1. The Ctrl-Transformer architecture is an interesting extension to transformers
2. The experiments are fairly conclusive in showing the value of Ctrl-transformers
3. There is extensive evaluation and discussion on how the various components are
contributing and how things are failing, showing that the investigation here is 
rigorous.

### Weaknesses
Weaknesses:
1. My main concern is around the writing, but I believe it can be potentially fixed 
by a careful pass on the paper.  However, I could not figure out a lot of the details
even after going back and forth between the different sections and the Appendix, and
that is the reason for my low confidence in the review. Detailed comments below.

Detailed Comments:
l87: Equation (3) defines FFN and Equation (4) uses SubLayer, which is undefined
l173: The last part there where \beta are used to combine u's is not clear. What 
are the u's ? Where is that R-tilde used?
l177: In equation (9) the value read from the registers, r, is the second argument to
Sem function, but where is it used in Equations (10)-(14)?
l191 mentions "sharing parameters across layers" -- how is reflected in the calculation
of the total number of parameters that is present in the Appendix.
l195: What is the function MHA( )? Where is it defined?
 - Is it true that the memory is only updated by branch? I don't see the memory getting
  explicitly updated in the semantics of any of the operators.

Theorem 1: The statement of the theorem lacks rigor - first the class of programs is only
described informally; moreover, the semantics of programs is unspecified. Looking ahead
to Appendix D, there appear to be plenty of assumptions mentioned there, if they are indeed
needed for the proof, then the statement of the Theorem should also include those.

l225-235 on "Error under finite temperatures" is also difficult to understand since it 
introduces terms like minimum opcode-margin that are undefined.

l307 says d_s = 128, but d_s on Line 1636 in the appendix is defined as N_r d_r + N_m d_m + N_f,
which is clearly larger than 128 for the particular choices of the parameters.

I found the parameter count calculation on Page 31 helpful. It would help further if you
could further clarify Params_ctrl calculation on Line 1638.

### Questions
I do not have a very specific question. Based on the detailed comments in the Weaknesses section above, if there is anything that can help clarify those points, then it will help me better understand the details of the paper.

### Soundness
3

### Presentation
3

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
This paper presents an augmented version of a Transformer that includes controller blocks. These blocks select and execute symbolic primitives. The program state and Transformer activations influence each other. The controller module is made differentiable via the Gumbel-Softmax trick. This Transformer variant outperforms other Transformer variant on length generalization over several synthetic tasks. The Transformer also outperforms other variants on DROP and some program synthetic tasks.

### Strengths
* The paper appears to be well executed. The authors should be applauded, as it seems like a considerable effort to construct such a model and get it to converge to reasonable solutions across the tasks studied.
* The empirical results demonstrate strong length generalization on synthetic tasks, as well as strong performance on more real-world tasks such as DROP.
* The paper provides some formal results related to the expressivity of the proposed controller module.

### Weaknesses
* The authors differentiate their approach from prior work on integrating programming primitives with neural models (e.g. neural GPUs, neural turing machines) by claiming that prior work has incurred "significant architectural complexity" or add "training brittleness". However, it's not clear that the proposed approach significantly mitigates these limitations.
* My understanding of prior work was that a main challenge of such methods is that it can remain difficult to learn complex programs from weak supervision. While discrete operations can technically be made differentiable using continuous relaxations (e.g. Gumbel-Softmax) this doesn't necessarily make such models trainable in practice. Indeed, the authors design various additional objectives and annealing schedules. Additionally, the synthetic tasks seem to rely on trace supervision to learn complex compositional programs. I understand the non-synthetic tasks such as DROP don't use trace supervision, but also don't seem to require learning very compositional programs, and it wasn't clear what is actually being learned by the neural controller for DROP despite the claims related to interpretability. I don't want to propose running more experiments, but curious to hear the authors response to this. Can this approach scale to learning complex programs from weak supervision? Or is the approach useful even without this ability?

Nits:

* "Emergent" seems like a strong claim in the title, given the strong architectural bias.
* Some of the exposition and notation could potentially be improved, e.g. `d_{ff}` is presumably FFN hidden dimension but not explicitly defined (line ~88) and the relation between `\mathcal{S}^{(l)}` and `s^{(l)}` was not immediately clear (line ~155) despite the potentially confusingly similar notation.
* It might be interesting to discuss recent work on RASP, Tracr, and ALTA that demonstrate that program primitives can be directly compiled to and represented by standard Transformers.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
2
