# RL Grokking Recipe: How Does RL Unlock and Transfer New Algorithms in LLMs?

- Decision: Accept (Poster)
- Scores: 8, 8, 2, 4, 4

## Abstract
It remains an open question whether LLMs can acquire or generalize genuinely new reasoning strategies, beyond the sharpened skills encoded in their parameters during pre-training or post-training.
To attempt to answer this debate, we introduce DELTA — Distributional Evaluation of Learnability and Transferrability in Algorithmic Coding, a controlled benchmark of synthetic coding problem families designed to probe two fundamental aspects: learnability—can LLMs, through reinforcement learning (RL), solve problem families where pretrained models exhibit failure with large enough attempts (pass@K=0)?—and transferability—if learnability happens, can such skills transfer systematically to out-of-distribution (OOD) test sets? Unlike prior public coding datasets, DELTA isolates reasoning skills through templated problem generators and introduces fully OOD problem families that demand novel strategies rather than tool invocation or memorized patterns.
Our experiments reveal a striking grokking phase transition: after an extended period with near-zero reward, RL-trained models abruptly climb to near-perfect accuracy.
To enable learnability on previously unsolvable problem families, we explore key training ingredients such as staged warm-up with dense rewards, experience replay, curriculum training, and verification-in-the-loop.
Beyond learnability, we use DELTA to evaluate transferability or generalization along exploratory, compositional, and transformative axes, as well as cross-family transfer. Results show solid gains within families and for recomposed skills, but persistent weaknesses in transformative cases. DELTA thus offers a clean testbed for probing the limits of RL-driven reasoning and for understanding how models can move beyond existing priors to acquire new algorithmic skills.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors of this paper present a controlled, fully verifiable benchmark suite for program synthesis spanning Manufactoria DSL puzzles, BouncingSim bouncing ball simulation, and competitive programming tasks. The authors position this benchmark suite as a programming-domain counterpart of OMEGA (synthesizable math families), producing unit-test verifiable instances, yielding rich per-test reward signals and making external tool-shortcutting much harder.

With this benchmark suite, the authors investigate learnability, whether RL can train a model to solve problem families that the base model fails at, even after many sampled attempts (high pass@K), and generalization, whether the learned procedures transfer to OOD problems along the exploratory, compositional, and transformative axes.

The key empirical finding is that a two-phase reward schedule, which involves a warm-up stage with dense per-test rewards and then switching to a binary full-pass reward, induces a grokking-like transition. After a long, low-reward plateau, performance abruptly jumps to high accuracy and then stabilizes, including on families where the base model had a pass@K=0. Across tasks, gains transfer well on the exploratory and compositional splits, while performance on the transformative split remains weak. The authors interpret this as evidence that RL can unlock solution strategies beyond what is reachable by sampling the base model alone. The authors also caution that the warm-up is not a universal recipe and the benefits depend on model capacity, target family difficulty, and curriculum compatibility.

### Strengths
**Clear Learnability Results**: On a family where the base model has pass@K=0, the two-phase (dense to binary) schedule yields a grokking-like transition from failure to full pass. The grokking phase is a strong signal that the model has found an algorithmic procedure.

**Generalization Along Principled Axes**: Using OMEGA-style exploratory, compositional, and transformative splits, they show strong transfer on the first two and pinpoint limits on the third. This is a useful signal for where future work is needed.

**Comprehensiveness**: Broad experimental sweep across domains, families, defining thoughtful difficulty ladders, and training recipes, including ablations on replay, feedback-in-the-loop, and curriculum-based data, supporting the central claims.

**Dataset Contribution**: The authors supply an OOD problem space by introducing a bespoke Manufactoria DSL, a new syntax unlikely to appear in pretraining and pairing it with templated generators. The authors also organize families into a scalable difficulty ladder (Basic, Easy, Medium, Hard) based on aggregate model performance, so researchers can stage experiments by capability.

### Weaknesses
**Under Characterized Reward Switch Schedule:**
The two-staged dense -> binary schedule is central to the learnability claim, yet the paper does not ablate warm-up length or switch timing. The only concrete examples are an illustrative 100-step saturation during per-test warm-up, 450-step post-switch exploration, and a 100-step warm-up case in the appendix. The schedule hyperparameters otherwise appear fixed. This leaves open the minimal warm-up needed, whether earlier/later switches change sample efficiency or stability, and the robustness of this claim. More details strengthen the result and clarify robustness.

**Evidence for Curriculum Task Compatibility:**
The paper’s claim that “curriculum must be task-compatible” is compelling, but with only two examples (REGEX -> HAS, COMPR -> HAS), it is hard to generalize and see if the effect holds across other families or difficulty tiers, or whether there are cases of strong negative transfer. A small source x target compatibility/staged results matrix (warm-up on source S -> RL on target T) across more families/tiers, alongside this paper’s BouncingSim slice, would strengthen the paper and enable the research community to understand and quantify this paper’s claims of how curriculum design impacts learnability dynamics.

### Questions
Could the authors clarify how they decided to switch from dense to binary (fixed step vs plateau/success trigger), the warm-up length (was it only illustrative?), and whether they tried a gradual weight shift from dense to binary?

Minor Comments/Typos:
- Specicaly -> Specifically
- consruction -> construction
- BouncingSim vs BounceSim. Figure 1 uses BounceSim while the rest of the paper uses BouncingSim. Is this on purpose?
- Figure 1 claims four axes, including “Domain level”, but experiments report three (exploratory, compositional, and transformative). Is there a reason why there were no results from the Domain-level?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper examines the learnability and generalization of the RL algorithm in the domain of large language models. The paper presents two benchmarks, Manufactoria and BouncingSim, for testing novel algorithmic skills and compositional algorithmic skills. These tasks are proven to be unsolvable for LLMs with an adequate sample size. To test the learnability of the collected benchmark, the authors design a method to assign partial credit to a single test case and then transition to full credit. With this method, the RL algorithm (GRPO) exhibits the grokking phenomenon, and its performance grows rapidly within a few steps. Additionally, the experiment suggests that with the selected curriculum, the RL algorithm can learn more effectively. Bouncing2D is designed to test the generalization. The experiment results show that RL can exhibit good exploration and compositional generalization, but still yield poor transformative results.

### Strengths
1. Two different benchmarks to test the learnability and generalization of RL algorithms.
2. There are debates about the Grokking phenomenon; the work provides a testbed for testing Grokking, which is quite hard for the most performing language model, GPT-5. With a simple warm-up that includes unit-test credit assignment, the model exhibits a 'grokking' moment in the experiment.

### Weaknesses
1. The justification of OOD-ness is not trivial to me, see Questions 1 and 2 below. 
2. The paper discusses learnability and generalization on two different datasets. There is no strong correlation between these two datasets. See Questions 4 and 5 below.

### Questions
1. In section 2.1, these problems are not common programming problems or Turing-machine tasks. What do the problems belong to in terms of problem complexity? Does the problem belong to push-down automata or simply a finite-state machine? Or is the problem much harder than a Turing machine?
2. Following Q1, how much OOD-ness comes from the program syntax? If the program syntax is written in Python, can modern LLMs solve these problem families? Does the Grokking phenomenon come from the syntax? Does the language model suddenly understand the syntax and solve the problem perfectly?
3. In section 3.2 and Figure 6, how does the result `its convergence speed is still slower than the baseline GRPO algorithm, likely because the reused traces are off-policy` come from? Figure 6 shows Experience Replay and baseline converge to a common performance.
4. Although Bouncing2D is designed for testing generalization, what is the learnability of different difficult tiers?
5. What kinds of generalization ability are needed to learn across different families in Manufactoria? Or what are the atomic algorithmic skills needed between families to make the generalization impossible?

### Soundness
3

### Presentation
3

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
This paper introduces a benchmark and process for controllable generation of synthetic questions in those categories. The three benchmarks include augmented programming questions seeded from LiveCodeBench-Pro, 2D elastic collision simulation, and the puzzle game Manufactoria. The benchmark is designed to be able to generate questions with interpretable distribution shifts with the categories, for example smaller or larger polygons in the elastic collision simulator or difficulty in the coding questions (as measured by LLM success rate). The paper then runs various RL experiments on Manufactura and BouncingSIM. They show advantages to record-replay and switching from denser proxy rewards (number of unit tests passed) to the target reward mid training can both improve RL performance. Multiple different dynamics in the BouncingSIM are explored with separate training on subsets and then testing on different subsets or combinations of dynamics. They show very little transfer between different classes, but going from two classes of dynamics to their composition preserves much of the original accuracy.

### Strengths
- The paper introduces two interesting new benchmarks with can be decomposed into subsets of their mechanics to generate different types of problems
- There are some nice demonstrations of some techniques for RL
- There is a pretty clear difference shown between the compositional versus different classes in BouncingSIM

### Weaknesses
- I do not see a coherent story between the different tests and experiments run and I think most of the experiments do not support the goals laid out in the introduction
- I was frequently confused about what benchmarks were being described and what experiments were run
- I often could not tell whether a figure was representational or actually data
- Generally I found this paper difficult to read.

### Questions
- Figure 4 has overlapping text on the x axis.
- groking is a technical term which involves training accuracy saturating with little increase in test accuracy initially and (much) later an increase in test accuracy with little to no train accuracy increase. Is that the intended use here?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper considers the capabilities of RL-finetuned LLMS, specifically, whether they are able to learn to solve previously unseen problems, and how well do they generalize to harder versions of seen problems. To this end, it makes three main contributions:

First, the authors present Delta-Code: A controlled benchmark of synthetic coding problems, designed to evaluate: 
1) the “learnability” of how to solve problems that have not been encountered during an LLM's pre-training. To this end, Delta-Code incorporates problems related to solving puzzles of a classic Flash game (Manufactoria).
 2) the generalization to harder instances of the problems encountered during training. To enable this evaluation, Delta-Code contains coding tasks related to 2d bouncing object simulations.

Second, the authors utilize Delta-Code to evaluate whether fine-tuning an LLM using RL can allow said LLM to solve coding problems that it has not seen before and which require “new strategies”. They show that coding problems allow for richer supervision (by using the ratio of unit test passed by the proposed solution, instead of a binary reward) and that this supervision eventually enables the LLM to learn to solve the novel coding tasks. It is also noted that this approach can also fail for very challenging problems.

Third, the authors use Delta-Code to evaluate the generalizability of an RL-finetuned LLM. The use three generalizability axes: explorative, compositional and transformative, and conclude that these models fail to generalize when the test coding problems require solutions which are significantly different than the ones encountered at training time.

### Strengths
Delta-Code introduces a new family of coding problems that have not been encountered by current LLMs, making it a good tool for fair evaluation of their capabilities. Moreover, the family of 2d simulation coding problems could be useful for more reliable evaluation of a model’s problem-solving-generalization capabilities. Together, these can facilitate further research in this direction.

According to the paper, there is currently a debate on whether using RL to finetune an LLM allows for novel problem solving strategies to be learned. If so, then this paper represents a good step towards answering this question.

### Weaknesses
Readability: Many of the terminology used in the paper is not well defined, but its meaning is only explained intuitively. For example, learnability; transferability; explorative, compositional and transformative axes of generalization, reasoning strategies; skill family. Perhaps these are not well-defined terms, but with only an intuitive definition, it’s hard to judge how well the proposed benchmark evaluates them. In addition, RL training of LLMs is a big part of this paper, but the setting is never explicitly stated, with some details becoming available only at the end of page 5. I think the paper would greatly benefit from a “Background section” which clearly defines the background knowledge needed to better understand it.

It is possible that I am wrong, but the experimental results in sections 3 and 4 do not seem surprising, so they appear to be limited in their significance.

### Questions
Why were competition coding problems added to the benchmark?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies on a family of synthetic coding problems whether RL can learn tasks that were previously unsolvable and generalize to new unseen tasks. The primary contribution lies in controlling which distributions RL training and testing happen on. The takeaway from the learnability experiment is that using dense rewards for coding problems (fraction of passed tests) during a warmup-phase followed by the standard binary reward can turn some of the previously unsolvable tasks (i.e., 0 pass@128 prior to RL) into solvable ones. The takeaway for the generalization experiments is that RL enables both exploratory and compositional generalization, but fails at the transformative ones.

### Strengths
- the paper studies highly relevant and important questions at the intersection of RL & LLMs: what are the limitations of RL for learning new skills and generalizing to new domains
- the obtained results are interesting and put into the context of the broader literature that has previously discussed some these points

### Weaknesses
- the connection to grokking is unclear and confusing: in grokking, the model overfits the train set initially with a sudden increase in the test performance. the learnability experiments have close to 0% pass rate on the training set at the beginning so there's no train-test performance gap. my interpretation of the results is that the initial phase of close to 0% pass rate reflects a sparsity of non-zero gradients, and the training run first needs to accumulate sufficient informative/positive samples to reinforce.
- the paper claims that dense rewards can generally enable learning on previously unlearnable problem distributions, but the empirical evidence for this is limited to synthetic toy tasks. the explanation for this seems plausible, but it's unclear wether using dense rewards is necessary or significant in more realistic use cases, where the dataset consists of a mixture of problem types and difficulty levels. in realistic cases, generalization between different problems might be sufficient to enable learning previously unsolvable tasks and adding dense rewards might not have any effect. adding more experimental evidence to support the claims would greatly benefit the paper

### Questions
how was the code competition dataset used? generally it's not obvious to me which tasks were trained on and which were tested on in some of the experimental results.

### Soundness
2

### Presentation
2

### Contribution
3
