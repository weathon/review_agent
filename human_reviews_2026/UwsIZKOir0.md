# Provable Benefit of Curriculum in Transformer Tree-Reasoning Post-Training

- Avg Score: 2.80
- Decision: Reject
- Scores: 0, 4, 6, 2, 2

## Abstract
Recent curriculum techniques in the post-training stage of LLMs have been widely observed to outperform non-curriculum approaches in enhancing reasoning performance, yet a principled understanding of why and to what extent they work remains elusive. To address this gap, we develop a theoretical framework grounded in the intuition that progressively learning through manageable steps is more efficient than directly tackling a hard reasoning task, provided each stage stays within the model’s effective competence. Under mild complexity conditions linking consecutive curriculum stages, we show that curriculum post-training avoids the exponential complexity bottleneck.  
To substantiate this result, drawing insights from the Chain-of-Thoughts (CoTs) solving mathematical problems such as Countdown and parity, we model CoT generation as a states-conditioned autoregressive reasoning tree, define a uniform-branching base model to capture pretrained behavior, and formalize curriculum stages as either depth-increasing (longer reasoning chains) or hint-decreasing (shorter prefixes) subtasks. Our analysis shows that, under outcome-only reward signals, reinforcement learning finetuning achieve high accuracy with polynomial sample complexity, whereas direct learning suffers from an exponential bottleneck. We further establish analogous guarantees for test-time scaling, where curriculum-aware querying reduces both reward oracle calls and sampling cost from exponential to polynomial order.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
Proposes a framework to prove that curriculum training is provably good for post-training chain-of-thought generation.

### Strengths
Unable to identify or validate the strengths of the paper due to the opaque presentation.

### Weaknesses
The paper is poorly written all through.  The introduction states the problem in mathematical terms using symbols that are undefined. The notation section re-introduces the symbols but refers to the introduction for their significance.    Some specific examples below.

The problem statement and the assumptions in the introduction (lines 57-79) launch into formulaic equations without any supporting definitions of the quantities involved.   This makes it impossible to decipher the problem being addressed.

Preliminaries and notation (lines 125)  "For each prompt $x$, policies are conditional probability measures $\pi(. | x)$ on the output space."    

What does the subscript $k$ of $\pi_k$ mean in line 126?

Assuming the inputs $x$ and the outputs are embedding vectors of real numbers, $\pi$ is a probability density function.   In which case what does the derivative of one probability density function with respect to another (line 127) mean?  And the variable $o$ of line 127 is entirely undefined.

Lines 132-134:  The definition of pass-rate is circular in that it refers to the introduction section which in turn requires the notations to extract meaning.  

Theorem 1 of line 135 is meaningless in light of the opaque notation and definitions.  The important notions of a task $k$, and a curriculum of tasks $K$ are undefined.
What is assumed to be "absolute continuous"  in the theorem, line 137?

lines 143-146 of Theorem 1:  Difficult to understand what this means.
"Further assume a complexity–mismatch alignment ...up to harmless logarithmic factors in a confidence parameter"

Figure 2 caption and elsewhere. "secret index" is used but never defined.

### Questions
None at this time

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a theoretical analysis framework to explain the benefit of curriculum learning during LLM post-training. The framework treats CoT generation as a State-Conditioned Autoregressive Reasoning Tree (2S-ART).  The paper show step-wise curriculum post-training can reduce exponential depth dependence into polynomial order. The exponential-to-polynomial from curriculum learning explains why curriculum learning empirically outperforms non-curriculum approaches.

### Strengths
1. The paper is well-motivated by addressing a key open question of principled post-training explanation.
2. The proposed 2S-ART framework abstracts the curriculum learning process into a mathematically analyzable tree structure, providing a way to measure the complexity of reasoning.
3. The paper theoretically indicates directions for future research, namely that curriculum learning can reduce the difficulty for models to learn complex tasks.

### Weaknesses
1. The paper relies on an exponential complexity assumption, but the source of this exponential growth is unclear: is it due to decoding search (e.g., CoT tree expansion) or the intrinsic difficulty of reasoning tasks? Moreover, there is no empirical evidence that solving harder reasoning problems actually requires exponentially more reasoning steps.
2. The paper provides no empirical results to support the theory — even a small-scale experiment demonstrating that curriculum learning achieves a task with exponentially fewer training steps would greatly strengthen the claim. Without such evidence, the work remains largely theoretical and its practical significance to real LLM training remains unclear.

### Questions
Can the authors provide even small-scale experiments showing that curriculum learning reduces training complexity as predicted by the theory?

### Soundness
3

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
2

### Summary
This paper gives a learning‑theoretic account of why curriculum‑style post‑training can help LLM reasoning. It models chain‑of‑thought generation as a two‑states conditioned autoregressive reasoning tree (2S‑ART) and posits a uniform‑branching “base” model that spreads probability over legal next steps. Curriculum stages are formalized as (i) depth‑increasing (longer CoTs) or (ii) hint‑decreasing (shorter prefixes). Under outcome‑only (binary) rewards, the analysis shows curriculum avoids an exponential bottleneck, achieving polynomial sample complexity, while analogous guarantees hold for test‑time scaling by reducing both reward‑oracle calls and sampling cost from exponential to polynomial. Canonical tasks (Parity, Countdown) and a representation theorem connect the 2S‑ART abstraction to transformers.

### Strengths
2S‑ART captures stepwise CoT reasoning with legal action sets and state updates, making “prefix curriculum” and “hint curriculum” precise. The paper also establishes exponential‑to‑polynomial separations for both RL fine‑tuning under outcome‑only rewards and for test‑time scaling via curriculum‑aware querying.

### Weaknesses
1. Strong base‑model assumptions. The uniform‑branching coverage (“base model assigns comparable mass across legal children”) is convenient for proofs but unrealistic for modern LLMs whose next‑token distributions are highly skewed and prompt‑dependent. The main theorems hinge on this coverage/complexity alignment. 
2. Idealized task/trace structure. The framework assumes a single “correct” index path with a known legal‑set policy; many real problems have multiple near‑equivalent paths and context‑dependent constraints. The parity/countdown focus limits external validity.
3. Missing empirical calibration. The theory predicts polynomial scaling benefits; a small empirical study with real RLHF/RFT post‑training on diverse reasoning sets would strengthen the practical relevance.

### Questions
See weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a theoretical framework to prove that curriculum post-training can avoid exponential sample-complexity bottlenecks in both RL fine-tuning and test-time scaling.

### Strengths
- The notion of 2S-ART provides a unifying abstraction that captures a broad class of reasoning behaviors.
- Theoretically characterizing when and why curriculum is more effective in post-training is an interesting question. This paper presents a general framework as well as analyzes a concrete synthetic task of parity.

### Weaknesses
- The writing quality hinders comprehension. Below are several points that substantially obstruct my understanding:
  - In Theorem 1, the notation $\mathcal C(\pi_{k'}^\star | \pi_{k}^\star )$ seems undefined. It is unclear what training algorithm is used, and why $\|\frac{\pi^\star }{\pi_{\mathrm{ref}}}\|_{\infty}$  is an appropriate proxy for difficulty.
  - For Theorem 3, I could not locate a full proof in the appendix. It would help to provide a clear pointer and a proof sketch in the main text.

- The proof of Theorem 1 becomes straightforward once the assumptions (1,2,3) are imposed. Thus, it is crucial to justify that these assumptions hold for the reasoning tasks studied, but the paper does not provide a clear validation. It is also unclear how the concrete examples of parity satisfy the assumptions and how they connect back to the general theorem.
- The theoretical novelty appears limited, and the results may not have substantial interest to the broader community.
  - For theorem 1, the core idea that $\mathcal C(\pi_{k'}^\star | \pi_{k}^\star )$  surves as a proxy of difficulty is studied in previous works [1]. 
  - The benefit of curriculum used in Theorem 3 looks similar to existing analyses of CoT training [2, 3]. It would be helpful if the authors could clarify on the difference in setup and proof techniques compared to these work.

### Questions
- Is the $\Omega(d^{ k^\star+1})$ in Theorem 3.1 derived via an statistical query-style argument? If so, how does this lower bound relate to the general complexity measure in Theorem 1?
- Why is $\|\frac{\pi^\star }{\pi_{\mathrm{ref}}}\|_{\infty}$ an appropriate proxy for difficulty? (If I understand correctly, [1] only justified it under linear softmax model parameterization.) It would be helpful if the authors can provide a formal statement establishing its relationship to statistical or computational complexity.

Please also refer to the weaknesses. I am open to adjusting my score based on the authors’ responses and the discussion with other reviewers.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the optimization-based learnability of transformer in CoT reasoning tasks. The writing is difficult, with key notations missing or difficult to find, which obfuscated the main results and limits the value of the work. 

The paper presents four main theorems. 

Theorem 1 aims to show that curriculum improves divergence blow-up under certain bounded-divergence conditions, but the correctness is not checkable because the core definition of C(pi| pi’) is not given. (If I have to guess, it is something related to $||\pi_{k+1}/\pi_{k}||_{L^\infty}$ but then some of the assumed constants $C_{k,k'}$ might be in conflicts and need further clarification.)

Theorem 2 states that a random policy on the 2S-ART tasks defined in Definition 1 can be simulated by a transformer, if the FFN can simulate certain target operation. However, unless such policy can obtained by random initialization with high probability, the theorem neither give enough justification assuming such distribution as initial transformer policy, nor is it a suitable choice to show the hardness of the task in Theorem 3.


The results in Theorem 3 and 4 are more readable and relevant to the main message. The results appear legitimate and is as not difficult to understand as the previous ones. However the assumption of step-level oracle as curriculum is rather strong. In fact, it trivializes the problem and the result become very close to Kim & Suzuki, as both papers study learning CoT to solve parity tasks. The main difference is that Kim & Suzuki did not consider cases where irrelevant parity tokens exist. But the proof technique is largely the same, and the one-step GD analysis can be similarly derived.

### Strengths
Theorem 3 and 4 provide certain technical contribution, but needs a better way of presentation.

### Weaknesses
The writing of multiple sections are barely readable. Some definition of notations are missing or difficult to find, making the correctness of Theorem 1 not checkable. Even though the Theorem 3 and 4 can be of interest to theory folks, the paper still requires significant rewriting to be presentable. As for the value of Theorem 3 and 4, see above in the summary.

### Questions
None

### Soundness
1

### Presentation
1

### Contribution
2
