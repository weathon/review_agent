# On the Effect of Positional Encoding for In-context Learning in Transformers

- Decision: Reject
- Scores: 6, 4, 2, 6, 4

## Abstract
Transformer models have demonstrated a remarkable ability to perform a wide range of tasks through in-context learning (ICL), where the model infers patterns from a small number of example prompts provided during inference. However, empirical studies have shown that the effectiveness of ICL can be significantly influenced by the order in which these prompts are presented. Despite its significance, this phenomenon has been largely unexplored from a theoretical perspective. In this paper, we theoretically investigate how positional encoding (PE) affects the ICL capabilities of Transformer models, particularly in tasks where prompt order plays a crucial role. We examine two distinct cases: linear regression, which represents an order-equivariant task, and dynamic systems, a classic time-series task that is inherently sensitive to the order of input prompts. Theoretically, we evaluated the change in the model output when positional encoding (PE) is incorporated and the prompt order is altered. We proved that the magnitude of this change follows a convergence rate of $\mathcal{O}(k/N)$, where $k$ is the degree of permutation to the original prompt and $N$ is the number of in-context examples. Furthermore, for dynamical systems, we demonstrated that PE enables the Transformer to perform approximate gradient descent (GD) on permuted prompts, thereby ensuring robustness to changes in prompt order. These theoretical findings are experimentally validated.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper provides a theoretical analysis of how positional encoding (PE) affects the in-context learning (ICL) capability of transformers. It derives sufficient conditions on the transformer’s weight matrices that guarantee permutation invariance. For both linear regression and dynamical system tasks, it also proves that the error scales as $O(k/N)$ when $N$ is sufficiently large, demonstrating robustness of Transformers to prompt perturbations.

### Strengths
This paper derives sufficient conditions on the weight matrices that ensure permutation invariance in transformers with PE. Moreover, it quantitatively provides error bounds of  $O(k/N)$ for both linear regression and dynamical system task. Overall, this paper makes a valuable contribution, showing that Transformers remain robust to the permutation.

### Weaknesses
While this paper is theoretically strong and sufficiently novel, it is currently limited to linear tasks and considers only absolute positional encoding.
However, I think that these limitations are acceptable for the scope of this paper. Out of interest, I would like to ask the questions below.

### Questions
1. In the nonlinear case (e.g., logistic regression), would a similar quantitative analysis of output error bound be possible ? Although the error bound may differ from $O(k/N)$, could we still expect the bound decreasing when $N$ is large?

2. For other types of PE (e.g., Relative PE, Rotary PE), do you expect similar theoretical interpretations to hold, or would the underlying analysis have to be fundamentally different?

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
This is my second review of this manuscript. Compared to the previous version, there are no substantial changes in this revision.

This paper mainly investigates how positional encoding and different permutations of in-context examples affect the model’s output in in-context learning tasks. The authors provide theoretical results showing that positional encoding can enhance robustness to perturbations in the prompt order, and they further validate these findings with experiments on linear regression and dynamical system tasks.

### Strengths
1. The paper provides a mathematical quantification of the relationship between positional encoding and input order perturbations, which is novel.

2. It designs two empirical tasks — one order-dependent and one order-independent — to validate the theory, demonstrating a well-reasoned experimental setup.

### Weaknesses
1. Multiple experimental runs were not conducted, and the chosen tasks are relatively simple.

2. The positional encoding used in this paper appears to impose certain constraints on the input length of the model.

### Questions
1. I noticed that in the equation on page 4, line 202, the authors omitted the higher-order terms of $B$. In this case, is it still rigorous to use the equality sign?
2. I have some questions regarding the one-hot positional encoding used by the authors, and please correct me if I have misunderstood. If the hidden state dimension $D$ is fixed, since the positional encoding satisfies $p_i \in \mathbb{R}^N$, this implies that the number or length of in-context examples cannot exceed $D$ — more precisely, it cannot exceed $D - d - 2$. In other words, the number of in-context examples (or context length) is constrained by the dimension of the hidden state, which should generally not be the case. We would consider raising the score if the authors could provide further clarification on this issue.
3. I suggest that the authors perform multiple experiments on the same synthetic dataset with different model initializations to improve the robustness of the results. It would also strengthen the paper if more complex (preferably nonlinear) models and tasks were included in the experiments to enhance persuasiveness and generality.

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
5

### Summary
This paper theoretically investigates how positional encoding (PE) affects in-context learning in Transformers when the ordering in the prompt changes. The authors use one-hot positional encoding and prove that prediction changes as O(k/N) where k is the permutation degree and N is the number of examples, for both linear regression and first order difference equations (differential?). They show that under specific weight matrix conditions, Transformers can maintain permutation invariance despite PE. Experiments on synthetic tasks validate the O(k/N) scaling relationship. The authors claim PE enables robust ICL on order-sensitive tasks.

### Strengths
S1: The paper provides clean theoretical bounds (Theorems 3.1, 3.2) with explicit convergence rates for changes in the prompt order dependence.

### Weaknesses
W1: The impact of the core contribution is unclear. The fast that positional encoding affects output when prompt order changes is quite an expected result. The "mystery" that transformers show order sensitivity despite architectural permutation invariance is immediately resolved by noting PE exists.

W2: Mostly theoretical contributions. It is hard to extend these theoretical contributions to actionable insights. In fact, it isn't even clear if these theorems help growing intuition for how ICL should behave in real language models. Moreover, the theory doesn't seem to predict new phenomena.

W2: One-hot PE is completely disconnected from reality. As far as I know, no modern LLMs, or even transformers trained for specific purposes. uses one-hot positional encoding. How these results would impact RoPE embeddings should have been discussed.

W3: Existence proofs are hardly surprising. Proposition 1 and Condition 3.1 show that "there exist" Transformers satisfying certain properties. But it is hard to really understand the impact of such statement. In reality, a vast amount of computation can be expressed from a Transformer, and the existence of such a weight isn't surprising, nor belief changing.

W4: The "dynamical systems" task is ill presented or explained.

W5: There are no validation or even qualitative checks on real language models or other transformers trained on real data.

### Questions
Q1: How do these results extend to RoPE or any PE scheme actually used in modern LLMs?
Q2: Are there any results on real-data trained Transformers which even qualitatively justify the main finding?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper theoretically studies how positional encoding (PE) affects the in-context learning (ICL) capabilities of Transformers. The authors consider two representative tasks—linear regression (order-invariant) and first-order dynamical systems (order-sensitive)—and establish formal results showing that (1) PE introduces bounded deviations in output under prompt permutation, with errors scaling as O(k/N); and (2) for dynamical systems, PE enables approximate gradient descent behavior, providing robustness to order changes. Their experiments confirm the theoretical claims.

### Strengths
1. This paper studys an important yet underexplored aspect of ICL—how prompt order and positional encoding interact and provide interesting theoretical results.
2. This paper is clearly written with well-motivated sectioning.

### Weaknesses
1. The main theoretical results are mostly constructive in nature (e.g., there exists a Transformer). However, it remains questionable whether real-world Transformers actually behave according to these theoretically constructed results.

2. The theoretical results are not particularly surprising — constructing specific Transformers to demonstrate how positional encoding improves performance on time-series tasks is rather expected.

3. Experiments focus on synthetic data. Adding natural sequential tasks or ablations with alternative PE types (e.g., RoPE, ALiBi) would strengthen the contribution.

### Questions
1. Could the authors provide more experiments on natural sequential tasks or include ablations with alternative positional encoding types (e.g., RoPE, ALiBi)?

2. Could the authors provide some theoretical results that go beyond the constructive setting and better reflect realistic scenarios?

### Soundness
3

### Presentation
3

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
This paper analyzes how positional encodings (PEs) shape in-context learning (ICL) when the order of exemplars is perturbed: under an idealized attention model with explicit absolute positions, it proves output sensitivity to a (k)-degree permutation shrinks roughly as (O(k/N)) with the number of shots (N), gives a sufficient condition under which attention remains effectively permutation-invariant, and provides a constructive mechanism showing that PEs can implement approximate gradient-descent–style updates on simple sequential tasks; the trends are validated on synthetic linear regression and first-order dynamical systems, and the results motivate practical prompt/evaluation hygiene (fix or average exemplar order, use more shots, small permutation ensembles), though external validity for modern softmax attention and popular PEs (RoPE/ALiBi/learned APE) is not established.

### Strengths
*   A novel analysis that formalizes bounds linking order sensitivity in ICL to permutation degree (*k*) and the number of shots (*N*).
*   A constructive mechanism (approximate gradient descent) that links explicit position signals to robustness on order-sensitive sequence tasks.
*   Clean synthetic experiments whose trends match the theory, yielding actionable recommendations for prompt and evaluation hygiene (e.g., fixing/averaging order, adding shots, light ensembling over orders).
*   The paper is well-presented, clearly articulating when positions break permutation invariance in ways that are beneficial for ICL behavior.

### Weaknesses
*   The analysis targets an idealized setting (concatenated one-hot absolute PE + linearized attention), so its applicability to the more common transformer setting of softmax attention with RoPE, ALiBi, or learned PEs remains unclear.
*   Generality to complex, real-world data is unproven, with no tests on pretrained LLMs or on heavy-tailed/multimodal inputs.
*   The sufficient conditions (e.g., for permutation-invariance; matrix closeness assumptions) seem strong and are not shown to emerge during standard pretraining.
*   Experiments are small-scale and synthetic; there is no ablation on modern architectures or reporting of the constants/prefactors that would govern real-world effect sizes.

### Questions
*   Do the *O(k/N)* trends and robustness claims hold for non-Gaussian, heavy-tailed, or multimodal real-world datasets, and can they be observed in small open-source LLMs with standard PEs?
*   Can the theory (or new bounds) be extended to sinusoidal/learned APE, RoPE, and ALiBi under softmax attention? If not, where exactly does the analysis break, preventing it from guiding our understanding of real ICL?
*   Would it be possible to empirically measure how closely pretrained models satisfy your sufficient conditions? Could these conditions be relaxed or reinterpreted to make them more verifiable in practice?

### Soundness
3

### Presentation
3

### Contribution
2
