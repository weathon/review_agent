# An evolutionary perspective on modes of learning in Transformers

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
The success of Transformers lies in their ability to improve inference through two complementary strategies: the permanent refinement of model parameters via _in-weight learning_ (IWL), and the ephemeral modulation of inferences via _in-context learning_ (ICL), which leverages contextual information maintained in the model's activations.
Evolutionary biology tells us that the predictability of the environment across timescales predicts the extent to which analogous strategies should be preferred. Genetic _evolution_ adapts to stable environmental features by gradually modifying the genotype over generations. Conversely, environmental volatility favors _plasticity_, which enables a single genotype to express different traits within a lifetime, provided there are reliable cues to guide the adaptation.
We operationalize these dimensions (environmental stability and cue reliability) in controlled task settings (sinusoid regression and Omniglot classification) to characterize their influence on learning in Transformers.
We find that stable environments favor IWL, often exhibiting a sharp transition when conditions are static. Conversely, reliable cues favor ICL, particularly when the environment is volatile.
Furthermore, an analysis of learning dynamics reveals task-dependent transitions between strategies (ICL $\to$ IWL and vice versa). We demonstrate that these transitions are governed by (1) the asymptotic optimality of the strategy with respect to the environment, and (2) the optimization cost of acquiring that strategy, which depends on the task structure and the learner's inductive bias.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors investigate in-context learning versus in-weights learning in Transformers (though I don't see why the experiments couldn't be done with other memory-based architectures, like RNNs).

Adopting meta-learning settings,  both Omniglot and sinusoid regression, they independently manipulate the stability of the (outer-loop, across tasks) task distribution, and the reliability of the (inner-loop, within-task) contextual cues.

Then, testing with a new task allows then to estimate how much the network relies on in-weights or in-context learning, based on whether its answer to the final query input is closer to that predicted from current in-context cues, or from the ongoing training task, respectively. 

They show that higher contextual reliability favors in-context learning, while higher task stability favors in-weight learning. Different tasks have different dynamics. Increasing the number of possible classes decreases preference for in-weights learning in the Omniglot task.

### Strengths
The question is interesting. The experiments are informative. The review of the literature (on both evolution and ML) is nice.

### Weaknesses
There seems to be no fatal flaw in the paper that I can see.

It may be argued that some of the results are not really earth-shattering ("more stability = more overfitting?"), though the additional experiments in Figures 4 and 5 provide more details on the dynamics.

It's not clear how much the "relative cost" hypothesis helps, because there seems to be no precise definition of "cost", except for a-posteriori hardness on IWL? (I note that the parameter used to tune Omniglot task is the total number of classes; however, the sinusoid task has an infinite number of classes, yet it doesn't seem to clearly favor ICL or IWL more than Omniglot, but rather it seems to incline differently for various regimes of stability/reliability).

### Questions
If the authors can make their hypotheses a bit more precise and/or actionable it would probably increase the reach of the paper. Other than that I have no pressing questions.

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
5

### Summary
This paper investigates how environmental predictability influences the balance between in-context learning (ICL) and in-weights learning (IWL) in Transformers. The paper draws many analogies to evolutionary biology's phenotypic plasticity and genetic encoding and the circuit learned in transformers. The authors set up two model systems: sinusoidal regression and Omniglot and set up variables representing environmental stability (task consistency across training) and cue reliability (how informative are in-context examples). Key findings include: (1) high stability favors IWL while high cue reliability enhances ICL, (2) learning dynamics show task-dependent transience patterns (both ICL→IWL and IWL→ICL transitions, this latter seems quite novel), and (3) a "relative-cost hypothesis" suggesting that the computational ease of acquiring each strategy determines preference and transition dynamics.

### Strengths
S1: The experiments are quite novel and interesting, even though some past studies study curriculum learning where the loss landscape keeps changing, studying this together with the question of ICL vs IWL is genuinely interesting.

S2: The **quality** of the paper's presentation is great, even though the framing seems questionable (see W1). The paper is well-written and figures are clear. The experimental design is clean and easy to understand.

S3: The identification of IWL->ICL transience (in addition to previously documented ICL transience) seems novel and interesting.

### Weaknesses
W1: This is my only major concern of the paper. The connection to evolutionary biology seems to be just at the very high level. The framing makes sense, but does not actually offer any falsifiable statements or predictions. While this could have been a nice discussion point, I don't see why this should be a main theme of the paper instead of simply framing it as a study of circuit competition on non stationary losses. Furthermore the training method lacks any kind of evolutionary mechanism such as selection, mutation, reproduction. The authors themselves do acknowledge that there is no direct equivalence. It seems like the paper could be much better linked to other concepts like learning theory, meta-learning frameworks, complexity theory, etc. I really like the experiments and the experiments were genuinely interesting, but I am left confused why such a superficial connection to evolution was made, without discussing core concepts of evolution: G-P mapping, mutations, genetic drift, etc.

However, I am happy to discuss this further, perhaps there is a connection which is genuinely helpful that I'm missing.

W2: Beyond the weaknesses discussed in W1, it seems like the relative-cost hypothesis is a good intuition to have, but at the same time doesn't seem to be too novel compared to classic discussions in simplicity bias, circuit complexity, memorization budget in deep learning.

W3: The choice of learning rate scheduling is questionable when the data distribution is non-stationary. Perhaps this makes the results harder to interpret since it artificially slows down the speed of learning. However, I don't think this will qualitatively change the results too much.

### Questions
It would be good to cite some more papers which also explore IWL vs ICL:

https://arxiv.org/abs/2306.04891 <- this paper seems to co-pioneer the findings on transience, although they didn't focus on presenting it that way.
https://arxiv.org/abs/2412.01003 <- seems directly related to the cost of memorizing more processes and also discuss that circuit complexity slows ICL.
https://arxiv.org/abs/2506.17859 <- also seems related to the relative cost hypothesis.
https://arxiv.org/abs/2506.19351 <- discusses an Occam's razor on complexity.

Q1: I'm not so sure if this is possible, but is it possible to decompose the model prediction in the style of https://arxiv.org/abs/2412.01003, i.e. decomposing the probability itself by the ICL vs IWL probabilities?

### Soundness
3

### Presentation
2

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
The paper investigates when decoder-only Transformers rely on in-context learning (ICL) versus in-weights learning (IWL). Using an evolutionary analogy (phenotypic plasticity vs genetic encoding), the authors operationalize cue reliability and environmental stability and perform dense parameter sweeps on two controlled tasks (sinusoid regression and Omniglot binary classification). They introduce a preference score (SICL) comparing errors against ICL- and IWL-targets, present asymptotic preference maps and training-time transience dynamics, and posit a qualitative 'relative-cost' hypothesis to explain strategy emergence.

### Strengths
The paper comes with a clear, motivating framing that yields testable experimental axes (stability and cue reliability).

The scientific claims are substantiated with systematic empirical evaluation with dense sweeps and temporal analyses across many configurations.

Coherent qualitative findings across two tasks: stability favors IWL; reliable cues favor ICL; transience depends on relative difficulty.

Efficient experimental design allowed broad exploration and reproducibility in principle; hyperparameter grids and compute are reported.

### Weaknesses
I might be mistaken but is there a critical and central inconsistency?: SICL is defined as SICL = EIWL / (EICL + EIWL + eps) but interpreted (and plotted) as higher SICL meaning more ICL.

Insufficient robustness: only 3 seeds per configuration; many key effects (thresholds, transience) need more seeds and statistical tests.

Limited external validity: only two simplified tasks and a single small Transformer; applicability to larger models and potentially LLMs or naturalistic domains is untested.

Missing important ablations/controls: prompt-length (N) sweep, encoder-freeze/pretrain ablations for Omniglot, model-capacity sweep, and explicit conflict trials to directly distinguish ICL vs IWL.

Mechanistic evidence is lacking: no attention/head diagnostics, weight-change tracking, probes, or lesioning to support claims of circuit-level implementation or assimilation into weights.

The relative-cost hypothesis is qualitative and unquantified; no direct cost or sample-complexity metrics are provided to predict transience.

### Questions
Please correct and clarify the SICL definition and interpretation. If it was a typesetting mistake, state the intended formula and re-run affected figures and analyses. As a sanity check, include results for a synthetic pure-ICL and pure-IWL predictor showing the corrected SICL behaves as intended.

Provide explicit pseudocode for the evaluation protocol: how EICL and EIWL are constructed, how evaluator prompts are sampled, number of evaluation examples per measurement, and how conflict trials are generated and scored.

Increase robustness: re-run key configurations (those showing sharp transitions or notable transience) with >=5-10 seeds and report SEMs/confidence intervals and statistical tests for main claims.

Perform the following ablations/controls: (a) vary prompt length N; (b) freeze and/or pretrain the ResNet encoder for Omniglot to localize effects; (c) sweep model capacity (smaller/larger Transformers); (d) include explicit conflict trials during evaluation and report how often models follow prompt vs internal mapping.

Quantify the relative-cost hypothesis: measure steps-to-target-error for ICL-only and IWL-only baselines, parameter-efficiency, or representational complexity, and test whether these predict observed transience directions.

Add basic mechanistic analyses: track layerwise weight changes over training, analyze attention-head patterns, or use linear probes/lesioning to show distinct circuitry for ICL vs IWL and to support any assimilation claims.

### Soundness
3

### Presentation
4

### Contribution
3
