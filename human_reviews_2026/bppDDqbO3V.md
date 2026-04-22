# Dissecting the Role of Positional Encoding in Length Generalization

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
Length generalization (LG) is a persistent challenge for Transformers. Despite recent studies improving the models' LG capability, its underlying mechanisms are still underexplored. To better understand LG, we propose that LG requires alignment of the model’s inductive bias with the task’s computational structure, and validate this view with experiments on Transformers. Focusing on iterative tasks (e.g., Polynomial Iteration, Parity, Binary Copy), we systematically analyze different PEs and find that the misalignment persists for Transformers: the structural bias of softmax attention and computational biases from PEs destabilize LG under extrapolation. Notably, Transformers without positional encoding (NoPE) could show partial LG capability, potentially because implicit position encoding through hidden-state statistics and contextual token distributions preserves the consistent computation in extrapolation, though these signals decay with length, leaving the encoding misaligned with the task. Building on this mechanistic analysis, we introduce a lightweight enhancement—value-side relative coding with logit rescaling—that better aligns inductive bias with task structure. This sustains iterative computation and improves LG, offering insights for future PE design.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on the impact of positional encoding strategies (ALiBi, APE, FIRE, NoPE, RoPE, T5, YARN) on length generalization of Transformers on iterative tasks (specifically Polynomial Iteration, Parity, Binary Copy). The authors propose that successful LG relies on alignment between the iterative task's computational structure and the inductive bias of the positional encoding. Their mechanistic analysis shows that many popular PEs are misaligned with iterative tasks, helping to explain why they often perform worse than NoPE. Finally, they propose modified PEs aimed at improving alignment with iterative tasks.

### Strengths
The mechanistic analysis is clear, convincing, and interesting. The NoPE statistical analysis nicely complements the constructive argument in Kazemnejad.

### Weaknesses
A limitation is studying only iterative tasks. In particular, the Logit controller and Value-side relative PE appear to be specifically designed to improve LG for iterative tasks, but their impact on LG for other types of tasks (such as the many others studied in Kazemnejad) is unclear. To be practically useful, we would hope for PEs that could improve LG on many kinds of tasks, not just a limited subset.

See also questions.

### Questions
Can you more explicitly position the paper relative to Kazemnejad, noting the novel contributions w.r.t. Kazemnejad? Kazemnejad et al. (2023) show the failure of LG and the relative superiority of NoPE over other PEs over a range of tasks (Fig F.5 shows (lack of) LG for Parity for NoPE, T5, ALiBi, APE). Kazemnejad further prove that NoPE can theoretically represent both absolute and relative PEs, e.g. for a specific weight configuration in the first layer, and all subsequent layers, respectively. In my reading, the novelty of the current paper lies in: a specific study of *iterative tasks* only (adding the tasks Polynomial Iteration and Binary Copy to Kazemnejad which already studies Parity), a mechanistic explanation of the specific failure-modes of various PEs for this task, and a new statistical analysis of NoPE’s ability to encode position information (distinct from Kazemnejad’s proof which relies on constructing specific weight matrices). Is this accurate?

Studying 2- and 3-layer Transformers makes sense for the mechanistic analysis where you are looking for particular expected attention patterns, but do you know whether training deeper Transformers (more layers) on the same tasks show the same behavior shown in Figure 3 (i.e. does length generalization still degrade relatively quickly OOD, with NoPE extrapolating better than other choices of PE)? The trend where the LG improves from 2- to 3-layer makes one wonder if it might continue to improve with more depth -- and whether the relative performance of the different PEs might change.

What can the study of iterative tasks tell us about other classes of tasks for which LG is desired? Can we expect the Logit controller and Value-side relative PE to improve (or at least not harm!) LG for other classes of tasks with different structure (e.g. arithmetic, etc.)?

Minor notes:
L17. positional enconding (PE) (abbrev. never introduced)
L175 Typo “Algins”

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper analyzes the role of different positional encoding for length generalization on synthetic tasks like parity, binary copy, and such (typically tasks that can be accomplished by iterative local updates). The paper arguments for the misalignment between inductive biase of position encoding and the task as the key factor harming performance - and tries to propose some fixes that would created a better aligment - e.g. logit control and rescaling.

### Strengths
* Overall interesting analyses
* Sound lightweight extensions (ViPE) that shows effective results.

### Weaknesses
* Lack of benchmarking of ViPE on realistic benchamrks greatly undermines scope of the paper. 
* While the paper provides some valuable insights, part of it feels somewhat "obvious" -- of course, one would think that failure to length generalize is an issue of the lacking the right inductive biase; and adding more task-specific inductive bias, or better invariance-mainetance across length increase, length generalization can be improved. This does not feel like a substantively new insight- although the key strength that redeems the paper is in proposing a potential solution.
* Similar ideas (in the context of RNNs - but the principles seem to translate) have been also explored here [1]. The benchmarks in [1] (including those from its appendix) could be have been also useful to evaluate on.
* Even the proposed method still seems to disgracefully degrades around sequence length 43-48 -- suggesting that the generalization may not scale well. 

[1] Monotonic Location Attention for Length Generalization - Ray Chowdhury et al.

### Questions
n/a

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper investigates the mechanisms behind length generalization in Transformers, proposing that LG depends on the alignment between a model’s inductive bias and the computational structure of the task. Through synthetic experiments on iterative reasoning tasks, the authors analyze various positional encodings, including RoPE,  NoPE, and find that most fail to generalize to longer sequences. They show that NoPE can partially achieve LG via implicit positional signals emerging from hidden-state statistics and contextual token distributions, though these signals decay with length. Building on this analysis, the paper introduces ViPE combining value-side relative coding and logit rescaling, aligning model bias with task structure and substantially improving extrapolation performance.

### Strengths
1. The paper provides a novel explanation of how NoPE implicitly encodes positional information through hidden-state statistics and contextual token distributions, contributing theoretical clarity to understanding Transformers without explicit positional encodings. The proposed method ViPE introduces value-side relative encoding and logit rescaling, significantly improving length extrapolation and demonstrating the practical value of aligning model inductive bias with task structure.

2. The experiments are thorough and clearly presented, covering multiple positional encodings and iterative tasks. The visualization of attention maps and performance degradation effectively supports the paper’s main claims about misalignment and fragility in length extrapolation.

3. The paper offers a fresh view by framing length generalization as an alignment problem between a model’s inductive bias and the computational structure of the task, providing an insightful analytical framework.

### Weaknesses
1. All experiments are conducted solely on synthetic iterative tasks, leaving it unclear whether the conclusions generalize to natural language or more complex reasoning tasks. This considerably limits the paper’s practical value. For instance, in general length generalization settings using pretrained models (e.g., Qwen2.5), would the attention maps still exhibit such clear structural patterns? 

2. Since the paper focuses exclusively on synthetic tasks, and ViPE appears somewhat tailored to tasks with precise computational structures (is that correct?), I wonder how the authors envision its performance on more typical tasks, including natural language and general length generalization benchmarks. Given resource and time constraints, additional experiments are unnecessary, but I would appreciate the authors’ perspective on this point.

3. The analysis of NoPE seems to show only that NoPE can use statistical information to distinguish positions, but not that it actually does so in practice. The experiments in Section 5.3 merely demonstrate that NoPE encodes absolute and relative positions. Should this be considered only a lower bound on NoPE’s capability? That said, the authors’ analysis is valuable in that it inspired the design of ViPE, which is a positive contribution.

However, I’m not fully confident in my own judgment, and I'm willing to adjust my score after seeing other reviewers’ comments and the authors’ rebuttal.

### Questions
See weakness.

### Soundness
2

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
4

### Summary
The paper proposes a formal bridge between LLMs and Algorithmic Information Theory (AIT), arguing that (i) LLM training ``computationally approaches'' the Solomonoff prior by interpreting loss minimization as an implicit program-length optimization, and (ii) LLM next-token prediction acts as a computable surrogate for Solomonoff induction under an approximation assumption. It then uses this lens to give unified explanations for in-context learning, few-shot learning, and scaling laws, and introduces a few-shot example selection rule (pick low-confidence correct examples). This selection rule improves accuracy on three text-classification benchmarks (SMS, emotion, and AG News) with Qwen and Llama models.

### Strengths
1. The paper presents an interesting theoretical connection between LLMs and the Solomonoff prior. 

2. The paper argues how context, few examples, and more compute/parameters drive predictions toward a target distribution. The authors show that low-confidence correct examples are more beneficial for ICL than easy examples.

### Weaknesses
1. The novelty of the paper lies in the connection between LLMs and Solomonoff induction, but from the perspective of the evaluation (ICL with correct samples that have low confidence) this has limited novelty. 

2. Experiments evaluate only few-shot classification with confidence-based selection across three datasets and four instruction-tuned models. There’s no ablation on prompt length, budget K, alternative selection criteria (entropy, gradient-free influence, diversity), or tasks beyond classification.

3. Results show consistent gains for ``low-confidence'' selection, but it is not clear where the models make mistakes. An error analysis would be good. 

4. The paper would benefit from a proofread. See for example, ``Thus, the LLM as a whole functions as a deterministic Turing machine.''

### Questions
What kinds of errors do the models make?

How are the samples in the prompt selected (beyond being correct and low confidence?)

How many samples in the prompt are there?

### Soundness
3

### Presentation
2

### Contribution
2
