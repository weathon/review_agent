# ICLR Benchmark Results

Date: 2026-04-01 19:21
Critic/Merger: minimax/minimax-m2.7 (OpenRouter)
Neutral: minimax/minimax-m2.7, Related Work: minimax/minimax-m2.7:online (OpenRouter)

## FCBbh0HCrF

- GT: Accept (Poster) (avg 7.0)
- Predicted: Reject (4.3/10)
- Match: No

### Final Review

## Summary
This paper identifies the unrealistic assumption of synchronous data reception in existing online VFL frameworks and proposes an event-driven online VFL framework where only a subset of clients are activated by events at each round while others passively collaborate. The authors adapt Dynamic Local Regret (DLR) to handle non-convex models in non-stationary environments, providing theoretical regret bounds with additional terms accounting for partial client activation.

## Strengths
- **Problem identification**: The paper correctly identifies a practical gap—prior online VFL work (Wang & Xu, 2023) assumes all clients receive synchronous data streams, which is unrealistic in scenarios like sensor networks or multi-company VFL where data arrives event-driven for only a subset of participants.
- **Comprehensive experimental evaluation**: The paper tests across 9 baseline configurations (3 frameworks × 3 activation schemes), multiple datasets (i-MNIST, SUSY, HIGGS), and both stationary and non-stationary data streams with appropriate ablation studies on key hyperparameters (window length l, attenuation coefficient α, activation probability p, threshold Γ).
- **Theoretical grounding**: The regret analysis extends prior work (Aydore et al., 2019) to account for partial client activation, with the additional term (2W·**G**) specifically derived for missing gradient elements from passive clients (Theorem 1, Remark 1).

## Weaknesses
- **Incomplete view problem limits communication savings**: The paper acknowledges (Section 6) that passive clients must still send embeddings to the server each round to avoid the "incomplete view" problem. This means communication savings come only from the backward pass, not forward pass. The practical benefit is thus more limited than the title suggests.
- **No statistical confidence in experimental results**: Tables 1-4 report single-run error rates and computation/communication costs without error bars or confidence intervals. This reduces confidence in the reliability of the reported improvements.
- **Event threshold requires manual tuning**: Table 4 shows the activation threshold Γ significantly affects accumulated error rate (from 0.0595 at Γ=-0.2 to 0.1057 at Γ=0.6), but no principled method for setting Γ is provided, limiting practical deployability.
- **Incremental algorithmic contribution**: The primary novelty lies in problem identification and applying existing DLR methodology (Aydore et al., 2019) to VFL with partial activation. The event-driven mechanism itself (threshold-based activation) is straightforward.

## Nice-to-Haves
- An ablation varying passive client query frequency would help quantify communication-computation trade-offs more precisely.
- Analysis of how stale embeddings (from passive clients using old data) affect model quality over time in non-stationary settings would strengthen the non-stationary evaluation.

## Novel Insights
The paper makes an important observation that real-world VFL systems (sensor networks, multi-company collaborations) naturally operate in an event-driven fashion where data arrives at only a subset of clients per round. The experimental finding that DLR with partial activation maintains reasonable accuracy while reducing computation by ~30-40% (comparing DLR-Random p=0.5 vs DLR-Full) is practically valuable. The analysis correctly shows that the regret bound degrades gracefully with partial activation through the (1/p_min) scaling term.

## Potentially Missed Related Work
- Recent work on asynchronous VFL (e.g., Chen et al., 2020 on VAFL; Zhang et al., 2024 on asynchronous VFL for kernelized AUC) may be relevant for positioning the event-driven paradigm within the broader asynchronous VFL literature.

## Suggestions
- Include error bars across multiple random seeds to establish statistical significance of reported improvements.
- Add a principled mechanism for setting the event threshold Γ, either through theoretical analysis or an adaptive scheme based on data characteristics.

---

## 5kMwiMnUip

- GT: Reject (avg 1.4)
- Predicted: Reject (1.5/10)
- Match: Yes

### Final Review

## Summary
This paper explores five jailbreaking methods against Large Language Models: Multishot Jailbreaking, Mirror Dimension Technique, Cipher Method, "You Are Answering the Wrong Question" Method, and Textbook Jailbreaking. The authors claim these methods demonstrate vulnerabilities in current LLM safety mechanisms and propose their findings as a potential benchmark against defenses like LlamaGuard.

## Strengths
- The paper addresses a timely and important topic in LLM security, identifying legitimate vulnerability categories such as the tension between conversational coherence and safety protocols, and the exploitation of models' information synthesis capabilities.
- The discussion section (Section 5) provides some conceptual analysis of why certain attack vectors work, such as the trade-off between coherence and safety, and the challenges of fictional framing for creative models.

## Weaknesses
- **No reproducible experimental methodology**: The methods section describes techniques conceptually without providing specific jailbreak prompts, number of trials, evaluation criteria for "success," or which LLM versions were tested. Without these details, the work cannot be replicated or verified.
- **Unsubstantiated claims in title and abstract**: The paper claims to use "chain-of-thought reasoning" as a core approach, but this is never defined, operationalized, or explained in the methodology. Similarly, the abstract claims findings "can serve as a benchmark against emerging security measures such as LlamaGuard," yet no evaluation against LlamaGuard or any defense system is presented.
- **Results appear without supporting methodology**: The conclusion reports specific success rates (e.g., Multishot "0.20 to 0.80," Wrong Question "0.15 to 0.90") but the paper provides no explanation of how these numbers were obtained, sample sizes, confidence intervals, or statistical analysis.
- **Undefined "Reference Method"**: The paper reports results for a "Reference Method" (achieving 1.00 success rate) but this method is never defined or described in the Methods section, making the results incomprehensible.
- **Figures lack interpretability**: Figure 2 contains garbled/unreadable data, and Figure 3 lacks proper axis labels specifying what metric is being measured, undermining the claimed quantitative contributions.
- **No ethical considerations**: As a security paper documenting attack methods, there is no discussion of responsible disclosure to affected model providers, IRB approval, or ethical safeguards—standard expectations for adversarial AI research.

## Nice-to-Haves
- Baseline comparisons to established jailbreaking methods (GCG, PAIR, MasterKey) to establish novelty.
- Analysis of failure cases to understand when methods do not work.

## Novel Insights
The discussion in Section 5 correctly identifies a fundamental tension in LLM design: models trained to maintain conversational coherence and user satisfaction may be systematically vulnerable to gradual prompt escalation attacks. The observation that "coherence-seeking" behavior creates exploitable attack surfaces is a meaningful conceptual contribution, though it requires empirical validation the paper does not provide. The categorization of attack vectors by underlying vulnerability type (context exploitation, fictional framing, obfuscation, correction manipulation, information synthesis) provides a useful organizing framework, even if the specific techniques are not novel.

## Potentially Missed Related Work
- Zou et al. (2023) "Universal and Transferable Adversarial Attacks on Aligned Language Models" — foundational work on automated jailbreak generation that should be compared against.
- Chao et al. (2023) "Jailbreaking Black Box Large Language Models in Twenty Queries" — directly relevant black-box attack methodology.

## Suggestions
- Provide complete experimental details: specific prompts used, number of trials per method, exact model versions tested, and clear definition of what constitutes a "successful" jailbreak.
- Either remove the chain-of-thought claim from the title or explicitly describe how CoT reasoning is integrated into the attack methodology.
- Test against actual defense systems (LlamaGuard, Claude Safety, etc.) to substantiate the benchmark claim, or remove this claim.
- Add a responsible disclosure statement describing how findings will be communicated to affected model providers.

---

## P7f55HQtV8

- GT: Accept (Poster) (avg 6.5)
- Predicted: Accept (6.0/10)
- Match: Yes

### Final Review

## Summary
The paper introduces QuaDiM, a conditional diffusion model for quantum state property estimation (QPE) that generates measurement records conditioned on Hamiltonian parameters. Unlike autoregressive baselines that impose sequential qubit ordering, QuaDiM uses a denoising process that treats all qubits simultaneously. The method is evaluated on predicting correlation functions and entanglement entropy for 1D anti-ferromagnetic Heisenberg models with up to 100 qubits, demonstrating improved performance over baselines particularly in limited-data and low-sample regimes.

## Strengths
- **Novel methodological contribution**: First application of non-autoregressive diffusion models to quantum state property estimation, addressing the conceptual mismatch between sequential autoregressive assumptions and non-sequential qubit entanglement structure.
- **Comprehensive empirical evaluation**: Experiments across multiple system sizes (L=10,40,70,100), two property prediction tasks (correlation and entanglement entropy), and five baselines including classical shadow, kernel methods, RNN, and transformer-based LLM4QPE. Tables 1-2 show consistent improvements.
- **Extensive supplementary analysis**: Appendix includes ablations on positional embeddings (Table 9), denoising steps (Table 8), OOD generalization (Table 7), alternative physical models (Table 6 for XY model), and alternative measurement protocols (Table 10 for tetrahedral POVM), demonstrating robustness across conditions.
- **Clear theoretical formulation**: Appendices A and B provide detailed derivations of the learning objective connecting diffusion model theory to the QPE setting.

## Weaknesses
- **Misleading "unbiased" claim**: The paper emphasizes "equal, unbiased treatment of all qubits" as a key advantage over autoregressive models, but Table 9 reveals that removing positional embeddings severely degrades performance (RMSE jumps from ~0.05 to ~0.45). The model still relies on positional information—it just avoids autoregressive factorization. The framing should acknowledge this nuance.
- **Statistical significance not adequately discussed**: Main paper Tables 1-2 omit standard deviations, which appear in Tables 3-4. At some configurations (e.g., L=70, M=10000), differences between QuaDiM (0.0117 ± 0.0029) and LLM4QPE (0.0155 ± 0.0038) are within ~1.3 standard deviations, making some claimed improvements statistically marginal.
- **Limited physical model validation**: All experiments use 1D Heisenberg chains, but the core motivation—non-sequential qubit interactions—is most relevant for 2D lattices. The claim of practical applicability to "IBM's latest commercial quantum computer" (with 1000+ qubits in 2D architectures) lacks empirical support.
- **Computational overhead under-explored**: Diffusion requires T=2000 denoising steps. Table 8 shows QuaDiM (T=2000) generates only 5.7 samples/sec versus LLM4QPE's 14.6 samples/sec. While reduced-step inference (T=500) helps, the efficiency-quality trade-off deserves more systematic analysis.

## Nice-to-Haves
- Comparison against other non-autoregressive generative approaches (VAEs, normalizing flows) using the same transformer backbone to isolate whether diffusion specifically provides benefit beyond removing autoregressive ordering.
- Experiments on 2D quantum systems to directly validate the core motivation about non-sequential qubit interactions.

## Novel Insights
The paper makes a valuable observation that autoregressive models impose an ordering bias on qubit systems where entanglement structure lacks inherent sequentiality. However, the finding that positional embeddings remain essential (Table 9) reveals a subtlety: the benefit is not "order-independence" but rather "non-autoregressive factorization"—the model can still use positional information but processes all positions jointly rather than sequentially conditioning. This distinction matters for understanding what inductive biases are actually being changed.

## Potentially Missed Related Work
- None identified. The related work section adequately covers autoregressive quantum state modeling (RNN, transformer-based approaches), variational methods (DBMs, VAEs, GANs), and relevant diffusion model foundations.

## Suggestions
- Revise the "unbiased treatment" framing to accurately reflect that QuaDiM removes autoregressive conditioning rather than all positional information. Acknowledge that positional embeddings remain important for performance.
- Report standard deviations in main results tables and discuss which improvements are statistically significant versus marginal.
- Add systematic analysis of computational cost: compare total FLOPs or wall-clock time for training and inference across methods, not just sample throughput.

---

## gc8QAQfXv6

- GT: Accept (Oral) (avg 9.0)
- Predicted: Accept (5.5/10)
- Match: Yes

### Final Review

## Summary

This paper investigates catastrophic forgetting (CF) in Large Language Models during continual instruction tuning by introducing function vectors (FV)—compact representations of task-specific functions extracted from attention head activations—as both a diagnostic tool and basis for mitigation. The authors demonstrate that FV similarity correlates strongly with forgetting patterns across models and tasks, leading to the insight that CF stems primarily from biased function activation (changes in P(θ|x)) rather than overwriting of task functions (changes in P(y|x,θ)). Based on this analysis, the paper proposes FV-guided training with two regularization terms that show consistent improvements across multiple continual learning baselines.

## Strengths

- **Novel cross-domain contribution**: The paper successfully bridges mechanistic interpretability (function vectors) with continual learning, offering a fresh perspective on understanding and mitigating CF. This approach differs meaningfully from traditional parameter-based or replay-based methods.

- **Strong empirical results with consistent improvements**: Table 2 demonstrates that FV-guided training improves existing methods across 4 language models and 4 benchmarks, with meaningful gains (e.g., Llama2-7b-chat with IncLora: +3.34 GP, +25.25 IP). The improvements are not marginal and hold across different base methods (IncLora, EWC, OLora, InsCL).

- **Causal intervention experiments support mechanistic claims**: Findings I and II (Section 5) show that adding source FV during inference recovers performance, while subtracting biased target FV also mitigates forgetting. These interventions provide empirical support for the claimed mechanism.

- **Correlation analysis demonstrates FV superiority**: Figure 6 shows that FV similarity correlates with performance more strongly than last-layer hidden state similarity or parameter L2 distance, validating FV as a useful diagnostic tool.

## Weaknesses

- **No ablation of individual loss components**: The proposed method combines FV consistency loss (ℓ_FV) and FV-guided KL divergence loss (ℓ_KL), but the paper never ablates them separately. Without this, readers cannot determine whether both terms are necessary, or whether one dominates the improvement. This limits understanding of the method's core mechanism.

- **Intervention experiments limited to single model**: The causal pathway claims (Section 5) are supported by intervention experiments (Figure 3) only on Llama2-7b-chat. The paper draws general conclusions about forgetting mechanisms in LLMs, but the empirical support is limited to one architecture.

- **No uncertainty quantification**: All results are presented as point estimates without error bars, confidence intervals, or statistical significance tests. This is a significant omission given the variability inherent in LLM fine-tuning.

- **Negative cases not analyzed**: Table 2 shows some negative results (e.g., FP decrease of -2.80 for InsCL+FVG on NI-Seq-M1, Llama3-8b-chat). The paper dismisses these briefly as "conflict between diverse gradient information and our regularization," but deeper analysis of when and why FVG harms plasticity would improve practical guidance.

- **Hyperparameter choices not justified**: The top-10 head selection and layer choice (layer 9 for KL loss) are not ablated. The paper uses different FV head sets for different models without analyzing sensitivity to these choices.

## Nice-to-Haves

- Analysis of why generation tasks cause more forgetting than classification tasks—this empirical observation (Table 1) is presented without mechanistic explanation despite the paper's focus on FV dynamics.

- Computational cost discussion: FV extraction requires causal mediation analysis, adding overhead that should be quantified.

## Novel Insights

The paper's most compelling insight is the distinction between "biased function activation" versus "function overwriting" as mechanisms of forgetting. The intervention experiments provide direct evidence: adding the source FV (from the original model) during inference recovers performance, suggesting the task function still exists but is not properly activated. This challenges the conventional view that forgetting occurs because new tasks overwrite old task representations. The latent variable interpretation (P(θ|x) vs P(y|x,θ)) provides a useful mental model, though the theoretical justification remains underdeveloped.

## Potentially Missed Related Work

None identified. The paper adequately covers catastrophic forgetting in LLMs and mechanistic interpretability work, including Kotha et al. (2024)'s task inference hypothesis.

## Suggestions

- **Required ablation study**: Add experiments showing GP/IP with only ℓ_FV, only ℓ_KL, and both together to determine each component's contribution.

- **Extend intervention experiments**: Replicate the causal pathway experiments (Figure 3) on at least one additional model (e.g., Llama3-8b-chat) to support the general claims.

- **Add statistical uncertainty**: Report standard deviations across multiple runs (at minimum 3 seeds) for key results in Table 2.

- **Analyze FV head selection sensitivity**: Show results with top-5, top-10, and top-20 heads to validate that the specific head selection matters.

---

## ZHTYtXijEn

- GT: Reject (avg 2.3)
- Predicted: Reject (3.0/10)
- Match: Yes

### Final Review

## Summary

The paper proposes DIRAD, a structural network adaptation method that grows networks via directed edge generation and edge-node conversion (ENC) to resolve "statistical conflicts" where opposing gradients cancel out. It extends this to PREVAL, a continual learning framework that detects new tasks by validating internal node predictions and assigns samples to appropriate models—all without task labels or stored replay data. Experiments on downscaled MNIST and FashionMNIST demonstrate task detection and retention across sequential classification tasks.

## Strengths

- **Novel mechanism for gradient conflicts**: The ENC operation transforms edges into modulatory nodes whose multiplicative structure can align previously opposing gradients—a genuinely new approach to escaping local optima caused by gradient cancellation across samples.

- **Principled growth criteria**: The adaptive potential framework provides a clear, local decision rule for when networks should grow (immediate AP exhausted but total AP nonzero), with a formal neutrality guarantee that structural modifications don't alter existing responses.

- **Conceptual clarity on continual learning**: PREVAL correctly decomposes continual learning into (1) detecting novel data and (2) assigning samples to existing models, with separate L0/L1 networks for task performance and prediction validation.

- **Demonstrated network efficiency for simple tasks**: On 2-class MNIST classification, L0 networks require <20 nodes and <50 edges, compared to >3200 edges in a minimal fully-connected network—confirming dramatic compression for the task networks themselves.

## Weaknesses

- **Theoretical condition is claimed but not verified**: Equation 21 is presented as a sufficient condition for adaptation to proceed, yet the paper explicitly states "We did not check whether this condition would guarantee a global optimum theoretically, and it is an open question." Calling this a theoretical guarantee without proof is a significant overclaim.

- **Severely limited experimental scope**: Experiments use only 14×14 MNIST/FashionMNIST with 2 classes per task and only 3 tasks total. The paper acknowledges that "higher-dimensional experiments were infeasible" due to computational cost, which raises concerns about scalability of the core claims.

- **Baseline comparison is methodologically problematic**: PREVAL (autonomous task detection) is compared against EWC/PNN/MAS (which require explicit task boundary signals). The paper acknowledges these are "upper limits" but still presents performance comparisons that conflate fundamentally different capabilities. A fair comparison would require either (a) giving baselines task labels to compare retention directly, or (b) comparing against other autonomous task-detection methods.

- **No ablation studies despite multi-component system**: The paper explicitly states "we don't perform any ablation analysis." With multiple mechanisms (edge generation, ENC, validation thresholds, priority ordering), understanding which components drive performance is essential but absent.

- **L1 network complexity undermines minimal-size claims**: While L0 achieves impressive compression, L1 networks show "more than 10-fold increase in edges." The claim that DIRAD produces "orders-of-magnitude simpler" networks applies only to task networks, not the complete system required for continual learning.

- **Substantial accuracy degradation and task detection failures**: Accuracy drops from 96% (Task 1) to 73% (Task 3) on MNIST, and 1 in 8 runs fails to detect a task change on FashionMNIST. The paper attributes this to "discernability" differences between classes but provides no diagnostic analysis of failure modes.

## Nice-to-Haves

- Analysis of why certain classes (C3 at 0.62, C8 at 0.50 mean accuracy) perform significantly worse than others—structural investigation could reveal systematic limitations.

- Visualization of L1 network growth trajectories to validate the "directed" and "minimal" nature of adaptation for the prediction networks.

- Sensitivity analysis beyond TCP: many thresholds (R1=5, R2=0.1, Tconf=1.5, TSV=0.01) are chosen without justification and could be overfit to the experimental setup.

## Novel Insights

The ENC mechanism represents a genuinely novel approach to gradient-based learning: rather than accepting that opposing gradients across samples must cancel, it introduces modulatory structure that can selectively amplify or invert gradients per-sample. This reconceptualizes the relationship between network architecture and optimization dynamics—instead of pre-specifying capacity and hoping optimization finds a path, the network grows precisely where gradient conflicts block adaptation. The insight that "total adaptive potential" (sum of gradient magnitudes) can be exploited even when "immediate adaptive potential" (net gradient) is zero is conceptually valuable, though the paper's experimental limitations prevent evaluating whether this translates to practical benefits on meaningful tasks.

## Potentially Missed Related Work

The paper could benefit from discussing:
- Gradient-based architecture growth methods like GradMax (Evci et al., 2022), which also uses gradient information for network expansion but through different mechanisms.
- Neural architecture search methods with gradient-based growth.
- Recent continual learning methods using self-supervision for task detection without labels.

## Suggestions

1. **Run controlled baseline comparisons**: Compare PREVAL's classification accuracy against EWC/PNN/MAS when all methods are given task labels for the training phase—this isolates the retention capability from the detection capability.

2. **Add ablation on ENC vs. edge generation**: Demonstrate that ENC specifically resolves cases where edge generation alone would fail (e.g., XOR-like problems or tasks with known gradient conflicts).

3. **Validate on at least one non-trivial benchmark**: Even single-task CIFAR-10 would demonstrate scalability beyond 14×14 inputs; current experiments don't establish whether the approach works when the "minimal network" claim actually matters.

4. **Formally prove or remove the convergence guarantee claim**: Either prove that the covariance condition (Eq. 21) guarantees convergence to a solution, or reframe it as a heuristic motivation rather than a theoretical foundation.

---

