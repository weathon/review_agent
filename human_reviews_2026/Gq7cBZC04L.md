# Steering Language Models for Theorem Proving

- Avg Score: 3.50
- Decision: Reject
- Scores: 0, 4, 6, 4

## Abstract
Recent progress in automated theorem proving leverages Large Language Models (LLMs) for their capacity to comprehend informal mathematical statements and generate corresponding formal proofs. Even though these techniques perform well, very little exploration has been done to understand how language models interpret and utilize these informal mathematical cues to generate formal proofs more effectively. To address this, we explore activation steering, a lightweight, inference-time mechanism that identifies linear directions in a model’s residual activations corresponding to informal “thought” traces, and nudges those activations to improve proof construction entirely without finetuning. Unlike previous approaches, activation engineering offers valuable insights into language models’ internal reasoning dynamics encoded in their activation space. We evaluated these activation vectors on two distinct tasks: formal proof generation from formal theorems and formal proof generation from informal problem descriptions. Our contributions are twofold: (1) we propose an activation-based intervention technique to guide proof synthesis in LLMs; and (2) improve performance across two different decoding strategies without additional training.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper proposes an activation-steering approach for neural theorem proving by computing steering vectors as differences of mean residual activations between prompt pairs that do v.s. do not include informal natural-language reasoning, then add these vectors at selected layers during inference. 
They claim this both sheds mechanistic light on how informal reasoning is encoded and improves proof search without training. Experiments are reported on Llemma‑7B, InternLM2‑7B, and InternLM2.5‑StepProver, with results on MiniF2F and PutnamBench.

### Strengths
1. Compute‑efficient knob: The method is a parameter‑free inference‑time intervention. The paper describes a simple pipeline (difference of means; residual addition; layer selection) that is easy to reproduce conceptually. Algorithm 1 is clear.

### Weaknesses
1. Minimal novelty beyond known activation‑steering: the paper does not introduce new objectives, diagnostics, or provably better layer/scale selection. CAA [2] already articulate the core technique and its caveats. 
The core mechanism of the paper is a direct application of contrastive activation addition (difference-of-means steering) and residual injection, but the paper neither validates linearity assumptions in this domain nor provides rigorous sensitivity analyses for layer choice and scaling factor. The “valley” heuristic is only qualitatively motivated by a cosine-similarity plot; no statistical tests or robustness checks are provided.

2. Baselines appear misconfigured/outdated. The paper’s InternLM2.5‑StepProver baseline (48.2% on MiniF2F) is substantially below InternLM2.5‑StepProver’s own paper [1], which reports 65.9% on MiniF2F‑test (and significantly stronger results elsewhere). Without reconciling search budgets and evaluation protocol, the claimed 18‑point gain may largely reflect a weak baseline rather than a strong method.

3. Key claims rely on extremely fragile evaluations. Reported improvements on MiniF2F use one specific sampling/search setting, with no seeds, CIs, or significance tests. On PutnamBench, the improvement is 6→7 solved (Lean; 0.9%→1.1%), which is within run-to-run variance for theorem provers and is not accompanied by error bars or per-category breakdowns.

4. Comparison to current SOTA is missing. As of 2025 July (the contemporaneous cutoff of iclr 2026), DeepSeek‑Prover‑V2 reports 88.9% on MiniF2F‑test and solves 49/658 PutnamBench problems; Seed‑Prover reports 100% MiniF2F and strong Putnam/IMO performance. The paper neither compares against nor discusses these systems, making it hard to judge practical relevance. 

5. Typos: "Roc1" on p. 9 => "Rocq", "dataset(Lin et al." => "dataset (Lin et al.". Also, the spelling "Lean‑STaR" and "LeanSTaR" are inconsistent. Also wrong model name: "Lemma (Azerbayev et al., 2024)" => "Llemma (Azerbayev et al., 2024)" 

[1] Wu, Zijian, et al. "InternLM2. 5-stepprover: Advancing automated theorem proving via critic-guided search." 2nd AI for Math Workshop@ ICML 2025. 2025.

[2] Panickssery, Nina, et al. "Steering llama 2 via contrastive activation addition." arXiv preprint arXiv:2312.06681 (2023).

### Questions
see weakness

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
3

### Summary
The authors propose a simple method of activation steering to inject informal reasoning into an LLM for formal theorem proving. The authors also demonstrate this improves performance in downstream tasks.

### Strengths
The proposed activation steering method requires only forward passes on a trained model, which is very computationally lightweight. The method of constructing the steering vector (difference of means) is lightweight but effective. The authors demonstrate that steering increases performance on downstream tasks in three models.

### Weaknesses
In the last year there has been an abundance of work in long-CoT provers, such as DeepSeek-Prover-V1.5/V2, Goedel-Prover, Self-play Theorem Prover, Kimina-Prover. In fact all major formal theorem-proving LLMs since 2025 are long-CoT. They explicitly answer questions that the authors seek to answer: “how do informal reasoning patterns inform formal proving within a model’s internal representations?” (L85) and “how a model processes and integrates informal guidance with formal reasoning” (L118), by performing informal chain-of-thought reasoning before generating the formal proof. Other scaffolds such as DSP+ and Hilbert also follow an informal planning stage followed by a formal proof stage.

The authors have not mentioned or compared their perspective to such recent work. Instead, the models the authors tested (Llemma, InternLM2, etc) are from early 2024 and all predate long-CoT models. Since virtually all recent prover models use the chain-of-thought format, this seems to limit the significance of this work to practitioners in LLM-based theorem proving. There are some questions to be answered before this work can be actually used, such as if activation steering applies to any recent long-CoT prover model, and what the conceptual difference is between long-CoT prover models and activation steering (or is the difference only in computational cost?). For this reason I am hesitant to recommend this paper for ICLR.

### Questions
On L297–298, the authors mention that “we additionally examine proof characteristics including average length, tactic distribution, and the frequency of intermediate lemma usage (via the `have` tactic)”. Where is the analysis of tactic distribution and frequency of `have` tactics?

Minor suggestions:

- L42: Lemma -> Llemma

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes an activation steering method for automated theorem proving with LLMs: it computes contrastive “informal-reasoning” vectors from paired prompts (with/without natural-language sketches) and injects these vectors into the residual stream at selected layers during inference. The authors motivate the approach with the hypothesis that reasoning features are captured by approximately linear directions, and present a layer-selection procedure based on cosine similarity "valleys." On MiniF2F and PutnamBench, steering improves proof success rates for several 7B math-tuned models under both sampling and best-first decoding, without parameter updates. The work also analyzes where steering is most effective (later layers), how it changes proof length distributions, and discusses limitations and transfer to Rocq.

### Strengths
1. Novelly frames informal NL guidance itself as a steerable linear direction in the model’s residual stream and applies it to theorem proving, distinct from prior fine-tuning or retrieval approaches.

2. Provides a simple, contrastive difference-of-means construction for steering vectors from paired prompts, adapted to proof settings. Demonstrates non-trivial gains on MiniF2F and improvements on PutnamBench Lean.

3. Evaluates across three math LLMs and two benchmarks, reporting pass rates and ablations (layer sensitivity, search budgets, proof-length effects). Provides a LoRA comparison, showing favorable parameter-efficiency of steering (competitive without training).

4. The paper is well structured, with intuitive figures and concrete hyperparameters. It clearly states assumptions (approximate linearity).

5. Shows scaling with search budget and evidence that benefits concentrate in later layers and in shorter proofs, which is a useful guidance for future theorem-proving pipelines. The early sign of cross-system transfer (Lean-trained vectors helping Rocq in at least one case) also hints at portability of reasoning directions. All of those make this work valuable for future works to reference.

### Weaknesses
1. The paired prompts come from Lean-STaR-style data and an internal filtering step; it’s unclear how sensitive results are to the exact pairing scheme, dataset domain, and prompt formatting. For example, authors can provide robustness checks: different pairing heuristics, smaller data subsets, and cross-domain steering vectors (e.g., algebra vs. geometry only).

2. PutnamBench gains are modest, and Rocq discussion rests on a single highlighted success. It would be good to include some more analyses to establish reliability beyond MiniF2F.

3. While vectors are derived from NL-augmented prompts, the mechanism could also capture style/format or search-friendly biases rather than genuine reasoning. To fix this, consider control experiments against, for exmaple, (a) steering vectors from semantically shuffled NL, (b) from synthetic boilerplate, and/or (c) from unrelated NL text, to isolate causal factors.

4. Claims about "more structured reasoning" would benefit from automatic metrics. Table 3 is a good start but not sufficient for mechanism claims: example evaluations can include rates of have/calc usage, lemma reuse, etc.

### Questions
1. LoRA is the only training baseline. Missing are previous baselines that use inference-time alternatives or retrieval-augmented proving. Can consider to add head-to-head comparisons with matched compute.

2. The Rocq success is intriguing. Can you report some more results in addition to a single anecdote? Also, does the system transfer happen for other proof assistants as well?

3. Slightly inconsistent terminologies: e.g., "miniF2F" and "MiniF2F" both appear.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper explores activation steering, a fine-tuning-free, inference-time intervention, to improve LLMs' ability to generate formal mathematical proofs. The authors observe that informal mathematical reasoning expressed in natural language can provide important structural guidance for proof construction, but existing models rarely use it effectively. The authors hypothesize that informal reasoning induces distinct activation patterns in the model’s residual stream, and these can be linearly isolated and reapplied to improve formal proof generation. Concretely, they construct steering vectors that capture these patterns by contrasting model activations between prompts with and without informal reasoning. Injecting these vectors into transformer layers during inference steers the model toward reasoning-rich proof trajectories without changing model weights. The authors evaluate on MiniF2F and PutnamBench and show the proposed method improves theorem-proving success rates across multiple models such as Llemma-7B, InternLM-2, and InternLM2.5-StepProver.

### Strengths
1. The concept of steering model activation to improve theorem proving is novel and interesting. 
2. The proposed activation steering method is lightweight, fine-tuning-free, requires only forward passes to compute steering vectors, making it readily pluggable into existing LLMs.
3. The authors demonstrate strong gain brought by activation steering on MiniF2F, +18.2%.

### Weaknesses
1. With activation steering, while short proofs improve significantly, long or highly compositional proofs show limited benefit and sometimes even degraded performance due to noisy reasoning insertions.
2. The gain is noticeable for InternLM2.5 but modest for smaller models.
3. The robustness of activation steering is unclear with respect to prompts and hyperparameters.
4. It would nice to evaluate the effect of activation steering on more recent state-of-the-art theorem-proving models such as Goedel Prover.
5. The paper lacks comparisons with frontier theorem-proving frameworks such as Seed-Prover [1], Goedel Prover [2], and LLM-based provers such as GPT-5, Qwen-235B, Claude Sonnet, and Grok.



[1] Chen, Luoxin et al. “Seed-Prover: Deep and Broad Reasoning for Automated Theorem Proving.” ArXiv abs/2507.23726 (2025): n. pag.

[2] Lin, Yong et al. “Goedel-Prover-V2: Scaling Formal Theorem Proving with Scaffolded Data Synthesis and Self-Correction.” ArXiv abs/2508.03613 (2025): n. pag.

### Questions
Do the authors have any insights on why activation steering brings significant gains on MiniF2F but only minimal improvements on PutnamBench? Similarly, why is the gain more noticeable for InternLM2.5 compared to smaller models? Under what conditions does activation steering tend to work well, and when does it fail?

### Soundness
2

### Presentation
3

### Contribution
2
