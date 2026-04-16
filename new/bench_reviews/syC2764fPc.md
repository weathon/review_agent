## Summary

The paper proposes Context-Alignment (CA), a paradigm for adapting LLMs to time series tasks that goes beyond token-level alignment by establishing structural alignment (via dual-scale nodes) and logical alignment (via directed edges) between TS data and language prompts. The authors develop the Dual-Scale Context-Alignment GNN (DSCA-GNN) framework and instantiate it as FSCA, which segments TS data into demonstration examples and inserts GNN-based mixing layers into frozen GPT-2. Experiments across long-term, short-term, few-shot, and zero-shot forecasting, plus classification, show consistent improvements over baselines.

## Strengths

- **Strong and consistent empirical performance** across a wide range of TS tasks. FSCA achieves the best results in most long-term forecasting datasets (Table 2), short-term M4 (Table 3), few-shot forecasting (Table 4, ~6.7–15.8% MSE reductions vs. closest LLM-based and Transformer baselines), zero-shot cross-domain transfer (Table 5, 13.3% gain over PatchTST), and classification (76.4% average accuracy, Fig. 2). The margins are especially notable in low-data regimes.

- **Creative few-shot construction**: Splitting a single TS into N internal demonstration segments (where later segments serve as ground truth for earlier ones) is an elegant design that is well-matched to the forecasting task and exploits the in-context learning paradigm in a novel way for TS.

- **Ablation study confirms structure matters**: Table 6 shows that removing the DSCA-GNNs (A.1), randomizing the adjacency (A.2), or removing the coarse-grained branch (B.1) all degrade performance, supporting the claim that the hand-designed graph provides a useful inductive bias beyond what random or no structure provides.

- **Dual-scale design is architectically sound**: The idea of maintaining both fine-grained (token-level) and coarse-grained (modality-level) representations with learnable interaction addresses a real issue—long TS sequences lack natural structural boundaries that LLMs expect from language.

## Weaknesses

### Major:

- **Gap between conceptual framing and mechanism implementation.** The paper's central claim is that CA activates LLMs' "deep understanding of language logic and structure" rather than merely performing "superficial token embedding processing." However, what DSCA-GNNs actually implement is a GCN layer with hand-designed edge patterns (e.g., "TS tokens → prompt tokens," "prompt → next TS segment") and cosine-similarity-weighted edges—essentially a structured cross-token mixing layer. There is no evidence that this engages any special "linguistic" capability of GPT-2 rather than providing additional parametric mixing capacity. The edges encode trivial task-specific relations ("previous data predicts future"), not linguistic logic. The paper does not probe LLM internals, analyze learned representations, or otherwise substantiate the claim that LLMs are "contextualizing and comprehending" TS data rather than benefiting from better feature routing. Without such evidence, the paradigm-level claim overreaches what the mechanism delivers.

- **Insufficient ablation to isolate the source of gains.** The ablation study (Table 6) compares FSCA against removing the GNN entirely (A.1), randomizing edges (A.2), and removing the coarse branch (B.1). However, it does not compare against: (i) a simple cross-attention or MLP-based mixing layer with the same parameter budget (which would isolate whether the *graph structure* specifically matters vs. just having extra learnable capacity); (ii) the few-shot segmentation scheme alone without the GNN (which would isolate whether the internal demonstration structure itself accounts for much of the gain, especially in few-shot settings); or (iii) a 2×2 ablation that separately disables structural alignment (dual-scale nodes) vs. logical alignment (directed edges). The current ablations show "our design is better than removing it," but cannot attribute gains specifically to "context-level alignment" versus "extra parametric capacity plus clever prompting."

- **Unequal prompt structure across methods in few-shot/zero-shot settings.** In Tables 4–5, FSCA encodes internal TS segments as demonstration examples—effectively providing the model with an explicit "predict next from previous" structure. It is unclear whether competing LLM-based baselines (Time-LLM, S²IP-LLM, GPT4TS) are given equivalent demonstration formatting or only single-description prompts. If FSCA benefits from a more informative in-context signal that baselines lack, this conflates prompt engineering gains with architectural gains.

### Minor:

- **Single LLM backbone (GPT-2 only).** All experiments use GPT-2 with 4–6 unfrozen layers. Given that the paper claims to "activate and enhance LLM capabilities" as a general paradigm, validation on at least one additional backbone (e.g., LLaMA, GPT-J) would substantiate generalizability. Further, since only a small subset of GPT-2 (4 of 12 layers) is used, it is unclear whether the method truly leverages LLM-scale capabilities or simply operates on a small frozen transformer.

- **Zero-shot evaluation limited to ETT-family datasets.** All cross-domain zero-shot experiments (Table 5) transfer between ETT variants, which share the same data source and differ only in sampling frequency/horizon. This is a relatively easy transfer setting. True cross-domain zero-shot transfer (e.g., ETT → Weather or Electricity) would more convincingly demonstrate generalization.

- **No analysis of learned graph structures.** The paper claims directed edges encode "logical relationships," but never visualizes or analyzes the learned edge weights. Does the GNN learn interpretable routing (e.g., recent patches attend more to prompts), or does it act as a generic learned mixing matrix? This analysis would directly test the paper's conceptual motivation.

- **Computational overhead not reported.** DSCA-GNNs are inserted at multiple LLM layers, yet no comparison of training/inference time or parameter overhead versus baselines is provided. Given that efficiency concerns are common for LLM-based TS methods, this is a notable omission.

### Trivial:

- The notation across Sections 3.2–3.3 is dense and at times inconsistent (e.g., the superscript numbering of $\tilde{z}$ in Form 6 is clarified to be equivalent, creating momentary confusion), but this is a presentation issue.

## Nice-to-Haves

- Test on at least one additional LLM backbone to validate paradigm-level claims.
- Add cross-domain zero-shot experiments (ETT → non-ETT datasets).
- Report standard deviations across multiple runs, especially for the small-margin improvements in long-term forecasting (Table 2).
- Disentangle structural vs. logical alignment via a 2×2 ablation (with/without dual-scale × with/without directed edges).

## Removed Points

- **Missing comparison with recent TS foundation models (e.g., TimesFM, Moirai, Lag-Llama).** Per hard rules, do not flag missing related works; I cannot confirm whether these baselines are appropriate or even existed at submission time.
- **Parameter count fairness with other LLM-based methods.** The DSCA-GNN parameters (linear layers $f_e$, $f_z$, weight matrices $W_k$, interaction matrix) are relatively small compared to GPT-2. This is a minor concern not rising to the level of a methodological flaw.
- **Error bounds / statistical significance.** Single-run evaluation is standard practice for this benchmark suite (following GPT4TS, Time-LLM, PatchTST protocol). Requesting standard deviations is a nice-to-have, not a core flaw.
- **Claim that prior methods "merely align embeddings" is unfair.** The paper does engage with how Time-LLM and S²IP-LLM work (Sec. 2.2), arguing they focus on token-level alignment. While the dichotomy is somewhat overstated (these methods do use prompts), the paper's characterization is not entirely inaccurate—it is a matter of emphasis rather than a factual error.
- **Rigidity of graph construction for arbitrary prompts.** The paper explicitly scopes to VCA (vanilla prompt) and FSCA (demonstration prompt) and shows how each gets a specific DSCA-GNN framework. That it doesn't generalize to arbitrary prompts is a reasonable limitation to note but not a fundamental flaw given the paper's stated scope.

## Novel Insights

The internal TS segmentation strategy for constructing few-shot demonstrations is arguably the most impactful design choice, yet it is discussed as a secondary contribution. The ablations conflate this design with the GNN architecture, making it difficult to assess their relative contributions. The dual-scale node construction (treating entire TS or TS segments as single coarse nodes) is a genuinely useful way to compress lengthy TS sequences into LLM-compatible structures, and this design pattern—more than the "logical alignment" framing—may be the transferable insight for future LLM-for-TS work.

## Suggestions

- **Run a minimal ablation**: FSCA without DSCA-GNNs but with the same few-shot segmentation and prompt duplication fed through GPT-2. This single experiment would cleanly separate the contribution of the demonstration structure from the graph-based mixing.
- **Visualize learned edge weights** in the fine-grained GNN to assess whether interpretable logical patterns emerge, directly validating or refuting the "logical alignment" claim.
- **Report per-dataset classification results** rather than just the aggregate bar chart, and clarify how many datasets used FSCA vs. VCA.

## Score and Decision

**Calibration.** I compared against:
- *Time-LLM* (ICLR 2024, scores 8/8/8/8/3, Accept poster): Strong empirical results, novel repurposing of LLMs for TS, but overclaimed generalization. This paper is weaker—Time-LLM's alignment mechanism is more straightforward and better validated.
- *TEST* (scores 6/8/5/5, Accept poster): Text prototype alignment for TS with LLMs. Similar concept space, but this paper overclaims more.
- *Multi-level aligned embeddings paper* (scores 5/5/3, Withdrawn/Reject): Similar overclaiming issues with mechanism not matching framing, weaker experiments.
- *SensorLLM* (scores 6/5/5/6, Reject): LLM alignment for sensor data with evaluation issues.
- *GraphSTAGE* (scores 6/5/5/6, Reject): GNN for TS, limited validation.

This paper has stronger empirical results than the rejected papers, but its central conceptual claim is oversold relative to the mechanism. The ablations are insufficient to validate the paradigm-level claim. The empirical gains are real and consistent, but their attribution to "context-level alignment of linguistic logic" versus "clever prompting plus extra parametric layers" remains ambiguous. The paper is above the reject-level papers but below Time-LLM and TEST in terms of clarity of contribution.

**Score: 5.5** — The method works well empirically and the few-shot segmentation idea is creative, but the paradigm-level claims about "logical alignment" and "activating LLM comprehension" are not substantiated by the mechanism or the ablations. A more modest framing (graph-based cross-modal mixing with internal demonstration formatting) paired with sharper ablations would make this a stronger contribution.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>