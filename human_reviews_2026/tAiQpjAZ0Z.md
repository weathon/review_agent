# Value-State Gated Attention for Mitigating Extreme-Token Phenomena in Transformers

- Avg Score: 5.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 6, 6, 4

## Abstract
Large models based on the Transformer architecture are susceptible to extreme-token phenomena, such as attention sinks and value-state drains. These issues, which degrade model performance, quantization fidelity, and interpretability, arise from a problematic mutual reinforcement mechanism where the model learns an inefficient 'no-op' behavior by focusing attention on tokens with near-zero value states. In this paper, we propose Value-State Gated Attention (VGA), a simple dedicated and stable architectural mechanism for efficient performing of 'no-op' attention by directly breaking this cycle. VGA introduces a learnable, data-dependent gate, computed directly from the value vectors (V), to modulate the output. Through a theoretical analysis of the underlying gradients, we show that gating the value-state with a function of itself is more effective at decoupling value and attention score updates than prior methods that gate on input embeddings. This creates a direct regulatory pathway that allows the model to suppress a token's contribution based on its emergent value representation. Our experiments demonstrate that VGA significantly mitigates the formation of attention sinks and stabilizes value-state norms, leading to improved performance, robust quantization fidelity, and enhanced model interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a learnable gating mechanism driven by each token’s value vector to modulate its attention output, directly breaking the feedback loop behind extreme-token pathologies like attention sinks and value-state drains. This architectural fix markedly improves Transformer stability, model performance, and quantization fidelity.

### Strengths
1.VGA introduces a value-based gate that reactively regulates attention outputs, effectively breaking the feedback loop behind attention sinks—an improvement over prior input-gated methods.

2.The gradient analysis clearly shows how VGA decouples attention magnitude from value norm suppression, providing a principled fix to the mutual reinforcement cycle.

3.Tests on BERT, GPT-2, and OPT show VGA reduces activation outliers and improves stability without hurting perplexity, outperforming register tokens, learnable sinks, and IGA.

4.VGA yields exceptional INT8 post-training quantization robustness, with negligible performance loss compared to severe degradation in baselines.

### Weaknesses
1.Evaluations are limited to ~125M-parameter models; behavior on billion-scale LLMs remains unknown.

2.VGA requires architecture modification and retraining or fine-tuning, limiting plug-and-play adoption.

3.Experiments focus on language modeling only; generality across modalities or downstream tasks is unverified.

4.Slightly lower raw perplexity than some baselines (e.g., register tokens), suggesting it optimizes for stability over peak task accuracy.

### Questions
1.How does VGA scale to large-scale LLMs and long-context attention?

2.Can VGA be retrofitted into pretrained models via fine-tuning, or must it be trained from scratch?

3.Will value-based gating also help in non-language domains such as ViTs or multi-modal Transformers?

### Soundness
2

### Presentation
2

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
This paper introduces Value-State Gated Attention (VGA), an modification for Transformer models that aims to mitigate extreme-token phenomena (attention sinks, value-state drains). The authors provide an analysis of the gradient dynamics to motivate the mechanism of VGA, arguing VGA decouples high attention allocation from the destructive suppression of value norms. Experimental validation is conducted on a synthetic task and language modeling task, and quantization, show improve performance.

### Strengths
- The motivation is clear and is supported by an empirical validation on a controlled synthetic task.
- VGA shows some improvements on language modeling benchmarks, including better perplexity as well as quantization results.

### Weaknesses
- Reported results lack standard deviations or error bars, making it difficult to assess the reliability and statistical significance of the improvements.
- The paper claims VGA is a general enhancement applicable to any Transformer-based model, but it lacks evaluations beyond language tasks, such as in vision Transformers (e.g., ViT), which would strengthen the generalizability argument.
- An experiment illustrating the disadvantages of IGA over VGA would be valuable. For example, could you extend the results in Figures 4 and 5 to include IGA, showing how it fails to fully mitigate attention sinks or value drains in the same settings?
- Please provide more details on the creation of Figure 5. Explain the annotations (e.g., dots, dashed lines) and why specific training steps like 0.2k and 1k are marked?

### Questions
See weaknesses.

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
4

### Summary
This paper aims to address extreme-token pathologies of transformers, including attention sinks and vanishing value-state vectors. By first attributing these phenomena to the mutual reinforcement loop in a typical softmax attention layer, the authors draw upon control theory to break this loop by proposing the Value-State Gate Attention (VGA) mechanism. In a nutshell, VGA adds a simple gating mechanism that is computed based on the value vectors V to regulate the gradient flow to them. The authors identify that such a mechanism enables a self-regulatory term that dynamically and adaptively adjusts the gradient path, effectively avoiding value drain. Experiments on a synthetic Bigram-Backcopy task and on BERT, OPT-125M, and GPT-2 (124M) show the effectiveness of the proposed mechanism.

### Strengths
1. The method is well-motivated and addresses the mutual reinforcement loop in a novel, mechanistic way. 
2. The method is very simple, and can be incorporated into existing practices with minimal overhead.
3. The paper is also well-written and clear in presentation.
4. Experiments show consistent gains by this simple fix. A nice bonus is the promising results on low-precision settings.

### Weaknesses
1. This is a nitpick. While the experiments show promising results, it is a bit limiting in terms of scales as the experiments only considered sizes of ~100M. This is vastly smaller than modern models of billion-scale parameters. It is therefore a question of whether the same gains can be achieved on larger-scale models. 
2. The experiments mainly focus on language modeling. I think assessing the effectiveness of the proposed method on more diverse task domains such as vision will greatly improve the work.
3. Forgetting gate mechanisms are widely used now and there have been many design choices: per-channel gates, temperature in sigmoid, normalized value states. The current design is simplistic, but could benefit greatly from ablating these different design choices to further enhance the performance of the method.

### Questions
Please see the weaknesses section.

### Soundness
3

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
The paper proposes Value-State Gated Attention (VGA), a lightweight and simple architectural add-on for Transformers that combats “extreme-token” pathologies - attention sinks and value-state drains. The key idea is a reactive, negative-feedback gate computed from the value vector $V_j$ itself, which multiplicatively modulates a token’s contribution at the attention head output. A gradient analysis argues this decouples pressure on value norms from attention-score updates, breaking the mutual-reinforcement loop that drives sinks/drains. Empirically, VGA reduces sink formation on a synthetic task and improves activation stability, perplexity, and post-training quantization (PTQ) robustness on BERT/OPT/GPT-2, with negligible overhead.

### Strengths
- The method is simple and clear. Turning the gate into a function of the value state (not the input) is a crisp design that directly targets the failure mode; the negative-feedback interpretation is compelling.

- The gradient pathway analysis makes the stabilization story plausible and distinguishes VGA from input-gated variants.

- Minimal code/param/compute overhead; orthogonal to attention-score computation; drop-in for many Transformer flavors.

- Presented experiments contain synthetic validation, standard LM backbones, and a relevant application, where extreme activations are especially harmful. Results consistently show fewer sinks, stabler value norms, and quantization gains.

### Weaknesses
- My main concern is that empirics are limited to a small set of baselines. Stronger comparisons against other sink-mitigation families (register tokens, softmax alternatives/clipping, predictive gates, state interventions) would better position VGA.

- No evidence at very large scales or on long-context regimes where sinks/drains become acute. It’s unclear how VGA interacts with KV caching, RoPE/positional schemes, and very deep stacks.

- There should be a formal metrics for the determination of “extreme tokens.” While qualitative/aggregate indicators are shown (norm stabilization, performance), clearer, standardized sink/drain metrics (incidence rates, attention concentration statistics, gradient norms) would strengthen claims.

### Questions
- Is $g_j$ is a scalar (as Eq. 6 suggests)? Any results with vector (per-dimension) gates or applying the gate before vs. after the output projection $W_O$?

- Are gates shared across heads or learned independently? Any empirical difference?

- Any preliminary results on vision or multi-modal Transformers where sink-like effects also appear?

### Soundness
3

### Presentation
3

### Contribution
3
