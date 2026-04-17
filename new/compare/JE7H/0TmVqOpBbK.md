---
job_id: 5401b1da-a222-4d4f-ae6c-f06942ea164c
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 0TmVqOpBbK.pdf
paper: Scaling Laws Meet Model Architecture: Toward Inference-Efficient LLMs
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper studies scaling laws, architecture design, and inference efficiency for dense LLMs, which fits ICLR’s core areas of representation learning, optimization, and large‑scale learning.

## Minimum Quality
Pass ✅.  
The paper is in English and has all key sections (Abstract, Introduction, Background, Methodology in Sections 3–4, Experiments/Results in Sections 4–5, Related Work in Section 6, Limitations and Conclusion). Methods and experiments are technically substantial, and I do not see a fundamental flaw that would mandate desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find hidden prompts or attempts to manipulate automated reviewing in the provided content.

---

# Expected Review Outcome:

## Summary

The paper studies how architectural choices in dense decoder‑only LLMs, specifically hidden size, MLP‑to‑attention parameter ratio, and grouped‑query attention (GQA), affect the joint tradeoff between pretraining loss and inference throughput under fixed parameter and token budgets.  

The authors introduce a conditional, architecture‑aware extension of Chinchilla‑style scaling laws (Eq. (3)) that calibrates loss as a function of normalized hidden size \(d_{\text{model}}/\sqrt{N}\) and \(r_{\text{mlp/attn}}\), and use it within a search framework (Alg. 1, Eq. (4)) to pick architectures that satisfy a target loss while maximizing measured throughput.  

They train more than 200 models from 80M to 3B parameters on Dolma‑v1.7, show systematic U‑shaped loss curves in Figures 4 and 5, and then use the fitted scaling law to design 1B and 3B models (Panda and Surefire) that attain small but consistent gains in accuracy and up to ~42% higher measured inference throughput compared to LLaMA‑3.2 baselines (Table 1, Figure 7, Table 6).

## Strengths

1. **Careful and extensive experimental sweep across architecture space.**  
   The paper trains >200 dense models across five parameter scales (80M, 145M, 297M, 1B, 3B) and many combinations of \(d_{\text{model}}/\sqrt{N}\) and \(r_{\text{mlp/attn}}\) (Table 4). This is a nontrivial experimental effort, and the resulting patterns in Figures 4 and 5 are quite compelling: at fixed \(N\) and \(D\), the loss vs. \(d_{\text{model}}/\sqrt{N}\) and loss vs. \(r_{\text{mlp/attn}}\) relationships are consistently U‑shaped across parameter scales. The fact that the optimal normalized hidden size and ratio appear fairly stable across 80M–297M is an interesting and actionable empirical finding.

2. **Clear connection between architecture and inference throughput.**  
   The work provides systematic throughput measurements on real serving stacks (vLLM and SGLang) and multiple GPUs (A100 and H200). Figures 3, 9, 10, 11, 12, 13, 14 and the detailed statistics in Table 6 collectively show that, at fixed \(N_{\text{non‑embed}}\), increasing \(d_{\text{model}}\), \(r_{\text{mlp/attn}}\), and GQA yields substantial throughput gains across a wide range of batch sizes. The FLOPs derivation in Appendix K ties these empirical patterns back to theory: Eq. (Total-FLOPs) simplifies to \(2P_{\text{non‑emb}} + 2 n_{\text{layers}} T d_q\), so increasing \(r\) or \(d_{\text{model}}\) effectively shrinks \(d_q\) under fixed parameters, reducing both FLOPs and KV cache size.

3. **Architecture‑aware conditional scaling law with strong predictive quality on held‑out model sizes.**  
   The proposed multiplicative calibration in Eq. (3) is simple but surprisingly effective: Figures 6 and 25 show low MSE and high Spearman correlation when training on smaller scales and predicting loss for larger models (e.g., fit on 80M,145M,297M and predict 1B). Figure 6 explicitly shows that the ranking of architectures at larger N is well captured by the conditional law. This is precisely what matters for architecture search. The paper also includes a thoughtful ablation of additive vs multiplicative forms and of including extreme outlier ratios \(r_{\text{mlp/attn}} \notin [0.5,5]\) (Figure 25), which strengthens confidence in the modeling choice.

4. **End‑to‑end demonstration on 1B and 3B models.**  
   The paper does not stop at fitting curves on small models but actually instantiates larger architectures guided by the scaling law and search framework. Table 1 shows that Panda‑1B improves average accuracy on nine benchmarks by 2.1 points over LLaMA‑3.2‑1B (57.0 vs 54.9) while slightly reducing loss (2.782 vs 2.803), and Panda‑3B and Surefire‑3B slightly beat LLaMA‑3.2‑3B on both loss and average accuracy. Figure 7 (left) is particularly convincing: Panda‑1B lands exactly at the minimum of both the scaling‑law prediction and the actually trained 1B variants. On the efficiency side, Figure 7 (center, right) and Table 6 show substantial throughput gains for Surefire‑1B and 3B across frameworks and hardware, with up to ~42–47% improvements at higher batch sizes.

5. **Methodologically transparent and easy to reproduce.**  
   The training setup is clearly detailed: Dolma‑v1.7 sampling, token budget of \(100 N_{\text{non‑embed}}\) (5× Chinchilla), optimizer and LR schedule (Table 5), and exhaustive architecture list (Table 4). Inference measurement procedures are specified (vLLM/SGLang, 4096‑in/1024‑out, 5 runs, hardware type). The conditional scaling law fitting uses standard Levenberg‑Marquardt least squares. Equation (4) formulates the architecture search problem cleanly, and Algorithm 1 concisely summarizes the pipeline. Practitioners could plausibly rerun these experiments or adapt the framework.

6. **Nice “spicy” empirical insights about how existing open models are suboptimal.**  
   Tables 7 and 8 compare Panda/Surefire models to a set of 1B and 3B open‑weight baselines. The observation that OLMo‑2‑1B is close to the predicted optimal ratio but uses less inference‑efficient choices (e.g., GQA=1), and that LLaMA‑3.2‑3B vs Qwen‑2.5‑3B sit at opposite ends of the accuracy/throughput Pareto front, is insightful. This validates that the framework can both critique and improve upon existing design trends rather than just replicate them.

7. **Figures effectively convey the central story.**  
   In particular, Figure 1 (train‑loss and throughput contours over \(d_{\text{model}}\) and \(r_{\text{mlp/attn}}\)) and Figure 9/10 provide a compelling visualization of the “sweet spot” region where both loss and throughput are favorable. Figure 8 is also good: it reveals that coefficients of the conditional law can shift with scale and that fitting on data near the target size yields better 3B predictions, which is an important caveat for practitioners.

## Weaknesses

1. **Model size and token scale are modest for current LLM practice, which limits external validity.**  
   The largest models studied are 3B parameters trained on 100B tokens, which is 1–2 orders of magnitude smaller in both model and data scale than many widely used LLMs. The U‑shaped loss curves in Figures 4–5 and the fitted parameters in Eq. (3) might change qualitatively at larger scales, where optimization dynamics, data diversity, and regularization regimes differ. The authors partially address this in §5.1 and Figure 8 by showing that coefficients depend on training scale and that fitting on 1B better predicts 3B than using 80M–297M, but they still extrapolate from low‑ to mid‑scale regimes. The paper should be more explicit that these conditional laws are empirically validated only up to 3B / 100B tokens, and that applying them to 7B–70B models is speculative.

2. **Central separability assumption in Eq. (3) is not deeply justified and may fail in important regimes.**  
   Equation (3) assumes that the effect of \(d_{\text{model}}/\sqrt{N}\) and \(r_{\text{mlp/attn}}\) on loss is multiplicatively separable. This is convenient for fitting but not obviously correct. For example, at fixed \(N\), large \(d_{\text{model}}\) with high \(r\) both reduce the number of heads and push more capacity into MLPs, so interactions between them are quite plausible. The paper briefly evaluates one joint non‑separable formulation in Appendix J (Figure 26) and finds worse fits, but that specific parametric form \(\log(dr/\sqrt{N}) + (\sqrt{N}/dr)\) is very restrictive. It is not clear from the presented evidence that the true relationship is separable; the poorer performance of one particular non‑separable form does not rule out other simple joint models. This matters because the architecture search in §3.4 and §5.1, including the optimality conditions \(\partial L / \partial d_{\text{model}} = 0\) and \(\partial L / \partial r = 0\) leading to the Panda configurations, fundamentally rely on separability for clean optimization.

3. **Theoretical treatment of the U‑shaped dependence is mostly descriptive rather than explanatory.**  
   In §3.3 and Figure 4–5, the authors fit \(c_0 + c_1 \log x + c_2/x\) to approximate the empirical U‑shapes in \(L(d/\sqrt{N}|r,N,D)\) and \(L(r|d/\sqrt{N},N,D)\). This choice is somewhat ad hoc: there is no theoretical argument connecting this form to underlying capacity or optimization principles, nor any exploration of how sensitive the fitted minima are to the functional family. For instance, alternative symmetric forms like \(c_0 + c_1 (\log x - \mu)^2\) or rational functions could provide equally good fits but different minima. Since the optimization in §5.1 relies on exact derivatives of Eq. (3) to locate minima, the lack of justification for the specific parameterization raises concern that the obtained optima (e.g., \(d_{\text{model}}/\sqrt{N} \approx 0.08\), \(r \approx 1.0\)) may be artifacts of the chosen functional class rather than robust properties of LLMs.

4. **Use of pretraining loss as the only constraint in Eq. (4) may not tightly capture downstream performance.**  
   The constrained search in Eq. (4) uses a target training loss \(L_t\) obtained from LLaMA‑3.2 models. While Table 1 and Tables 9–10 show that lower loss architectures generally correlate with better downstream averages, the mapping is not perfect. For instance, in Table 1, Surefire‑1B has higher loss (2.804) but slightly better avg score than LLaMA‑3.2‑1B (55.4 vs 54.9) despite sharing the same \(L_t\). Panda‑3B° in Table 2 has lower loss than Panda‑3B but identical average accuracy (62.5). The paper does not explore failure cases where training loss mispredicts downstream ordering, nor does it provide any calibration linking \(\Delta L\) to expected changes in benchmark accuracy. Given that Eq. (4) is the core of the “accuracy constraint” story, it would be valuable to show scatter plots of loss vs. task scores across architectures, or to fit a simple downstream scaling law, to demonstrate that operating at a fixed small band above \(L_{\text{opt}}\) reliably preserves application performance.

5. **Architectural search space and constraints are narrower than the framing suggests.**  
   The title and abstract present the work as a general framework for “model architecture‑aware scaling laws” for LLMs, but the actual search space is quite constrained:  
   - Only dense decoder‑only Transformers are considered; number of layers \(n_{\text{layer}}\) is fixed per scale, so the aspect ratio tradeoff is explicitly excluded (§3.1). This is reasonable for isolating effects of \(d_{\text{model}}\) and \(r\), but the conclusion “toward inference‑efficient LLMs” might be misread as more general than it is.  
   - The per‑head dimension \(d_{\text{head}}\) is fixed to 64 (<1B) or 128 (≥3B), and GQA is treated as a small local search over divisors of \(n_{\text{head}}\), not integrated into the scaling law. Figures 11 and 14 and Appendix I show that GQA has strong effects on throughput and noisy effects on loss, but the paper stops short of providing any principled guidance beyond “enumerate and early stop once worse than GQA=4”.    
   - Important architectural axes such as rotary vs ALiBi, positional encodings, attention types (e.g., sliding‑window, sparse), or normalization schemes are not considered. That is fine for scope, but the claims around “model architecture‑aware scaling laws” would be more accurate if clearly scoped to “hidden size and MLP/attention parameter allocation under fixed layers in dense Transformers”.

6. **Some mathematical details and assumptions deserve clarification.**  
   - In the derivation around attention parameters, the paper writes \(4 d_{\text{model}}^2 \propto N_{\text{attn}} = N_{\text{non‑embed}} \times \frac{r}{r+1}\) under fixed \(r\). This is qualitatively correct but elides the effect of GQA and non‑square projections. It would be helpful to explicitly note that this is an approximation and state the conditions (e.g., ignoring bias, layernorm, and KV projection shrinkage) under which it holds.  
   - In Eq. (3), the factors \((a_0 + a_1 \log(d/\sqrt{N}) + a_2 \sqrt{N}/d)\) and \((b_0 + b_1 \log r + b_2/r)\) are not constrained to be positive. In principle, the optimizer could pick coefficients that drive the product negative, leading to predicted losses below zero. In practice, fitted values in §5.1 are such that the product stays positive on the data manifold, but the paper should discuss any parameter or domain constraints used in fitting to avoid degenerate solutions.  
   - In §5.1, the authors mention solving \(\partial L/\partial d_{\text{model}} = 0\) and \(\partial L/\partial r = 0\) to obtain the 1B and 3B optima. Given that the fitted calibration functions are nonconvex in these variables, there can be multiple stationary points. It would be useful to state whether they checked global optimality (e.g., by scanning a grid as in Figure 1) or could have landed at a local minimum or saddle.

7. **Evaluation is mostly on classic multiple‑choice benchmarks and lacks diversity.**  
   The downstream evaluation focuses on nine standard QA/commonsense datasets (ARC‑E, ARC‑C, LAMBADA, HellaSwag, OpenBookQA, PIQA, SciQ, WinoGrande, CoQA) with zero‑shot accuracy. While these are widely used, they bias toward short‑context, multiple‑choice tasks and do not test code, math, reasoning chains, or long‑context behavior where architecture choices (e.g., GQA and KV cache size) could matter differently. Also, the gains reported are relatively modest: at 3B scale, Panda‑3B and Surefire‑3B improve average accuracy over LLaMA‑3.2‑3B by only 0.6–0.7 points (Table 1, Table 10). This is statistically plausible but not overwhelming. It would strengthen the significance claim to include at least one challenging reasoning or long‑context benchmark and to report confidence intervals or multiple seeds where possible.

8. **Limited discussion of how the framework relates to or complements existing efficiency work beyond architecture scaling.**  
   The related work (§6 and Appendix B) focuses primarily on scaling laws and efficient architectures but spends little time situating the contribution relative to system‑level efficiency studies, compression methods, or broader inference‑efficiency surveys. The work also does not compare to or discuss recent benchmarks that measure efficiency across methods (e.g., structured sparsity, quantization, distillation). This makes it harder to position the practical impact of the proposed architecture search relative to other ways practitioners already speed up inference.

Overall, the paper is technically solid and empirically thorough, but the core modeling choices (separability, functional form) are somewhat heuristic, and the scale and scope are narrower than the headline might suggest.

## Potentially Missing Related Work

1. **Zhou et al., “A Survey on Efficient Inference for Large Language Models”, 2024.**  
   This survey systematically categorizes techniques for inference‑efficient LLMs, including architectural, system, and algorithmic strategies. It is directly relevant for positioning this work, which focuses on architectural design under scaling laws. It should be cited and discussed in §6 (Serving Systems / Inference‑Efficient Model Design) to clarify how the proposed conditional scaling law complements other efficiency strategies.

2. **Chen et al., “Towards Coarse‑to‑Fine Evaluation of Inference Efficiency for Large Language Models”, 2025.**  
   This paper proposes a structured framework for evaluating inference efficiency beyond simple throughput metrics. Since this submission uses vLLM/SGLang throughput under a fixed 4k/1k configuration as the single metric, integrating or at least referencing Chen et al.’s evaluation methodology would help justify the chosen measurement setting and could be noted in §4 (Inference Setup) and §5.1 (Ablation of inference efficiency).

3. **Yuan et al., “EfficientLLM: Efficiency in Large Language Models”, 2025.**  
   EfficientLLM provides a benchmark and empirical study of various efficiency techniques for LLMs. It is closely aligned with the paper’s empirical focus and could serve as a reference point for comparing the gains from architecture tuning versus other methods (e.g., pruning, quantization). This should be mentioned in §6 and perhaps in §5.1 where the authors discuss how much throughput improvement is achievable purely from architecture choices.

4. **Whitmore et al., “Efficient Inference of Large Language Models through Model Compression”, 2025.**  
   This work focuses on model compression for inference efficiency, which is an alternative to architectural redesign under fixed parameter budgets. Including it in §6 would help clarify that the proposed approach is complementary: it optimizes architecture pre‑training time, whereas compression is typically post‑hoc.

5. **Zhang et al., “Efficient Inference for Large Vision‑Language Models: Bottlenecks, Techniques, and Prospects”, 2026.**  
   While focused on vision‑language models, this paper analyzes inference bottlenecks and techniques, which are conceptually similar to the KV cache and FLOP bottlenecks discussed in §3.2 and Appendix K. It would be useful to reference in §6 to signal that the architectural ideas and scaling‑law framework could extend or contrast with multimodal settings.

## Questions

1. **On separability and functional form in Eq. (3):**  
   Could the authors provide more evidence that the loss is approximately separable in \(d_{\text{model}}/\sqrt{N}\) and \(r_{\text{mlp/attn}}\)? For example, do 2D loss surfaces over grids in \((d/\sqrt{N},r)\) differ significantly from the product of the fitted 1D curves? A visualization akin to Figure 1 comparing measured loss vs. Eq. (3)’s prediction over a dense grid at a single \(N,D\) would help.

2. **Robustness of the optimal configuration to alternative curve families:**  
   If you fit a different parametric family for the U‑shaped curves, such as \(c_0 + c_1(\log x - \mu)^2\) or a spline, how much do the implied optima \((d_{\text{model}}/\sqrt{N}, r)\) move for 1B and 3B? If the minima are stable across families, that would considerably strengthen the claim that 0.08 and ~1.0 are “sweet spots”.

3. **Loss vs downstream performance calibration:**  
   Have you examined scatter plots of training loss vs average downstream accuracy across architectures at fixed \(N,D\) (say 1B scale)? A simple analysis like Spearman correlation between loss and average score, or a linear/logistic fit, would quantify how reliable Eq. (4)’s loss constraint is as a proxy for accuracy. If available, please share such plots or coefficients in the rebuttal.

4. **Sensitivity to data distribution and training regime:**  
   All experiments use Dolma‑v1.7 and a fixed \(5\times\) Chinchilla token multiplier. How sensitive do you expect the conditional scaling law coefficients and the optimal architecture to be if the dataset mix, token budget, or optimizer schedule were substantially different (e.g., higher‑quality data, different LR decay)? If you have even small‑scale evidence (e.g., a subset trained on a different corpus), it would be useful to mention.

5. **GQA search strategy and potential integration into the scaling law:**  
   Appendix I shows that loss vs GQA is noisy, but have you tried modeling it with a discrete prior or a categorical regression instead of full enumeration (e.g., treating GQA as a factor level)? Even if not incorporable into Eq. (3), a small conditional model \(L(\text{GQA} \mid d/\sqrt{N}, r, N, D)\) could shorten the local search. Any additional insight into why certain GQA values (e.g., 7 for Surefire‑3B) work better than others at fixed heads would be helpful.

6. **Confidence intervals or variability measures for throughput:**  
   Table 6 and Figures 7, 15, 19, 23 present single throughput numbers averaged over 5 runs. Are the run‑to‑run variances small? Reporting standard deviations or confidence intervals would clarify whether the claimed improvements (e.g., 42% vs LLaMA‑3.2‑3B) are stable across repeats and system noise.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating
3: good.  
The empirical methodology is extensive and reasonably careful, the FLOPs analysis is consistent, and the conditional scaling laws predict losses well on held‑out sizes. Some modeling assumptions (separability, functional form) are heuristic and not deeply justified, and evaluations are limited in scale, but there are no obvious fatal flaws.

## Presentation Rating
3: good.  
The paper is generally clear, with informative figures (notably Figures 1, 3–5, 6–8) and detailed tables (1, 2, 6–10). A few mathematical approximations could be stated more explicitly, and the scope of claims vs actual search space could be sharpened.

## Contribution Rating
3: good.  
The paper contributes a useful conditional scaling‑law formulation that incorporates architectural knobs, along with strong empirical evidence and a practical search framework that yields measurable throughput and accuracy gains over widely used baselines. The work is incremental relative to prior scaling‑law literature and limited to smaller model scales, but still valuable for the community.

## Overall Rating
6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The combination of a large, well‑designed architecture sweep, a simple but effective conditional scaling law, and concrete improvements over LLaMA‑3.2 in both accuracy and throughput makes this a solid, practically relevant submission. The main concerns are about the generality of the fitted laws (scale and dataset), the heuristic nature of the functional form and separability assumption, and somewhat modest accuracy gains. On balance, the strengths outweigh the weaknesses, and I lean toward acceptance.

## Reviewer Confidence
4: confident.  
I am familiar with scaling‑law literature and LLM architecture/serving work, have carefully read the math (Eqs. (1)–(4), FLOPs analysis), and checked figures and tables, though I cannot fully verify all implementation details.