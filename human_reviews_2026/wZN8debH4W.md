# Planned Diffusion

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Most existing large language models are autoregressive: they generate text one token at a time, and cannot decode any new tokens until they have decoded every token before it.
Discrete diffusion language models offer a promising alternative by generating multiple tokens in parallel, but sampling from them requires a _denoising order_, the strategy for deciding which tokens to decode at each step.
Determining the right denoising order is difficult, and existing approaches use heuristics that create a steep trade-off between quality and latency.
We propose _planned diffusion_, a system that trains the model to determine its own denoising order.
Planned diffusion uses a single model that transitions between autoregressive and diffusion-based generation: first, the model autoregressively generates a plan that partitions the response into semantically independent chunks, defining a denoising order that parallelizes sampling across chunks; second, the model executes this plan via diffusion denoising.
On AlpacaEval, a suite of 805 instruction-following prompts, planned diffusion achieves Pareto-optimal trade-off between quality and latency, achieving 1.27x to 1.81x speedup over autoregressive generation with only 0.87\% to 5.4\% drop in win rate.
Our empirical results show that planned diffusion exhibits superior performance scaling on downstream tasks compared to autoregressive baselines while offering the runtime flexibility to precisely navigate the quality-latency trade-off.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Planned Diffusion, a hybrid decoding framework that first generates a short autoregressive plan (with control tags like <async>…</async>, <sync/>, topic and predicted span length), then diffuses multiple spans in parallel under a bidirectional mask. It integrates KV-cache reuse across stages and uses an energy-ordered unmasking rule for diffusion. On AlpacaEval (805 prompts), it reports 1.27×–1.81× speedup over AR with 0.87%–5.4% LC win-rate drop, and shows ablations on plan components, span-length prediction, and quality–latency knobs (step-ratio r, confidence threshold τ).

### Strengths
# Strengths

1) Clear plan-then-parallel idea  
- First produce a short AR plan, then decode multiple spans in parallel with a bidirectional mask.  
- Turns semantic parallelism into practical decoding parallelism.

2) Simple, single-model pipeline  
- Causal attention for planning; bidirectional attention within spans during diffusion.  
- KV-cache reuse and stage transitions are straightforward to add to existing inference stacks.

3) Transparent quality–latency control  
- Tunable knobs (e.g., diffusion-step ratio *r*, confidence threshold *τ*) expose a smooth speed–quality trade-off.  
- Pareto curves make the effect of these knobs easy to interpret.

4) Useful diagnostics  
- Component-wise ablations (control tags, span-length prediction, unmasking rule) clarify each module’s role.  
- The explicit plan → execute interface helps surface failure cases.

5) Reasonable reproducibility details  
- Checkpoints, fine-tuning settings, and prompt snippets are documented sufficiently for re-implementation.

### Weaknesses
# Weaknesses

1) Modest speedup and weaker quality
- End-to-end acceleration is limited (≈ **1.27×**).
- Reported quality can trail standard AR decoding (**49.2 vs. 50**).

2) Reliance on prompt-model data construction
- Plan annotations depend on a separate prompt model, introducing data-generation overhead and potential bias.
- The dependence raises questions about scalability of the approach as model/data sizes grow.

3) Narrow experimental scope
- Evaluation is conducted on a **single benchmark**.
- Lacks comparisons against a broader set of **model baselines** and decoding accelerators.

4) Planning limitations on complex tasks
- For tasks with strong cross-span dependencies, plan segmentation and span sizing are uncertain.
- Observed speedup may be confounded by **implicit output-length constraints** rather than true parallelism.

### Questions
ref weakness

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces planned diffusion, a hybrid text generation method combining autoregressive planning with diffusion-based parallel execution. The model first generates a structured plan with control tags defining independent text spans, then generates these spans simultaneously via discrete diffusion. Evaluated on AlpacaEval, the method achieves 1.84× speedup over autoregressive generation with a 6.8% drop in win rate, establishing a new point on the latency-quality Pareto frontier.

### Strengths
1. First text-only model combining discrete diffusion with autoregression in a unified architecture, addressing the speed-quality tradeoff from a novel angle
2. Hybrid attention masking elegantly enables both causal and bidirectional attention; KV caching strategy is well-designed for this architecture
3. Establishes new Pareto frontier point; sensitivity analysis confirms model learns accurate length prediction without systematic bias
4. Method is orthogonal to other acceleration techniques and continues improving with more training data

### Weaknesses
1. Only AlpacaEval benchmark; no evaluation on diverse tasks (summarization, QA, code generation, creative writing). How does performance vary across task types?
2. No direct comparison to other semantic parallelism methods (e.g., Skeleton-of-Thought, APAR, ParaThinker) despite extensive related work discussion. This is critical for establishing true contribution.
3. Relies on Gemini for training data annotation. What is annotation quality? How many examples were rejected? Could this be learned end-to-end without synthetic supervision?

### Questions
1. How does planned diffusion compare quantitatively to other semantic parallelism methods?

2. How does performance vary across task types beyond instruction-following (e.g., summarization, code generation, creative writing)?

3. What is the speedup variance across examples? Are there cases where planning overhead makes it slower than baseline?

4. What percentage of generations have poor plans? Can you provide failure case examples and error analysis?

5. What content types decompose well vs. poorly? Does the method struggle with sequential reasoning or narratives?

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
3

### Summary
The paper proposes Planned Diffusion, a hybrid text generation approach that first plans a response autoregressively (producing structure/length tags) and then fills multiple spans in parallel via discrete diffusion, aiming to shift the latency–quality Pareto frontier. On AlpacaEval, the method reportedly delivers ~1.8× speedup with a modest quality drop versus autoregressive decoding, and includes sensitivity analyses on span-length scaling and denoising step ratio.

### Strengths
- Clear, appealing idea: formal two-stage factorization (planning then parallel diffusion), with an explicit algorithm and attention-masking design. 
- Well-specified control language (<topic>, <async>, <sync/>) that makes semantic parallelism concrete and implementable.
- Empirical evidence of a new speed/quality trade-off vs. AR and diffusion baselines (latency–quality plots, critical-path analysis, scaling behavior).
- Sensitivity analyses help demystify behavior: best performance when using the model’s predicted span lengths (scale=1.0) and a tunable quality–latency knob via step ratio.

### Weaknesses
- Benchmark scope: Results focus on AlpacaEval with an LLM-as-judge (LCWR). This is a useful proxy but not a robust test of coherence/faithfulness across diverse tasks (e.g., reasoning, long-form, safety). Lack of human evals or broader benchmarks (e.g., MT-Bench, GSM-8K reasoning slices, instruction-following suites) weakens generality.
- Baselines & fairness details: Diffusion is configured with steps equal to token count, and fast-dLLM with specific hyperparameters; however, broader ablations (other drafting/verification, semi-AR methods, SoTA speculative decoding stacks) are limited. This makes it harder to judge the absolute Pareto gains.
- Robustness & failure modes: The method assumes reliable plan quality (topic labels, span counts). What happens when planning is wrong (e.g., underestimates length; cross-span dependencies appear late)? The sensitivity section is a good start, but a qualitative error analysis is missing.
- Training data annotation relies on a proprietary LLM (Gemini) to insert tags under constraints. Potential concerns: annotation consistency, domain shift to noisier instructions, and whether the model overfits the tag scheme rather than learning general “semantic parallelism.” More diagnostics would help.

### Questions
- Plan robustness: How often does the planner significantly under/over-estimate span length in the wild? Could a lightweight “repair” step (e.g., local AR patching) recover quality when spans are misplanned?

- Generalization: Have you tried the approach on models with different pretraining (e.g., pure AR LLMs plus diffusion fine-tune) or at different scales?

- Error modes: Is there any qualitative examples where planned diffusion fails (e.g., subtle cross-span dependencies) and discuss mitigation.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
* Problem: Autoregressive models produce high-quality text but are slow because they generate tokens sequentially, while diffusion models generate in parallel but often require many iterative steps to achieve similar quality.
* Solution: A new hybrid architecture, termed planned diffusion, that leverages (1) an autoregressive “planning” stage that decomposes the output into semantically independent spans, and (2) a parallel diffusion “execution” stage that denoises these spans simultaneously.
* Evaluation: Experiments and analysis including demonstrating that the method achieves a 1.84x speedup over autoregressive generation with a 6.8% drop in win rate on AlpacaEval.

### Strengths
1. Well-motivated: very relevant problem and one that addresses a key weakness in diffusion language models
2. Novelty: combining an autoregressive planning stage with a diffusion-based parallel generation stage within a single unified model.
3. Implementation: Proposes reasonable set of methods that includes a new control tag language, model training methodology, and inference algorithm that enable planned diffusion and navigation of a Pareto frontier between speed and performance.

### Weaknesses
1. Evaluation Scope: Evaluation is only on AlpacaEval and lacks any other benchmarks, tasks, or domains. 
2. Baselines: There is only one baseline that is not the vanilla baselines of autoregressive models and diffusion LLMs. 
3. Complexity: Quite a lot of complexity without full ablation to justify each design choice
4. Trade-off: A performance loss of 6.8% is still pretty substantial and it is not clear how much speed-up one could get with say a smaller model or speculative decoding.

### Questions
1. The improvement in speed comes at a cost which is performance. Performance is a much harder thing to raise, so it is hard to understand exactly how much is being sacrificed for the speedup. Is there a way to quantify the speedup with the performance kept constant or to show the performance with the same speed?
2. The following is mentioned: "To the best of our knowledge, this is the first text-only model that uses both discrete diffusion and autoregression." Does the emphasis of "text-only" mean that there are other models in different modalities that use both discrete diffusion and autoregression?
3. How does the model determine the optimal number and boundaries of spans in the autoregressive plan, and how sensitive are results to this segmentation?
4. Could you clarify how the computational cost of the diffusion stage scales with the number of spans and denoising steps?

### Soundness
2

### Presentation
2

### Contribution
3
