# Advancing LLM Reasoning with Natural Language and Numerical Feedback

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Recent advances in reinforcement learning (RL) with numerical feedback, such as scalar rewards, have significantly enhanced the complex reasoning capabilities of large language models (LLMs). Despite this success, we identify three key challenges encountered by RL with solely numerical feedback: performance plateaus, limited effectiveness of spontaneous self-reflection, and persistent failures. We then demonstrate that RL-finetuned models, even after exhibiting performance plateaus, can generate correct refinements on persistently failed problems by leveraging natural language feedback in the form of critiques. Building on this insight, we propose Critique-GRPO, an online RL framework that integrates both natural language and numerical feedback for effective policy optimization. Critique-GRPO enables LLMs to learn from initial responses and critique-guided self-refinements simultaneously while maintaining exploration. Additionally, we employ a shaping function to amplify learning from correct, especially unfamiliar, refinements and penalize incorrect ones. Extensive experiments show that Critique-GRPO outperforms all compared supervised and RL-based fine-tuning methods, achieving average Pass@1 improvements of approximately +15.0\%, +21.6\%, and +15.0\% on Qwen2.5-7B-Base, Qwen2.5-Math-7B-Base, and Qwen3-8B across eight challenging reasoning tasks. Notably, Critique-GRPO facilitates effective self-improvement through self-critiquing, achieving substantial gains over GRPO, e.g., a +16.7\% Pass@1 improvement on AIME 2024.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes a method called Critique-GRPO, which extends the GRPO framework by integrating steps of critiques and self-refinement, and shows its effectiveness over the GRPO baseline.

### Strengths
+ The effectiveness is supported by experiments on eight reasoning tasks across three Qwen base models. 
+ Reinforcing the self-refinement abilities during RL fine-tuning may increase the test-time performance.
+ The method design is a reasonable and well-justified extension.

### Weaknesses
+ While the components (natural language feedback, self-refinement) are not new, the paper's novelty lies in their integration into RL with verifiable rewards. However, this integration could be seen as an incremental step. 
+ The introduction of critiques and self-refinement increases the total training time. The efficiency should be reported for comparisons of the additional cost. 
+ The method relies on a much larger LLM such as GPT-4o as the critique model. This makes the comparison between the proposed method and GRPO baseline relatively unfair due to new supervision signals. The authors should also consider other distillation method for comparisons. Though in section 5.4, self-critique experiments are reported, the effectiveness of the results in the main table still relies on a powerful critique model.
+ The experiments are conducted on Qwen models. Please consider one beyond this family for supporting the generalization. 
+ The fundamental learning signal is still based on sparse, outcome-level rewards (e.g., +1 or 0 for the entire answer). The method uses NL critiques to generate a better trajectory to learn from, but it doesn't solve the core credit assignment problem. This weak, outcome-level supervision may still lead to performance plateaus, as the model cannot easily pinpoint which specific reasoning steps are responsible for a failure.

### Questions
N/A

### Soundness
4

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
4

### Summary
The authors introduce **Critique-GRPO** as an online RL algorithm that blends plain numeric rewards with natural-language critiques—indicative, with ground truth, and CoT. The loop is simple: get an initial answer, ask for a critique, refine, and learn from both passes inside a PPO-style update. Compared to vanilla GRPO they drop clipping and the KL penalty, and instead add a policy-shaping term $\(f(x)=\frac{x}{x+\gamma}\)$ with $\(\gamma=0.1\)$ to boost low-probability tokens in the refined response. Training-wise, they keep a small handful of refinements per prompt and say **7:1 initial:refined** works best.

On results, they report gains across **Qwen2.5-7B-Base**, **Qwen2.5-Math-7B-Base**, and **Qwen3-8B** over several benchmarks, with the biggest bump when the critique is **CoT** rather than just a binary signal. The takeaway from the paper is that the natural-language feedback helps break RL plateaus, keeps entropy from collapsing, and nudges the better reasoning and thereby better performance on the benchmarks.

### Strengths
1.  **Straightforward and reproducible.** The method augments GRPO with critique-guided refinements and a simple shaping term. The pseudocode and stepwise description are clear, and the hyperparameter table supports replication.

2.  **Well-motivated.** The paper documents plateaus and limited gains under numeric-only RL, then shows that natural-language critiques—especially CoT—convert a subset of failures into solves. The causal narrative is plausible and supported by analyses.

3.  **Robust empirical gains.** Consistent improvements are reported across multiple math and OOD benchmarks, including on a non-math base and a reasoning-focused backbone. Gains persist under greedy (temperature=0) evaluation.

4.  **Training-dynamics transparency.** The inclusion of entropy and response-length curves, along with a rationale for the 7:1 initial:refined sampling ratio, provides useful visibility beyond end metrics.

### Weaknesses
1. The results are predominantly Qwen based. After reading some papers recently which claim RL w/ random rewards work really well with Qwen (https://www.interconnects.ai/p/reinforcement-learning-with-random), I am skeptical about papers which show gains with qwen models. It would be nice if the authors can replicate the same experiments with a different base model
2. If ground truth is leaked (even partially) in case of Critique w/ GT, the policy can potentially learn to pattern-match GT traces rather than reason. I am curious to know what authors think regarding this. 
3. The method is computationally very expensive than doing RL/SFT since you generate initial responses and refinements, plus the critique tokens and longer contexts for second-pass decoding. 
4. You modify the objective (drop GRPO clipping/KL) and add critiques and add refinement sampling. Would be curious to know where the gains are actually coming from -- are wins from looser optimization or language feedback? Right now this is confounded.

### Questions
1. How did you get the numbers for Figure 2b? Did you use judge to get these numbers? If yes, what kind of judge?
2. Is there any reason why SFT is not done for RL experiments? Do we get the same gains if we do RL on the SFTed model as a base?

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
3

### Summary
This paper introduces Critique-GRPO, an online RL framework that combines natural-language feedback and numerical feedback to enhance LLM reasoning. Building on the GRPO algorithm, Critique-GRPO allows a model to learn jointly from initial responses and critique-guided refinements, where critiques may be rule-based or model-generated. A shaping function further prioritizes correct and unfamiliar refinements. Experiments on three Qwen-series backbones across eight reasoning benchmarks show consistent gains over both supervised and RL baselines, with meaningful improvements and strong self-critiquing capabilities.

### Strengths
Strengths
- The integration of textual feedback into online RL is a neat idea. Unlike prior works that either transform critiques into scalar rewards or use them offline, Critique-GRPO integrates them within the RL loop, enabling simultaneous exploration and refinement
- Comprehensive evaluation and consistent empirical gains. Results across Qwen2.5-7B-Base, Qwen2.5-Math-7B-Base, and Qwen3-8B on eight reasoning datasets demonstrate broad effectiveness, including out-of-distribution generalization
- Analysis section provide qualitative and quantitative insights on entropy dynamics, response-length efficiency, and self-critique exploration, which helps interpret why NLF sustains exploration while improving precision

### Weaknesses
Weaknesses
- Limited novelty beyond combination. Although integrating natural-language feedback with GRPO is useful, the framework mainly reuses known components like textual critique generation (introduced in Critique FT) on top of GRPO. The algorithmic novelty (e.g., removing clipping/KL + adding a shaping weight) is incremental compared to existing works (e.g, Dr. GRPO, DAPO, LUFFY, etc.)
- Heavy reliance on Qwen models and numerical comparisons. All experiments are performed on Qwen-series backbones. It remains unclear whether Critique-GRPO generalizes to other architectures (e.g., Llama-3). Including one non-Qwen model would strengthen claims of generality, as in recent RLVR works[1, 2].
- Ablations and control analyses appears incomplete. The paper compares variants with different critique types, but lacks fine-grained ablations isolating (i) the shaping function, (ii) removal of clipping/KL terms, and (iii) critique quality (noisy vs. clean). Such ablations would clarify which component drives most of the gain.

## References
[1]. Spurious Rewards: Rethinking Training Signals in RLVR. arXiv:2506.10947

[2]. The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning. arXiv:2506.01347

### Questions
- Since the textual feedback is generated by the reward model, how sensitive is Critique-GRPO to the quality and length of CoT critiques? For example, how does the method perform when critiques are partially wrong or adversarial?
- Could the authors provide more details on training cost? Specifically, how is the training efficiency (e.g., wall-time/GPU hours) of Critique-GRPO compared to standard GRPO, and how long does the critique generation take?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Critique-GRPO, an online RL framework for fine-tuning large language models with both numerical and natural language feedback.
Instead of optimizing only scalar rewards as in GRPO or R1-style methods, the model receives textual critiques that guide a refinement step before policy updating.
The authors test this approach on multiple reasoning benchmarks using Qwen-7B and 8B models, showing consistent improvements over R1-GRPO and other offline critique-based fine-tuning methods.
Ablation studies analyze critique types, shaping functions, and the role of online vs. offline training.

### Strengths
The method is well-engineered: integrating natural language feedback into an online RL loop is implemented cleanly and supported by solid ablations.

The study includes careful analyses of critique types, shaping functions, and self-critique variants, which justify each design choice.

Results are consistent, statistically stable, and show meaningful improvements across tasks.

### Weaknesses
1) The conceptual novelty is moderate. The idea of critique-guided refinement has been explored in prior offline methods (Critique-FT, CITL-FT). The contribution mainly lies in bringing it into an online GRPO-style RL framework and tuning several design components.

2) Most innovations—such as removing clipping/KL, adding a shaping function, or optimizing both initial and refined responses—are engineering-level adjustments rather than new theoretical ideas.

3) The study is limited to Qwen-7B/8B models and reasoning benchmarks. There is no evidence of cross-model generalization or scaling analysis, so it remains unclear whether the method truly closes the gap with larger models (e.g., Qwen-32B, GPT-4) or only provides small gains under constrained capacity.

4) The paper assumes that textual critiques are inherently superior to scalar rewards but does not compare against graded or structured reward signals (e.g., partial credit, continuous correctness scores). Without testing these intermediate forms, it is uncertain whether the gains come from the linguistic form of feedback or simply from providing richer supervision.

5) The reliance on GPT-4 to generate critiques limits practicality and makes the training process resource-intensive. Self-critique is promising but still weaker in performance.

6) (Minor) The analysis section remains surface-level; there is no deeper explanation of why critique feedback improves optimization dynamics.

### Questions
1) Could the authors compare their method to a graded reward baseline, where critique text is converted into a numerical score (e.g., 0–5 or partial correctness)? This would clarify whether textual feedback itself brings additional benefits.

2) Have the authors tried larger or different base models (Llama, Mistral, or Qwen-32B) to test the method’s scalability and generality?

3) How sensitive is the approach to the quality of critiques? For example, if the critiques are noisy or inconsistent, does the policy still improve?

4) Could the authors share more analysis on the optimization behavior, e.g. does the critique-guided refinement change token-level gradient patterns compared to GRPO?

5) Is there a plan to make the critique generation process cheaper or partially automated without GPT-4 dependence?

### Soundness
3

### Presentation
3

### Contribution
3
