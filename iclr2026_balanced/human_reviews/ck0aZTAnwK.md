## Human Reviewer 1

### Summary
This work studies how to pre-train when compute is abundant but data is limited. The authors show that standard approaches of increasing epochs or model size overfit, and that stronger regularization, especially much larger weight decay, helps sustain gains. They further show that ensembling independently trained models improves the loss asymptote, and that their best setup, combining regularization, epoching, and ensembling achieves the same performance with 5× less data. Finally, they find that most of their ensembling recipe benefit can be retained through distillation into a smaller model, with gains also reflected on downstream tasks.

### Strengths
The problem addressed is timely. The main ideas like stronger regularization, evaluating by scaling-law asymptotes, and using simple ensembling, are easy to follow and reproduce. The experiments consistently show clear gains and improved data efficiency, and the distillation results make the approach more useful. The paper is well-written, with clear experiments and a natural, convincing link between the sections.

### Weaknesses
See Questions section below.

### Questions
1. The novelty feels incremental in places. Ensembling and stronger regularization are well‑known tools; the main new piece seems to be “optimize for and compare asymptotes under fixed data.” Could you sharpen the claim of conceptual novelty beyond careful tuning + ensembling, and clarify what, if anything, is surprising in the results relative to prior data‑constrained scaling work?
2. Compute‑matching across alternatives could be clearer. For a fixed training FLOP budget, how do (i) a single larger model, (ii) K‑member ensembling, and (iii) spending compute to synthesize more training text compare?
3. Weight decay is pushed high and clearly helps, but other classic knobs (dropout, data noising, etc.) are not explored. Would be interested in knowing your thoughts on whether similar monotone‑scaling and asymptote reductions hold with these (or other) alternatives?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper presents a compelling and timely investigation into language model pre-training for a future where high-quality data is the primary bottleneck, not compute. The authors argue that as compute continues to scale faster than data generation, the community must shift from compute-efficient to data-efficient training paradigms. The work first demonstrates that standard recipes, which involve increasing epochs or model parameters, inevitably lead to overfitting and performance degradation. As a solution, the authors propose a "regularized recipe" using aggressive weight decay (up to 30x standard practice) to achieve stable, monotonic performance scaling. Building on this, they show that ensembling multiple smaller models is a more data-efficient use of compute than scaling a single large model. By combining these methods, they claim a significant 5.17x improvement in data efficiency. Finally, the paper grounds these compute-intensive findings in practice, showing that the gains from ensembling can be largely preserved by distilling them into a single, smaller model, and that improvements in validation loss translate to meaningful gains on downstream tasks.

### Strengths
This is a strong paper that could become a foundational empirical work for the emerging data-constrained training regime. The experiments are thorough, and the insights are valuable for both theoretical and practical applications. Specifically, I appreciate the following contributions of this paper.
1. **Extensive Experiments:** The claims are backed by an impressive scale of experimentation, reportedly over 2000 training runs. I like the scientific approach of diagnosing the failure of a baseline, introducing a robust regularized recipe, and then demonstrating the superiority of ensembling, which builds a convincing, step-by-step argument.
2. **Practical and Generalizable Insights:** The work successfully connects the "infinite compute" thought experiment back to practical applications. The demonstration that an 8-member ensemble can be distilled into a single student model that retains 83% of the performance gain is a crucial result, proving these methods are not purely theoretical. Furthermore, the validation on downstream benchmarks  confirms that the observed improvements in pre-training loss correspond to genuine enhancements in model capabilities.

### Weaknesses
1. **Architectural Specificity:** The empirical results are derived exclusively from Llama-style decoder-only architectures. It remains unclear whether the core findings, particularly the superiority of ensembling over scaling a single model, generalize to different architectures such as Mixture-of-Experts (MoE). More critically, under the infinite-compute regime, transformer architectures may not represent the optimal choice. The paper would benefit from discussion on architectural considerations within data-constrained training regimes.
2. **Reliability of Power-Law Extrapolation:** The paper's headline quantitative claims (e.g., 5.17× data efficiency) depend on extrapolating power laws fitted to limited data points (typically four model sizes). While the authors provide a sensitivity analysis and appropriately advise caution, the absence of formal statistical goodness-of-fit metrics (such as R² or confidence intervals) makes it difficult to assess the true reliability of these asymptotic estimates. The broader literature demonstrates that simple power laws can break down or exhibit sub-scaling behavior at larger scales, making long-range extrapolation risky. This concern is compounded by the non-standard width/depth configuration of the 1.4B model, which introduces a potential confounder that could affect the quality of the fit.

### Questions
The paper convincingly shows that tuned regularization eliminates the initial overfitting peak seen in standard recipes. However, the double descent literature suggests that with sufficient over-parameterization, an unregularized model's performance might enter a "second descent" and improve again. Could you comment on whether this second descent could, in the infinite limit, outperform the regularized recipe? Is there a theoretical or empirical reason to believe the regularized performance curve is strictly superior at all points in the over-parameterized regime?

### Soundness
3

### Presentation
4

### Contribution
4

### Rating
8

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper investigates language model pre-training in a fixed data infinite compute setting. To address overfitting in the data-constrained scenario, the authors propose a recipe with 30x larger weight decay and the asymptote of the scaling law. The final joint scaling recipe yields a 5x data efficiency gain over the baseline.

The authors then propose and demonstrate that model ensembles can be trained and distilled into a single student model while preserving most of the loss improvement.

Lastly, the authors show that their metrics generalize well to downstream benchmarks.

### Strengths
1. The motivation for the paper is practical, where internet data is exhausted. 
2. The empirical studies are thorough and clear. The authors produced a recipe that achieves 5x higher data efficiency and demonstrates the benefit of a much larger weight decay.
3. The level of attention to experiment details is commendable, and experiment setups are described in detail.
4. The settings the authors take into account are comprehensive, with results ranging from pretraining, model ensembling, and distillation.

### Weaknesses
1. Due to the shallow setup for the 1.4B model, it underperforms the 600M version without the proposed weight decay. While in Figure 3, the authors show how the regularized recipe fixes this, this still casts doubt on the following conclusions, given how sensitive power law fits are, and that 1.4B is the largest model variant that the author uses for curve fitting.  For instance, in Figure 9, 1.4B's downstream eval results also break the trend, most likely due to model architecture configuration. Since the authors only train on 200M tokens, this is not prohibitively expensive to fix.

2. The ensemble scaling part requires a heuristic that breaks in the over-parametrized setting. The heuristic itself, when and why it works well or not, could use more explanation to make the ensemble tuning recipe simpler and more principled.

### Questions
1. Is there any intuition behind distilling from a single model working better than distilling the ensemble?

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
8

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper explores a forward-looking problem: how to design pre-training strategies when compute grows faster than data availability. The authors simulate a future where web text becomes the limiting resource and propose several methods to make pre-training data-efficient under fixed data but infinite compute.

### Strengths
1. The paper identifies an under-explored but realistic future scenario where compute scales faster than data. This is a fresh take beyond the traditional “compute-optimal” (Chinchilla-style) scaling laws.
2. The authors construct a controlled 200M-token benchmark and systematically vary parameters, epochs, and regularization. The coordinate-descent tuning of weight decay, LR, and epochs shows strong empirical rigor.
3. The evaluation results demonstrate the effectiveness of the proposed approach.

### Weaknesses
1. The question raised in this paper is quite interesting. However, I am curious whether the conclusion would still hold if the dataset size were fixed between 30B and 100B tokens while training a 1B-parameter model. According to the Chinchilla paper, the optimal number of training tokens is approximately 20 x N, and over-training is quite common nowadays. For instance, the Qwen3 model was pretrained on about 36T tokens. It would be interesting to investigate, when training with more than 20 x N tokens, how to appropriately adjust the learning rate, weight decay, and number of epochs.
2. Could the authors briefly clarify how this paper differs from the Tensor Programs V: Tuning Large Neural Networks via Zero‑Shot Hyperparameter Transfer (https://arxiv.org/pdf/2203.03466)? Additionally, the μP paper shows that hyperparameters are closely tied to the model architecture, and the more recent Scaling Inference‑Efficient Language Models (https://arxiv.org/pdf/2501.18107) builds a scaling law that explicitly incorporates architecture. Could the authors elaborate on the impact of model architecture in their study?
3. The evaluation over three downstream tasks is limited. Is it possible for the authors to show the evaluation over Arc-Challenge, Lambada, HellaSwag, Winogrande, etc?

I would be willing to raise my score if the authors can address the above questions.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3