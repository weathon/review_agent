# SafeMoE: Safe Fine-Tuning for MoE LLMs by Aligning Harmful Input Routing

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 4

## Abstract
Recent large language models (LLMs) have increasingly adopted the Mixture-of-Experts (MoE) architecture for efficiency. MoE-based LLMs heavily depend on a superficial safety mechanism in which harmful inputs are routed safety-critical experts. However, our analysis reveals that routing decisions for harmful inputs drift significantly after fine-tuning, exposing a critical vulnerability to harmful fine-tuning (HFT) attacks. Existing defenses, primarily designed for monolithic LLMs, are less effective for MoE LLMs as they fail to prevent drift in harmful input routing. To address this limitation, we propose SafeMoE, a safe fine-tuning method tailored to MoE LLMs. SafeMoE directly mitigates routing drift by penalizing the gap between the routing weights of a fine-tuned model and those of the initial safety-aligned model, thereby preserving the safety-aligned routing of harmful inputs to safety-critical experts. Experiments on open-source MoE LLMs ranging from 7B to 141B parameters demonstrate that SafeMoE effectively mitigates HFT attacks, reducing the harmfulness score of OLMoE from 62.0 to 5.0, for example, while maintaining task utility within 1% degradation and incurring only 2% overhead. It significantly outperforms state-of-the-art defense methods for safeguarding LLM fine-tuning and remains effective in recent large-scale MoE LLMs such as gpt-oss and Llama 4. Our implementation is available at https://github.com/jaehanwork/SafeMoE.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This manuscript proposes SAFEMOE, a safe fine-tuning method tailored to Mixture-of-Experts (MoE) large language models (LLMs) to address the vulnerability of "safety routing drift"—a phenomenon where fine-tuning (both harmful and benign) causes the model’s routing decisions for harmful inputs to deviate from those of the initial safety-aligned model, thereby deactivating safety-critical experts and undermining safety. SAFEMOE mitigates this drift by introducing a regularization term that minimizes the KL-divergence between the routing weight distributions of the fine-tuned model and the safety-aligned model for harmful inputs. Extensive experiments on 8 MoE LLMs (ranging from 7B to 141B parameters, e.g., OLMoE, gpt-oss, Llama 4) demonstrate that SAFEMOE effectively reduces harmfulness scores (e.g., from 62.0 to 5.0 for OLMoE on the SAMSum task) while preserving task utility (≤1% degradation) and incurring only ~2% training overhead. The work also verifies the strong correlation between safety routing drift and model harmfulness, highlighting the importance of architecture-aware safety designs for MoE LLMs.

### Strengths
- Architecture-Aware Targeting: Unlike existing defenses (e.g., SafeInstr, SaLoRA) designed for monolithic LLMs, SAFEMOE directly addresses the unique vulnerability of MoE LLMs—their reliance on dynamic routing to safety-critical experts. This design fills a critical gap in current LLM safety research, where MoE-specific security challenges have been largely overlooked.
- Strong Empirical Validation: The authors conduct comprehensive experiments across diverse MoE LLMs (varying in parameter size, expert configurations, and vendors) and tasks (dialogue summarization, SQL generation). The consistent effectiveness of SAFEMOE across models (e.g., reducing harmfulness for 141B-parameter Mixtral) and benchmarks (JailbreakBench, HEx-PHI) enhances the work’s reliability.
- Practical Trade-Off Balance: SAFEMOE achieves a favorable balance between safety, utility, and efficiency. It reduces harmfulness significantly without sacrificing task performance (≤1% accuracy loss) and maintains low overhead (~2%), making it suitable for real-world fine-tuning services (e.g., OpenAI/Google API platforms).
- Clear Mechanistic Insight: The introduction and validation of "safety routing drift" as a key metric (via KL-divergence) provides a clear mechanistic explanation for MoE LLM safety degradation during fine-tuning. This insight not only supports SAFEMOE’s design but also guides future research on MoE security.

### Weaknesses
- While SAFEMOE addresses a practical gap, its core technical components (KL-divergence regularization, bi-level greedy optimization) are not fundamentally new—they adapt existing regularization and optimization paradigms (as cited in the paper) to MoE LLMs. Thus, the work does not introduce groundbreaking methodological innovations, limiting its potential to be a "landmark" contribution in LLM safety.
- The study’s threat model implicitly assumes users are either "benign" (conducting standard task fine-tuning) or "moderately malicious" (injecting a small fraction of harmful samples, e.g., 500 out of 5.5k samples in HFT). It does not rigorously evaluate scenarios where attackers use purely malicious datasets (100% harmful instructions) to fine-tune the model, leaving uncertainty about SAFEMOE’s robustness against extreme adversarial intent.
- Multiple figures (e.g., Figure 1b, 5b) show that safety routing drift and harmfulness scores increase disproportionately rapidly in the early stages of fine-tuning (e.g., first 100 steps for OLMoE). The manuscript does not provide a detailed mechanistic explanation for this phenomenon (e.g., whether it stems from rapid updates to gating network parameters or skewed gradient flow from task/harmful samples), weakening the depth of its analysis.

### Questions
Please refer to the "Weaknesses" Section.

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
5

### Summary
This paper study harmful fine-tuning defense on MOE LLMs. As a solution, this paper propose a simple regularizer to mitigate the drift of the routing weights.

### Strengths
1. This is the first paper that systematically study harmful fine-tuning on MOE models. 
2. This paper discovers an important finding that harmful fine-tuning makes the routing  weights drift away from that of the aligned model. 
3. The proposed method is simple and should work in my understanding.
4. Paper is extremely well written. The idea is presented clear with empirical evidence as motivations. The main experiments are also very comprehensive, justifying the effectiveness of the proposed method.

### Weaknesses
1. **A baseline should be added to comparison given their similarity.**
I suggest the authors to add a comparison with Lisa [1]. Similar to SAFEMOE,  Lisa adopts a proximal term which also somehow helps mitigate the harmful drift (although not only for routing weights). In addition, Lisa adds a few safety data in the fine-tuning process, which probably can help improve the mitigation effect. Therefore, I am wondering whether SAFEMOE can outperform Lisa even though Lisa has additional safety alignment data in the fine-tuning process. 

2. **Add a few discussions  and do some exploration on expert-wise regularization.** I first need to mention a related MoE drift control method. Please check Section 5.1 in H3fusion[2].  They shows that by **adding more regularize intensity** to safety MOE expert, the drift of this safety MOE expert during fine-tuning will be smaller. However, they show that this regularizer is not beneficial, because large regularization intensity of safety expert will make the model **becomes even more unsafe** (See the stage 4 in Figure 1). This conclusion is **contradictory** to yours. The reason of this discrepancy is probably that they are adding regularizer to only the safety expert but you are adding uniform regularization to all the experts together.  Combining the two papers, I think the a potentially better solution might be: add the proximal regularization to the non-safety expert but don't add proximal regularization (or add negative proximal regularization) to the safety expert. I conjecture the gate drift will be uniformly become larger for all the experts (but not in the case like Figure 4 that you display), and constraining that of the safety expert will not help.  My conjecture might be or might not be wrong. Could you perform this experiment to verify this alternative idea? 

 

[1] Lisa: Lazy Safety Alignment for Large Language Models against Harmful Fine-tuning Attack

[2] H3Fusion: Helpful, Harmless, Honest Fusion of Aligned LLMs

3. Some related work are missing. I list out a few most recent work below and please consider to read and discuss them. 




Unleashing the Unseen: Harnessing Benign Datasets for Jailbreaking Large Language Models

No, of course I can! Refusal Mechanisms Can Be Exploited Using Harmless Fine-Tuning Data

Harmful Fine-tuning Attacks and Defenses for Large Language Models: A Survey

Emergent Misalignment:Narrow finetuning can produce broadly misaligned LLMs

ESTIMATING WORST-CASE FRONTIER RISKS OF OPEN-WEIGHT LLMS

Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents 

Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight LLMs

Tamper-Resistant Safeguards for Open-Weight LLMs

CTRAP: Embedding Collapse Trap to Safeguard Large Language Models from Harmful Fine-Tuning

Self-Destructive Language Model

Targeted Vaccine: Safety Alignment for Large Language Models against Harmful Fine-Tuning via Layer-wise Perturbation

Vulnerability-Aware Alignment: Mitigating Uneven Forgetting in Harmful Fine-Tuning

LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning

Covert malicious finetuning: Challenges in safeguarding llm adaptation

Shape it Up! Restoring LLM Safety during Finetuning

AsFT: Anchoring Safety During LLM Fine-Tuning Within Narrow Safety Basin

 Gradient Surgery for Safe LLM Fine-Tuning 

Panacea: Mitigating Harmful Fine-tuning for Large Language Models via Post-fine-tuning Perturbation

Refusal-Feature-guided Teacher for Safe Finetuning via Data Filtering and Alignment Distillation

When Style Breaks Safety: Defending Language Models Against Superficial Style Alignment

Anchoring Refusal Direction: Mitigating Safety Risks in Tuning via Projection Constraint

Why LLM Safety Guardrails Collapse After Fine-tuning: A Similarity Analysis Between Alignment and Fine-tuning Datasets

Toward Secure Tuning: Mitigating Security Risks from Instruction Fine-Tuning

Safety Subspaces are Not Distinct: A Fine-Tuning Case Study

MoGUV 2: Toward a Higher Pareto Frontier Between Model Usability and Security

Layer-Aware Representation Filtering: Purifying Finetuning Data to Preserve LLM Safety Alignment

Virus: Harmful Fine-tuning Attack for Large Language Models Bypassing Guardrail Moderation 

Finetuning-Activated Backdoors in LLMs

 Safety alignment should be made more than just a few tokens deep

Mitigating Fine-tuning Risks in LLMs via Safety-Aware Probing Optimization

Fundamental Safety-Capability Trade-offs in Fine-tuning Large Language Models 

Navigating the safety landscape: Measuring risks in finetuning large language models

Scaling Trends for Data Poisoning in LLMs

Towards Resilient Safety-driven Unlearning for Diffusion Models against Downstream Fine-tuning

Fundamental Safety-Capability Trade-offs in Fine-tuning Large Language Models

Detecting Adversarial Fine-tuning with Auditing Agents

Your Task May Vary: A Systematic Understanding of Alignment and Safety Degradation when Fine-tuning LLMs 

There might be more exciting papers that are worth to read in this field and I encourage the authors to read more related papers and keep update on the field.

### Questions
I don't have further questions as I am kind of familiar with both harmful fine-tuning and MoE. This paper excites me, inspire me thinking and I do wish to get more insights from you.  I will consider to increase my score if the authors are willing to explore more on the direction I point out (see weakness 1&2), but otherwise I am also happy to keep my score.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a defense method against harmful fine-tuning attacks on MoE LLMs. The core idea is to regularize the MoE routing decisions during fine-tuning.

### Strengths
* The proposed method is specially tailored for MoE LLMs according to their routing nature, which prior work has not considered.
* The experiments span across MoE LLMs of diverse sizes, and include important ablation studies on method hyperparameters.
* The paper writing flows well overall.

### Weaknesses
* Questionable motivation. The authors show that harmfulness score is highly correlated with safety routing drift. However, is harmfulness score also correlated with other weights (non-routing weights) of the MoE model? And if the method regularizes the drifts in other weights rather than the routing weights, will the harmfulness score also be suppressed? Without these ablations, the motivation to "preserve the initial routing decisions" is questionable.

  Accordingly, I think it's necessary to compare your method with the fine-tuning defense in [1], where the authors also propose to constrain the fine-tuned weights to the original weights.

* Lack of consideration of more complicated attacks and adaptive attacks. The experiments solely focus on  the attack setting of explicitly mixing "500 harmful samples from BeaverTails" in the fine-tuning dataset. However, there are many different HFT attacks nowadays. I would suggest the authors consider more complicated HFT attacks (e.g., [2]), and even backdoor HFT attacks [3,4]. Additionally, the authors should discuss potential adaptive attacks against their proposed defense.

  I'm particularly interested in seeing the results of backdoor HFT attacks, where a trigger is used for activating harmful LLM responses (the model would refuse harmful requests if a trigger is not present). I doubt whether this backdoor attack setting can break the proposed defense -- since SafeMoE can only preserve the routing decisions of plain harmful instructions (where trigger is not included), but cannot regularize the case when backdoor triggers are secretly injected into the harmful instructions.

* Lack of ablation on $T_{reg}$. According to Table 3, it seems the overhead of SafeMoE is quite low. So what if the method performs regularization steps more regularly? Can you show the tradeoff of overhead v.s. safety across diverse choices of $T_{reg}$?

* Lack of human evaluation of the reliability of Llama-Guard-4-12B.

[1] Safety Alignment Should Be Made More Than Just a Few Tokens Deep. ICLR 2025

[2] Covert Malicious Finetuning: Challenges in Safeguarding LLM Adaptation. ICML 2024

[3] Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To! ICLR 2024

[4] Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training. Arxiv

### Questions
* In Figure 1, step 150, the safety routing drift of both SafeMoE and SaLoRA are similarly low (~0.7*1e-3), but their harmfulness scores are quite discrepant -- ~55% for SafeMoE and ~15% for SaLoRA. Why?
* In Figure 3, why use different benign datasets across the two settings (Alpaca v.s. SAMSum)?
* Can you include a comparison with directly freezing the router weights during fine-tuning (rather than regularizing them)?
* Does your method only work for MoE LLMs? Or can you adapt your method to work on non-MoE LLMs too?

### Soundness
1

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
The authors consider the problem of preventing subsequent fine-tunes of an aligned MoE-based LLM from eliciting harmful behavior or capabilities. In particular, this setting assumes the attacker can choose the fine-tuning data mix, but otherwise has no access to the model parameters or the training setup. Such attacks are referred to as Harmful Fine-Tuning (HFT). This threat model is relevant for e.g. the OpenAI fine-tuning API, where (potentially malicious) users can fine-tune OpenAI models on their own data. 

Prior work has identified the existence of safety-critical experts in aligned MoE models. The authors posit that one of the core mechanisms of HFT’s success in MoE models is routing drift, where the expert assignment changes substantially during fine-tuning, avoiding safety-critical experts during inference on harmful prompts, and allowing harmful outputs to be elicited. 

Given this, the authors propose adding a regularization loss supervising the routing scores to remain similar to the initial aligned model throughout training. This loss is computed on a set of harmful prompts, which need not feature in the training set provided by the attacker. 

Algorithmically, the authors propose a bi-level greedy optimization setup that alternates between minimizing the SFT loss and their routing regularization loss. 

SafeMoE on a range of mixture-of-experts LLMs and tasks by simulating harmful fine-tuning (mixing standard task data with a small set of harmful prompts) and measuring both safety and utility. Utility is assessed on SAMSum summarization (ROUGE-1) and SQL generation (exact match), safety is judged on JailbreakBench via an automated moderator, and comparisons are made against fine-tuning-stage baselines (SafeInstr, SaLoRA) and post-hoc edits (Antidote, SafeDelta). Across small and large MoE models, SafeMoE consistently cuts harmfulness dramatically while preserving or slightly improving task performance, remains effective under full-parameter fine-tuning with minimal training overhead, and shows robustness in sensitivity studies (varying harmful sample count, ratio, and regularization temperature). Analyses of training dynamics further link its gains to reduced routing drift - especially in upper layers—helping argue that the safety improvements stem from stabilizing expert selection rather than sacrificing capability.

### Strengths
- Presentation: the paper is clearly written and well-presented, which facilitated understanding its motivation, main proposed method and experiment results. 
- Strong results: the results indicate SafeMOE achieves significantly lower harmfulness scores than other methods, while preserving fine-tuning accuracy.  
- Low overhead: SafeMOE can be run at a low compute overhead by only sporadically optimizing the routing regularization loss. This makes it cheap for practitioners to adopt the method, making it more likely to become relevant for the community.

### Weaknesses
- Correlation vs. causation 
  - The author’s proposed correlation analysis in Section 3 seems to me to be confounded, and to conflate correlation with causation. The authors track routing drift and harmfulness score through fine-tuning, and observe that higher routing drift is associated with a higher harmfulness score. However, given that the harmfulness score of the final model is known to be higher than of the initial model (based on e.g. prior work), there are several metrics one could track that would also increase during training, even though they might not causally affect the harmfulness score. For instance, the distance between initial and final parameters likely increases during fine-tuning as well, and yet one would not conclude from this that “the further the initial parameters are from the final parameters, the more harmful the final model would be”. 
  - In this case, since the author’s hypothesized mechanism is routing drift, I believe a more appropriate and controlled experiment would be to freeze the routing weights to be those of the initial model, and evaluate the harmfulness score during fine-tuning when using those frozen routing weights at inference time. If the harmfulness score becomes significantly lower, this would give more direct and less confounded evidence to the authors’ claim.  
- Lack of stopgrad in $L_{reg}$ definition 
  - My understanding, from reading the code (lines 866–899 of `safemoe_trainer.py`) is that, in the routing regularization function, no stopgrad is applied to the input of the routing layer. Given this, my understanding is that the gradients of the routing regularization loss would propagate to the the residual stream vectors, and hence to all previous layers of the model; not just to the routing. 
  - Given this, it is not clear to me that all this regularization loss is doing is enforcing the routing to stay the same. For instance, it is possible that the effect of the method is to cause activations to remain largely the same, which would lead to a different conclusion as to how SafeMOE achieves its results. 
  - It would help put the authors’ contributions into perspective if we had an ablation where stopgrad is applied to the activations fed into the routing module. If the results are similar to those of SafeMOE, we can be reassured that the reason SafeMOE works is in fact due to avoiding routing drift. If not, it might be that a different effect is at play, and it would be important to understand what that is. 
  - Conversely, another potentially relevant ablation would be to use a regularization loss that simply supervises residual stream activations to stay the same for the harmful prompts (e.g. via an $L^2$ loss). My prior is that this would harm fine-tuning accuracy a lot more than SafeMOE does, but it would be interesting to check, in case the authors can run such an experiment easily (e.g. on a small model).

### Questions
- Why would it be prohibitive to optimize $L_{\text{reg}}$ together with $L_{\text{sft}}$, for instance using a Monte Carlo estimate of the $L_{\text{reg}}$ gradient? Mechanically, this would correspond to sampling a batch from $D_h$ at each optimization step. Is this because the sequence batch size is small, so that adding even a handful of harmful prompts to the batch would be prohibitive?  
- When defining routing drift and $L_{\text{reg}}$, what is the authors’ intuition for why the reference measure should come from the current model checkpoint ($w$) instead of from the aligned model ($w_\text{align}$)? I.e. why do you use $\mathrm{KL}(\text{aligned} || \text{current})$ instead of $\mathrm{KL}(\text{current} || \text{aligned})$? Do you expect this should make a difference, or should it not matter much? My understanding is that the more common form in KL regularization for fine-tuning is to use the latter. 

I will increase my score if the concerns regarding causal claims in section 3 and the absence of stopgrad in the regularization loss definition are addressed. The main things I would like to see in these directions are the ablations highlighted in the Weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
3
