# REA-RL: Reflection-Aware Online Reinforcement Learning for Efficient Reasoning

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Large Reasoning Models (LRMs) demonstrate strong performance in complex tasks but often face the challenge of *overthinking*, leading to substantially high inference costs. Existing approaches synthesize shorter reasoning responses for LRMs to learn, but are inefficient for online usage due to the time-consuming data generation and filtering processes. Meanwhile, online reinforcement learning mainly adopts a length reward to encourage short reasoning responses, but it tends to lose reflection ability and harm performance. To address these issues, we propose REA-RL, which introduces a small reflection model for efficient scaling in online training, offering both parallel sampling and sequential revision.  Besides, a reflection reward is designed to further prevent LRMs from favoring short yet non-reflective responses. Experiments show that both methods maintain or enhance performance while significantly improving inference efficiency. Their combination achieves a good balance between performance and efficiency, reducing inference costs by 36\% without compromising performance. Further analysis demonstrates that our methods are effective by maintaining reflection frequency for hard problems while appropriately reducing it for easier ones without losing reflection ability. Code is available at https://github.com/hexuandeng/REA-RL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes REA-RL, a reflection-aware online reinforcement learning framework aimed at improving the efficiency of reasoning in Large Reasoning Models (LRMs) without sacrificing accuracy. The work addresses the challenge of overthinking in chain-of-thought (CoT) reasoning, where models often generate excessively long, redundant reasoning steps that increase inference cost with limited benefit.

REA-RL introduces two key components:

Reflection Model: A lightweight model trained to identify the earliest point in a reasoning trace where the correct answer appears. Based on this, the model trims subsequent “overthinking” tokens and generates a concise revision. This enables efficient sequential revision that complements parallel sampling in online RL.

Reflection Reward: A new reward function based on the density of reflective tokens (e.g., "wait", "but") that penalizes non-reflective outputs, which often result from using only length-based rewards.

The proposed framework is implemented within a Grouped Relative Policy Optimization (GRPO) pipeline and combines both the original and revised responses for policy updates. Experiments on several math reasoning benchmarks (e.g., GSM8K, Math500, AMC23) show that REA-RL can reduce token usage by 36% while maintaining or even improving accuracy. The analysis further demonstrates that the system preserves reflection on hard problems and reduces unnecessary reflection on easier ones, achieving a better balance between efficiency and performance than prior approaches.

### Strengths
- The paper introduces a reflection-aware revision framework (REA-RL) combining a lightweight reflection model and reflection reward, achieving a 36% inference efficiency gain with no performance loss.  
- The detection method is principled, with a clear revision boundary strategy and efficient implementation.  
- The reward shaping component addresses the typical degeneration from using length-only rewards in online RL.  
- Experiments are thorough and well-structured, with both ablations and scaling comparisons.

### Weaknesses
- Limited to 7B distilled models need 32Bmodel for guidance; lacks generalization validation to pre-trained or larger LLMs.  
- The reflection detection relies on LLM-based heuristics, which may not scale or generalize across domains.  
- Added complexity in training pipeline (sequential revision) increases compute by ~10%.

### Questions
Truncation is somehow brutal? what if answeer first, and explain later.
when calculate  overthink signals ,wait, but , alternativly, is somehow way too heuristic?
and chose 20th  for punishment, do we have experiment to support this parameter choose?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces REA-RL, an online reinforcement learning framework designed to mitigate overthinking in reasoning language models. REA-RL employs a reflection model to detect and truncate redundant reasoning segments, producing revised trajectories for policy learning. It further incorporates a reflection reward to encourage appropriate self-correction and a refined length reward that promotes concise reasoning only when answers are correct. With this method, REA-RL achieves up to 36% shorter reasoning traces without accuracy degradation across multiple reasoning benchmarks, demonstrating more efficient and balanced reasoning behavior.

### Strengths
1. The paper presents an original and well-motivated idea of integrating reflection awareness into reinforcement learning to mitigate overthinking in reasoning models.
2. The approach is clearly explained, with intuitive motivation, and supporting experiments and ablations.
3. Experimental results are consistent and persuasive, demonstrating significant reductions in reasoning length while maintaining accuracy.

### Weaknesses
1. The revision model design seems questionable. The paper reports that using the revision model alone achieves even better results than revision model + gold answer, which is counter-intuitive. Since matching the correct answer is not a difficult task, this suggests that the revision mechanism may not be well aligned with correctness or may overfit to surface reflection patterns. Clarifying why this happens would strengthen the paper.
2. The reflection reward based on reflection-token density might lead to reward hacking. Models could insert reflective keywords (“wait”, “check”, “however”) without performing genuine self-correction. A more semantic or context-sensitive reflection metric could better ensure that the reward encourages meaningful reflection rather than stylistic mimicry.

### Questions
I’m curious about the training setup: the paper mentions that both the original and revised responses are used together during online updates. Have the authors tried using only the revised responses for training? It would be interesting to see whether excluding the original (possibly overthinking) trajectories leads to more stable learning or better efficiency–accuracy trade-offs.

### Soundness
4

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
The paper introduces REA-RL, a reflection-aware online reinforcement learning framework that improves the efficiency–accuracy trade-off of large reasoning models by curbing overthinking without discarding beneficial reflection. The core idea is to augment standard grouped online RL with two complementary mechanisms. First, a lightweight reflection model is trained to detect the first point in a sampled trajectory where the answer is already present; tokens after that point are treated as overthinking and the path is revised by truncating the think segment and prompting the policy to finalize the answer. This yields a sequential-revision signal that, when combined with parallel sampling, creates an additional scaling dimension and can be interpreted as a partial advantage that penalizes overthinking tokens while rewarding the revised completions. Second, a reflection-aware reward measures the density of reflective cues (e.g., wait, but) and penalizes paths whose reflection density falls in the lowest quantiles, thereby preventing the length reward from collapsing the model into non-reflective, error-prone plans. The authors also refine the length reward by zeroing it on incorrect generations. Across five math benchmarks and two generation budgets, the approach maintains or improves accuracy while reducing token cost by up to 36% on average relative to the original R1-7B initialization, and analysis shows it preserves reflection on harder problems while appropriately reducing it on easier ones. A workflow schematic on page 5 clarifies how parallel sampling, reflection-guided truncation, and reward shaping interact during training, and tables on pages 6–8 report the main and component-wise results supporting these claims.

### Strengths
The work is original in how it operationalizes reflection within online RL: rather than only rewarding short outputs, it explicitly detects and trims overthinking in-situ and then optimizes on both original and revised trajectories. This integration of parallel sampling with sequential revision is conceptually clean, computationally practical, and linked to an interpretable partial-advantage view that clarifies why overthinking tokens receive targeted penalties while preserving valid reasoning. 

Method quality is supported by a carefully designed reflection model distilled to 7B for speed, a simple yet effective reflection-density reward, and a refined length reward that avoids incentivizing short but wrong responses. The empirical study is broad for the domain, covering GSM8K, MATH500, Gaokao23, AMC’23, and AIME’24 with two budgets, with ablations isolating the effects of reflection modeling versus reward shaping and training-dynamics plots that explain how the method shortens traces without eroding accuracy. 

Clarity is aided by step-by-step descriptions, concrete prompts, and a readable workflow diagram on page 5; the case studies on page 16 make the truncation–revision mechanism tangible.

### Weaknesses
The reliance on answer-presence detection as the stopping criterion risks truncating useful verification steps when the answer is mentioned early in a speculative way, and while the authors mitigate this with a trained reflection model, there is limited quantitative reporting on its detection precision/recall beyond downstream accuracy and token ratios. 

The study focuses on math word problems with a single distilled 7B base; it remains unclear whether the approach scales to other reasoning domains, like multimodal or non-math reasoning, or to larger pretrained LRMs without distillation artifacts, especially with a reflection model trained for the specific in-distribution task.

The online pipeline introduces extra complexity and additional training time relative to pure parallel sampling, and although the gains seem to justify this, a more detailed wall-clock and latency analysis at inference would strengthen the case.

 Some baselines that aggressively shorten chains show larger efficiency but lower accuracy; while the paper explains the trade-offs, a controlled budget-equalized comparison could more cleanly quantify accuracy at equal compute.

### Questions
See weaknesses

### Soundness
2

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
4

### Summary
This paper proposes **REA-RL (Reflection-Aware Reinforcement Learning)**, a framework for improving reasoning efficiency in Large Reasoning Models (LRMs) without sacrificing accuracy. The authors identify the problem of *overthinking*—excessive reflection that increases inference cost—and introduce two key components: a **reflection model** to detect and remove redundant reasoning in real-time, and a **reflection reward** to preserve necessary reflective behavior. Combined, these techniques reduce inference token usage by **36%** while maintaining or improving performance across several reasoning benchmarks, offering a balanced solution between efficiency and reflection quality.

### Strengths
1. The paper targets an important and practical issue—over-reflection in large reasoning models.  
2. The proposed REA-RL framework is conceptually clear.  
3. The writing is easy to follow.
4. The experiments demonstrate the effectiveness of the proposed methods.

### Weaknesses
1. The paper evaluates only on the MATH domain, and does not include experiments in other reasoning domains (e.g., code, general QA, agentic tasks), which limits the demonstrated generality of the approach.  
2. All experiments are conducted solely on R1-Qwen-7B; including additional model families and scales would strengthen the empirical evidence and show broader applicability.  
3. The method introduces extra computational overhead, requiring double rollouts and an additional reflection-model inference.

### Questions
1. This work shows that the reflection model $M_{reflect}$ contributes to the efficiency–efficacy trade-off. Would it be possible to include a baseline that simply truncates the reasoning sequence when the first reflection token appears and then appends </think> to complete the response from that point?  
2. Compared with methods using only length-based rewards, approaches involving \( M_{\text{reflect}} \) seem to yield slightly lower accuracy. Do the authors have any insights into why this happens?  
3. In line 319, the paper claims that GRPO with accuracy-only rewards does not further improve accuracy, which seems inconsistent with previous findings such as *DeepScaleR* [1]. Could the authors clarify this discrepancy or provide an explanation?  

[1] *DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL.* https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2

### Soundness
2

### Presentation
3

### Contribution
2
