---
job_id: 939890e5-7389-45cc-b60e-5187f1e04f73
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: ZNAY3ivd62.pdf
paper: GUI-Spotlight: Adaptive Iterative Focus Refinement for Enhanced GUI Visual Grounding
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work is on multimodal LLMs, RL fine-tuning, and GUI visual grounding, which fits ICLR’s core areas (representation learning, RL, vision–language).

## Minimum Quality
Pass ✅.  
All required sections (abstract, introduction, related work, method, experiments, results, conclusion) are present, in English, and the work provides nontrivial methodology plus extensive experiments on standard benchmarks. No obvious fatal methodological or statistical flaws are apparent from the text.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any attempts to manipulate automated reviewing systems or hidden instructions within the paper text.

---

# Expected Review Outcome:

## Summary

The paper proposes GUI-Spotlight, a 7B-scale GUI visual grounding model that performs iterative “spotlight” focusing on screen regions via three explicit tools (crop, extract, find_color) and is trained with a modified GSPO-based reinforcement learning objective. The method uses a three-stage pipeline (SFT on tool-use trajectories, RL on filtered UGround data, then RL on high-res web data with tool-balanced sampling) and a composite reward over answer correctness, crop IoU, extract correctness, color-based focusing, and tool format validity. On ScreenSpot-Pro, UI-Vision, and OSWorld-G, GUI-Spotlight achieves competitive or superior accuracy compared to other 7B models while using only 18.5K curated training samples, and the paper includes a series of ablations on RL variants and reward designs.

## Strengths

1. **Clear, well-motivated problem and setting.**  
   The paper focuses on a very concrete and important bottleneck for GUI agents: reliable pixel-level visual grounding on high-resolution, cluttered interfaces. The motivation is well laid out in Section 1 with clear reference to ScreenSpot-Pro, UI-Vision, and OSWorld-G tasks.

2. **Simple but effective “think-with-image” tool framework.**  
   The tool design (Table 1) is deliberately minimal (crop, extract, find_color) yet well-chosen to support iterative narrowing on UIs. Algorithm 1 and the registry with offsets make the coordinate bookkeeping fully explicit, and the semantics of relative vs absolute coordinates are clear. Figure 1 gives a concrete example of how an instruction like “Click Send button” results in iterative crops and finally a precise click; that visualization nicely supports the claim that the model can use tools to home in on the target.

3. **Nontrivial RL formulation with stabilization tricks, backed by analysis.**  
   The modified GSPO objective in Section 3.2.2, especially the auxiliary $\mathcal{J}'(\theta)$ term that is active on tool-format-correct and result-correct samples, is well specified mathematically. The role of $C_b$ and $M_{b,t}$ in masking tokens and samples is clearly defined, and the stage-wise change of $\lambda$ and sampling scheme is explained. The empirical evidence in Figure 3 (right) directly shows that vanilla GRPO/GSPO collapses after ~300 steps whereas the proposed “tool-filtered positives + CE” scheme maintains and improves performance, supporting the stabilization argument.

4. **Thorough experimental coverage on strong benchmarks and data efficiency.**  
   The paper evaluates on three of the most relevant GUI grounding benchmarks (ScreenSpot-Pro, UI-Vision, OSWorld-G).  
   - In **Table 3**, GUI-Spotlight (init UI-TARS-1.5-7B) reaches 52.8% overall on ScreenSpot-Pro, exceeding V2P-7B (50.6%) and GTA-1-7B (50.1%) while using only 18.5K curated training examples versus millions for those baselines. The comparison between the variant initialized from Qwen2.5-VL-7B and its raw baseline (26.8 → 38.7%) also supports that the RL/tooling approach improves generic MLLMs, not only UI-specialized ones.  
   - On **UI-Vision** (Table 4), the UI-TARS-initialized GUI-Spotlight improves +5.3 points over UI-TARS-1.5-7B (18.1 → 23.4), making it the strongest 7B model reported there.  
   - On **OSWorld-G** (Table 5), GUI-Spotlight (UI-TARS init) gets 62.7%, slightly above UI-TARS-1.5-7B (61.9%) and competitive with 72B baselines, again with very modest extra data.

5. **Useful ablations and “what did not work” insights for RL on GUI grounding.**  
   Section 4 is unusually detailed in reporting RL algorithm variations and reward shaping.  
   - Figure 3 (left) compares several GSPO/GRPO variants such as Clip-Higher, KL removal, uncertain-prompt selection, etc. The bar chart makes it easy to see which modifications help or hurt. For instance, the selective high-uncertainty prompt routing and continuously updated reference policy both degrade accuracy (35.8% and 36.7%), clarifying that common RL tricks do not transfer straightforwardly to this multi-tool setting.  
   - Figure 4 (left) compares binary sparse versus dense answer rewards. Even though the dense reward is intuitively appealing, the graph shows that sparse rewards give slightly better post-convergence accuracy, a nonobvious but valuable design insight.  
   - Figure 4 (right) shows that giving relatively more weight to Extract vs Crop yields a ~10.5 point difference in final ScreenSpot-Pro accuracy, grounding the claim that “simpler tool actions” are easier for the model to exploit.

6. **Data collection and cleaning pipeline is clearly described and seems robust.**  
   Section 3.2.1 and Appendix A.4 give a fairly detailed description of the Selenium-based high-resolution web scraping, candidate element discovery, and strict filtering scheme using Qwen2.5-VL-72B as an auditor. The scoring formulas for bounding box accuracy $S_{BA}$ and consistency IoU are explicit, with thresholds given. This is helpful for reproducibility and for others wishing to build similar high-quality GUI grounding data.

7. **Evidence that multi-step reasoning is actually learned, not just hard-coded.**  
   The ablation in Section 5.4 and **Figure 5** is convincing: using the same base UI-TARS-1.5-7B model, “multi-turn conversational inference” and “repeated single-turn cropping and reclicking” achieve much lower accuracy (~7.6% and 47.6% respectively) than the fully trained GUI-Spotlight (52.8%). The figure makes explicit that the base model essentially has no multi-step “think-with-image” capacity until RL + tools are introduced.

## Weaknesses

1. **Limited architectural novelty; contribution leans heavily on training tricks and tooling rather than model design.**  
   While the iterative-tool spotlighting paradigm is interesting, the paper largely keeps the backbone architecture (UI-TARS / Qwen2.5-VL) unchanged and adds three relatively simple tools plus a modified GSPO loss. The conceptual framing as “think-with-image” is attractive, but the technical advances are essentially: (1) specific hand-designed tools; (2) one auxiliary CE term $\mathcal{J}'(\theta)$ and a bucketed sampling heuristic; (3) a hand-crafted composite reward. There is no new representation, module, or agent architecture beyond prompting. This makes the work feel more like an engineering and tuning study than a more fundamental algorithmic advance, which should be acknowledged more explicitly.

2. **Ambiguous measurement of “training data size” and comparative fairness.**  
   The abstract and multiple sections emphasize that GUI-Spotlight uses only “18.5K training samples” versus millions for baselines (e.g., UGround, V2P, GTA-1), and this is also highlighted in **Table 3** (“Training Data Size” column). However, it is not fully clear what is counted in 18.5K versus the pretraining of the backbones:  
   - Stage 1 relies on trajectories generated by Qwen2.5-VL-72B on UGround, but the text does not make clear whether these 2.5K trajectories are part of the 18.5K tally or additional.  
   - The base models UI-TARS-1.5-7B and Qwen2.5-VL-7B themselves were trained on substantial GUI or general data; Table 3’s comparison between “Training Data Size” entries for GUI-Spotlight and other GUI-specialized models could mislead readers into thinking overall training data is dramatically smaller, whereas much of the capacity is inherited.  
   Without a standardized accounting (e.g., counting only *incremental* task-specific data vs. total corpus used for backbone pretraining) the data-efficiency claims are somewhat overstated.

3. **Evaluation misses some directly relevant baselines and domains.**  
   The experimental section does not consider or even cite several very closely related GUI grounding works listed in the web context and not present in the references:  
   - WinClick (Hui et al., 2025) and WinSpot (Hui et al., 2025) focus specifically on Windows GUI grounding with MLLMs; these are highly relevant for multi-step grounding on desktop environments.  
   - Recent UGround or successor variants that treat GUI grounding in a unified referential segmentation setting (e.g., UGround: Unified GUI Visual Grounding with unrolled transformers) and  
   - GroundingGPT-style modular grounding architectures that attach a frozen LLM to dedicated vision modules.  
   While ScreenSpot-Pro, UI-Vision, and OSWorld-G are strong benchmarks, there is no discussion of how GUI-Spotlight compares conceptually or empirically with such modular grounding architectures or Windows-centric GUI grounding systems. This weakens the positioning of the work in the fast-moving GUI grounding literature.

4. **Reward formulation and weighting design is somewhat ad hoc and task-specific, with limited theoretical or empirical justification.**  
   The composite reward $R=\sum_k \alpha_k r_k$ uses hand-picked weights $(0.30,0.25,0.05,0.20,0.20)$ (Section 3.2.3 and Table 2) and fixed thresholds (e.g., 0.4 IoU for consistency in data filtering, binary format checks). While Section 4.2 provides some ablations on answer reward shaping and Crop vs Extract weight ratios (Figure 4), many design decisions still look empirically tuned without a deeper analysis. For instance:
   - Why is $r_5$ (Format) weight as high as 0.2 relative to Answer (0.3)?  
   - How sensitive is performance to the specific size of the find_color window (200×200) or the patch stride 10 used in Table 1?  
   - Are there observable pathologies where the agent optimizes for $r_2$ or $r_4$ (getting a big IoU or window coverage) but still misses the final click target?  
   These issues matter because the contribution claims rely heavily on RL reward shaping; more systematic exploration or at least qualitative failure analysis would strengthen the case.

5. **Limited qualitative analysis of model behavior and error modes.**  
   Aside from Figure 1’s single illustrative trajectory, the paper provides almost no qualitative examples of successes and failures across different tool types or GUI categories. For example:
   - On ScreenSpot-Pro (Table 3), GUI-Spotlight underperforms UI-Venus-7B in some domains (e.g., CAD where 51.0 vs 51.0 is similar, and Operating System where 46.9 vs 37.2 is better but still far from 72B models).  
   - On UI-Vision (Table 4), performance on the Spatial subset remains low (9.1%) despite improvements in Basic and Functional, suggesting that spatial reasoning remains a weakness.  
   - On OSWorld-G (Table 5), the Qwen-based GUI-Spotlight actually *degrades* layout understanding (41.9 → 40.1) while boosting element recognition and text matching.  
   The paper does not examine what kinds of errors still occur (e.g., mis-clicks on visually similar buttons, confusion among overlapped UI elements, mis-usage of find_color). This limits our understanding of where the approach truly helps and where its limitations lie.

6. **Some aspects of the GSPO modification remain under-specified or could be more rigorously analyzed.**  
   The mathematical description of the objective is mostly clear, but a few details are not fully spelled out:  
   - The clipping term $\mathrm{clip}(s_i(\theta), 1-\varepsilon, 1+\varepsilon)$ is defined, yet there is no discussion of how often clipping is active in practice or how $s_i$ behaves when sequences differ widely in length, with only a sentence remarking that log-prob differences are averaged over $|y_i|$.  
   - In $\mathcal{J}'(\theta)$, the denominator includes $\varepsilon$ “to avoid division by zero”, but in the bucketed sampling of Stage 3, there could be tools that are rare or never used; it is not explained what happens if some $S_t$ is empty or extremely small.  
   - The decision to set $\beta_{\text{KL}}=0$ (Table 7, Algorithmic exploration) is notable, especially given that many PPO/GRPO-style methods rely on nonzero KL to prevent policy drift, but the only justification is an empirical remark that KL was removed, with no analysis of resulting variance or stability beyond Figure 3.  
   These are not fatal flaws, but they leave the RL objective somewhat empirically grounded but not conceptually tight.

7. **The “iterative focus refinement” idea is not directly compared against more powerful tool sets or learned visual modules.**  
   The paper contrasts GUI-Spotlight with two “training-free” iterative baselines (Section 5.4 and Figure 5) that are quite weak: one is essentially multi-turn prompting without RL, the other is repeated local cropping. However, a more pointed question is: how much of the gain is due to explicit tools versus what one could get by simply increasing input resolution, using sliding-window crops without reasoning, or adding a learned attention-based zoom module rather than hand-coded tools? For example, comparing to a baseline that processes multiple fixed-resolution crops around a grid of candidate locations (without explicit tools, just concatenated views) would better isolate whether the tool-use RL is truly necessary. The current baselines make the proposed method look stronger than it might be in a more competitive setting.

8. **Exposition and notation could be tightened in a few technical sections.**  
   Although generally clear, there are a few places where notation or typography could be improved:  
   - In the reward design section (Section 4.2), the dense answer reward formula is split across lines with some odd spacing and typos (“c l o s e n e s s”, “a n s w e r”), which make it slightly hard to parse.  
   - In the bucket construction in Stage 3, the notation $S_t = \{b : \mathrm{tool}(b)=t, \mathrm{correct}(b)=1, \mathrm{format}(b)=1\}$ conflicts slightly with the later description that there is “bucketed uniform sampling across tool types”; more explicit pseudocode or an algorithm box would help.  
   - Figure 2’s y-axis is “Accuracy (%)” while the text describes “evolution of test accuracy over stages”; but the curve dips below the Stage-0 bar, which seems counter-intuitive given that Stage 1 is a warm-up. More textual explanation of why 1 epoch of SFT temporarily *hurts* ScreenSpot-Pro performance (from 39.3 to 12.8) before RL boosts it to 49.6 and 52.8 would help.

Overall, the weaknesses are not fatal, but they do limit how far I can go toward a very high rating; the work feels solid and useful but not at the “must-highlight oral” level.

## Potentially Missing Related Work

1. **Hui et al., “WinClick: GUI Grounding with Multimodal Large Language Models”, 2025.**  
   - Directly focused on GUI grounding using MLLMs in a desktop (Windows) environment.  
   - Should be discussed in Section 2 (GUI grounding) and compared in Section 5, particularly regarding iterative grounding strategies and performance on desktop GUIs.  
   - If benchmarks overlap (e.g., Windows UIs), consider including as a baseline or at least conceptual comparison on how WinClick handles multi-step grounding versus GUI-Spotlight’s tool-based RL.

2. **Hui et al., “WinSpot: GUI Grounding Benchmark with Multimodal Large Language Models”, 2025.**  
   - Introduces a dedicated GUI grounding benchmark for Windows.  
   - Should be mentioned in the Related Work section as an additional benchmark resource and may motivate future evaluation of GUI-Spotlight beyond ScreenSpot-Pro, UI-Vision, and OSWorld-G.  

3. **Qian et al., “UGround: Unified GUI Visual Grounding”, 2025.**  
   - Although some UGround variants are cited, this unified referential segmentation framework using unrolled transformers and dynamic layer selection appears not explicitly referenced.  
   - It is conceptually close, as it also attempts to unify GUI grounding across tasks and may provide alternative architectures that handle dense, cluttered screens.  
   - Should be cited around Section 2’s discussion of GUI grounding models and contrasted with GUI-Spotlight’s tool-centric approach.

4. **Zhang et al., “GroundingGPT: Multi-modal Grounding Architecture”, 2026.**  
   - Proposes a modular grounding architecture that maps language queries onto visual elements via adapters with a frozen LLM.  
   - Relevant for Section 2 and Section 5 as another way to turn a general LLM into a grounded agent, in contrast to GUI-Spotlight’s explicit tool RL fine-tuning.  
   - Comparing its modular approach to GUI-Spotlight’s GSPO-based RL would strengthen the discussion on design space (frozen LLM + adapters vs fine-tuned LLM + tools).

## Questions

1. **Clarification on the “18.5K training samples” figure.**  
   - Does 18.5K include: (a) the 2,561 SFT trajectories from Stage 1, (b) the 12K filtered UGround samples used for RL in Stage 2, and (c) the 4K high-resolution samples in Stage 3? Or only the latter two? Please provide a precise breakdown of how this number is computed.  
   - For fair comparison with Table 3 baselines, could you clarify whether their “Training Data Size” counts only task-specific grounding data or includes any pretraining data? A small table summarizing “incremental grounding data used” vs “backbone pretraining” per method would be helpful.

2. **Behavior and frequency of tool calls across datasets.**  
   - Could you provide statistics on how often each tool (crop, extract, find_color) is used per successful episode, and how this changes across training stages? For example, what is the average number of tool calls per ScreenSpot-Pro query at inference, and what fraction of episodes use find_color at all?  
   - Such statistics would help readers understand whether the model truly leverages the full tool set or mostly falls back to one or two simple patterns.

3. **Failure cases and qualitative behavior.**  
   - It would be very useful to see several examples (per benchmark) where GUI-Spotlight fails, with a brief description of the error mode (e.g., mis-using extract, misinterpreting icons, overshooting cropping).  
   - In particular for UI-Vision spatial queries (Table 4, “Spatial” column), could you show a few typical failures and comment on whether they are due to the backbone’s spatial reasoning limit or suboptimal tool composition?

4. **Alternative baselines without explicit tools.**  
   - Can you add a baseline that processes multiple fixed patches or pyramid crops in a single forward pass, without explicit tool actions and RL? For example, dividing the 4K image into a coarse grid of tiles, encoding them jointly with the text query, and training only with SFT or supervised regression.  
   - If you have preliminary results, please share; if not, a short discussion of expected trade-offs (latency, performance, training difficulty) would help contextualize the need for tools.

5. **Stability of training without the Format reward or with nonzero KL.**  
   - Section 4.1 reports that adding $\mathcal{J}'(\theta)$ prevents collapse. Do you have curves for variants where the Format component $r_5$ is removed, or the weight of $r_5$ is reduced?  
   - Similarly, what happens if $\beta_{\text{KL}}>0$ in the GSPO objective? Given that many RLHF pipelines rely on KL to stabilize against drift, it would be good to understand whether the zero-KL setting is truly necessary here.

6. **Applicability beyond GUI screenshots.**  
   - Do you anticipate GUI-Spotlight’s tool framework and reward shaping to transfer to other high-resolution grounding tasks, such as dense visual navigation or diagram understanding? If you have any anecdotal results or thoughts on generalization to non-GUI imagery, including potential limitations, that would be valuable to add.

Author responses that clarify these points, add missing comparisons, and possibly strengthen the baselines could change my assessment upward.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The methodology (tool-augmented GSPO RL) is generally sound, equations are consistent, and extensive experiments on multiple benchmarks support the central claims, though some design choices (reward weights, KL removal) remain heuristic and under-analyzed.

## Presentation Rating

3: good.  
The paper is overall well written, with clear algorithms, tables, and figures. Some mathematical typography and notational details in the reward section could be improved, and the related work section should incorporate several missing recent works.

## Contribution Rating

3: good.  
The work provides a solid, practically useful advance for GUI visual grounding with multi-step RL and tools, and offers detailed empirical insights. The conceptual and architectural novelty is moderate rather than high, but the combination of tool design, RL stabilization, and data pipeline is still valuable to the community.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  
The paper represents a well-executed and empirically convincing application of RL with explicit tools to high-resolution GUI grounding, with strong results and thoughtful ablations. Its main limitations are modest architectural novelty, somewhat ad hoc reward and algorithm design, and incomplete positioning versus very recent related work. With some tightening of comparisons, clarifications on data accounting, and more qualitative analysis, it would be a solid addition to ICLR.

## Reviewer Confidence

4: confident.  
I am familiar with GUI grounding and RLHF-style training, carefully checked the math in the core objective and rewards, and cross-referenced the experiments with the reported results, though I have not independently reproduced the experiments.