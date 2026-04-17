# Review, Revise, and Learn: Peer-Guided Preference Learning via LLM Self-Correction

- Decision: Reject
- Scores: 4, 4, 2, 8

## Abstract
Preference optimization plays a central role in achieving state-of-the-art performance in large language models (LLMs). However, preference learning requires large-scale, high-quality, or human-annotated datasets, which poses a significant challenge to the continual improvement of LLMs. We introduce PULSE (Peer-gUided Preference Learning via LLM SElf-correction), a collaborative framework of multiple LLM agents for scalable preference learning. PULSE is inspired by the academic peer-review process: an actor LLM first generates an initial response to a query, and critic peer LLMs evaluate and provide feedback on the response. The actor revises or corrects its response based on this feedback, and the critics finally assign scores to the revised response. The scores of the initial and revised outputs are used as preference scores to construct preference data. This process enables autonomous and collective reasoning of LLMs for constructing preference data without human supervision. However, preference data constructed by LLMs may be subject to noise or reward hacking. To mitigate the issue, we first provide a unified view on robust preference learning through the lens of risk minimization, and then propose a framework for robust training on self-correction datasets. Experiments show that PULSE significantly outperforms existing approaches, achieving performance gains up to 47.3\% and 34.6\% on Alpaca LC and Alpaca 2.0, and 23.9\%, 102.8\%, and 12.4\% on a collection of math, coding, and general reasoning tasks, demonstrating its potential to create and sustain scalable LLM ecosystems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes PULSE, which is a collaborative framework of multiple LLMs for preference alignment learning. Specifically, PULSE adopts a pipeline of peer review to construct the alignment training data without human supervision. Then PULSE provide a robust learning objective through the lens of risk minimization. Experiments demonstrate that PULSE outperforms existing baselines.

### Strengths
- The paper is written clearly to understand.
- There are multiple benchmarks selected from difference domains to evaluate the model.
- The experimental results show effectiveness of the proposed method.

### Weaknesses
-  It seems that the peer-review pipeline and the multiple LLMs are only used to construct the alignment training dataset, which is not directly related to the subsequent learning algorithm. There are some other methods that can also generate preference pairs by model without involving human supervision, such as RLAIF [1], SeRA [2] and more. I think that your pipeline integrated multiple LLMs only improves the complexity compared to above methods.
- In addition to the pipeline of preference data generation, this paper proposes a new preference learning algorithm. It is unclear that the performance gains come from the quality of your constructed dataset, or from the proposed preference learning method. 
- In Table 1, PeerReview-DPO using PeerReview underperforms the Snorkel using PairRM. It seems that the data quality judged by PeerReview may not better than that judged by a pretrained reward model. It is unclear about the necessity of such a complex multi-LLM preference judgement pipeline.
- There are some typo errors, such as “Lapaca” in the caption of Table 1.

[1] RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback.  
[2] SeRA: Self-Reviewing and Alignment of LLMs using Implicit Reward Margins.

### Questions
- Have you compared the proposed preference learning algorithm trained on existing preference datasets, such as UltraFeedback or HH-RLHF, instead of your constructed dataset, to verify its superiority?
- Why are the main results in Table 1 and Table 2 only conducted on one model, Mistral-7B-Instruct-v0.2? I think that the experiments presented earlier in the paper are more important, whereas the later experiments are conducted on multiple models. This makes me a bit confused.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents Review–Revise–Learn (RRL), a novel iterative alignment framework for multimodal large language models (MLLMs) that integrates model self-critique and feedback refinement into the preference optimization process.

### Strengths
1. The iterative review–revise–learn structure is an elegant generalization of preference alignment: it moves beyond one-shot comparisons toward process-level refinement.

2.The methodology is technically sound and well-motivated. The algorithmic steps are clearly formalized (Algorithm 1, Section 3.3), and the connection to policy improvement and reward regularization is mathematically justified.

3. RRL tackles one of the key limitations of existing alignment methods—static, one-pass feedback—by proposing a scalable iterative alternative.

### Weaknesses
1. Is the PeerReview Data Generation Necessary? Table 1 shows that PeerReview-DPO (23.75 Alpaca LC) performs worse than baselines like Skywork-DPO (27.45), suggesting the expensive PeerReview data itself is not superior to other AI feedback data. The win for PULSE (29.54) seems to come entirely from the loss $\mathcal{L}\_{\text{PULSE}}$. This is supported by Appendix A.7 (Table 7), which shows $\mathcal{L}\_{\text{PULSE}}$. also beats DPO on the standard UltraFeedback dataset. Could one achieve SOTA results by simply applying$\mathcal{L}\_{\text{PULSE}}$. to an existing, cheaper-to-generate preference dataset like UltraFeedback or Skywork’s?

2. The experiments use a diverse pool of 3 different models as critics (e.g., LLaMA3, Qwen2.5, Gemma2 for the Mistral actor) 22. This diversity is likely crucial for success. The paper lacks ablations on the critic pool's composition. What happens if $K=1$? What if the actor and all critics are the same model?

3. The critic LLMs provide two signals: (1) numerical scores, which are used to create the final $a_w \succ a_l$ label, and (2) structured text feedback, which is used to guide the $a_{init} \to a_{rev}$ revision 25. How important is signal (2)?  Have you run an ablation where the actor LLM is not shown the peer's text feedback and is simply prompted to "improve your previous answer"? This would help isolate the value of the "guidance" in "Peer-gUided" from the "self-correction" aspect.

4. Figure Quality: I noticed that the quality of some figures in the paper is quite low. The text within the plots (legends, axis labels, etc.) is distorted, aliased, and difficult to read. Could the authors please provide high-resolution vector versions of all figures in the final paper to ensure readability and professional presentation?

### Questions
Please see above.

### Soundness
3

### Presentation
3

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
This paper proposes PULSE: a multi-agent LLM preference learning framework. An actor LM first generates an initial response, then a critic LM generates feedback. The actor revises its generation based on the feedback and then the critic assigns scores. The scores of the initial and revised responses (from the actor LM) are used to construct a preference pair. This process allows the model to generate preference pairs without human supervision. 

In addition to their framework, the authors also proposes a variant of DPO loss that is more robust to noise in preference judgement. 

Experiments show that their peer review framework + robust loss function (PULSE) outperforms DPO / iterative DPO with  on common instruction following benchmarks: AlpacaEval, MT Bench.

### Strengths
1. The main idea of this paper - multi-agent collaborating to give and refine a response - is interesting and novel. How to incoperate different language models during inference to get a better response is definitely a field that is of interest of the entire community.

2. The paper is well organized (although the misuse of \citep and \citet should be resolved). The author proposed many different things - the multi-agent framework, the new loss DPO loss function and a hyper-parameter scheduling mechanism. The author backs up such choices with ablation studies. The author conduct experiments on many different benchmarks to validate their approach.

### Weaknesses
1. The paper proposes too many things together, which seems to me a bit lost of focus. In this paper the authors presents a multi-agent "peer review" system to construct preference data pairs, and a new DPO loss that is robust to preference noise. The two parts of this is orthogonal. It would be good if the authors studies one part extensively, rather than two parts together. For example, I don't see any experiments using standard preference datasets with the new PULSE loss.

2. The method is akin to using an LM-judge (since here it is using an LM as a critic) to evaluate response quality. It is hard to disentangle whether the gains come from the LM-judge approach is closer to the evaluation setup or because of the multi-agent setup or because of the new DPO loss. A more natural baseline would be directly use the LM-judge to assess response quality and perform DPO. I am not saying that the author should show such experiments, but having them would certainly be nice for the community to learn **where** the gains come from. I think the benchmark numbers are not that important if the paper can provide insights on what part works and what does not.

3. Looking at the numbers, I believe that the improvements are really limited — the gains on AlpacaEval are < 5 win rates which could be just because of the noise. Moreover, the authors do not disclose their eval protocol (sampling parameters, judge used in Alpaca Eval), which questions the reproducibility of this study. Again, I would be more interested in this paper if the authors can present a more detailed analysis of which part of their proposed method brings the most gains and why instead of trying to get higher benchmark numbers.

I do think that this paper has potential to be impactful, if they can focus on how they designed the multi-agent system. First to see if just doing inference works on improve quality, then go to training .. it might be because of my preferred style of research is to understand 1 thing thoroughly compared to proposing many things together.

### Questions
1. Can the authors show the exact evaluation protocol for evaluation ? (Alpaca Eval decoding params, judge used)

2. Can the authors explain the difference between their method (multi-agent peer review part) with using an LM-judge to evaluate quality?

3. Use \citet{} and \citep{} correctly.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a LLM post-training framework PULSE with two major contributions
* PULSE proposes a peer review and revision approach to synthesize preference data with better diversity.
First, an actor LLM generates initial responses, which are reviewed by multiple critic LLMs.
The actor LLM then revises the responses based on the feedback, and the revised responses are compared with the initial ones by the critic LLMs.
* PULSE proposes a loss function under the risk minimization framework to improve robustness.
The key theoretical tool is Equation 10, which shows that the loss function can be determined by the discriminate function, posterior distribution, and minimum conditional risk.
PULSE then follows the discriminate function and posterior distribution used in the Bradley-Terry model and derives the PULSE loss function by setting the minimum conditional risk as a quadratic function.

Experiments are conducted by training Mistral-7B, Llama-3-8B, Qwen2.5-7B, and Gemma-2-9B on UltraFeedback and evaluating on AlpacaEval, MT-Bench, and academic benchmarks, where PULSE demonstrates superior performance.
Ablation study illustrates the effects of designed components, *i.e.*, peer review and $\beta$-scheduling.

### Strengths
* It is common to synthesize data by revision in engineering.
The proposed method increases the training stability to noise in synthesized data, which is helpful to the community.

* It is interesting to derive the loss function under the risk minimization framework.
I carefully check the derivations of Equations 8 to 19 and found no issue.
Theoretical support enhances the persuasiveness of the proposed method and provides insights for future work.

* The experiment is comprehensive, which is conducted on four LLMs and yields consistent superior performance.
Ablation study is performed to illustrates the effects of designed components.

### Weaknesses
This is a generally high-quality paper and I did not find many flaws.
My main concern is that it seems there is no ablation on the PULSE loss.
Is it possible to achieve better performance by using Skywork RM to synthesizes preference data and PULSE loss to train the model?

### Questions
I see that under the risk minimization framework, PULSE and DPO losses share identical discriminant function and posterior distribution.
Why does using quadratic function as the minimal conditional risk lead to better stability?
Could the authors provide more theoretical analysis on this?
Additionally, can other preference losses be also explained under the risk minimization framework?
For example, are they equivalent to using different minimal conditional risks?
It would be beneficial to use a table for comparison.

### Soundness
3

### Presentation
2

### Contribution
3
