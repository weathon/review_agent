# MetaCaptioner: Towards Generalist Visual Captioning with Open-source Suites

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Generalist visual captioning goes beyond a simple appearance description task, but requires integrating a series of visual cues into a caption and handling various visual domains.  In this task, current open-source models present a large performance gap with commercial ones, which limits various applications such as data synthesis.  To bridge the gap,  this paper proposes CapFlow, a novel multi-agent collaboration workflow.  CapFlow demonstrates for the first time that, by capitalizing on open-source models, it is possible to achieve caption quality on par with GPT-4.1 in various domains with an 89.5\% reduction in costs.   By leveraging CapFlow as the data synthesizer, we produce high-quality visual captions from image and video domains at scale, and obtain a generalist visual captioner via fine-tuning, namely MetaCaptioner. Through extensive experiments, we show that MetaCaptioner not only achieves comparable captioning capabilities with commercial models but also reaches top-tier multimodal performance in the open-source community. We hope CapFlow and MetaCaptioner can benefit future multimodal research by providing a strong and cost-effective visual captioning solution. Our source code and models will be publicly released.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a method for generating a small model which is capable of producing high quality captions for real world image data. This requires a complex synthetic data pipeline called CapFlow. CapFlow uses an MLLM domain router along with domain specific agents to generate initial high quality captions for a large amount of data. This is followed by rejection sampling to remove low quality captions. The data is then used to train MetaCaptioner an 8B parameter model which produces high quality captions. The paper compares with several recent open source MLLMs along with the closed GPT4.1 and shows an improvement over the baseline QWEN model at cheaper cost than GPT 4.1.

### Strengths
- The paper makes a clear improvement over the baseline QWEN even at 8B
- The automatic captioning pipeline is well motivated and carefully designed
- The method could be useful for data generation tasks of its own

### Weaknesses
- Although the paper improved over QWEN the improvement does seem small unless scaling to 72B (Table 2)
- Significant training time (480 + 192 H200 days)

### Questions
I think there is a good contribution here but it is a bit hard to understand exactly what the practical impact of the work here is. On the good side I like the design of the CapFlow pipeline from a systems standpoint. The parts all make sense and automating it reduces human expense. I also like what the paper achieves with an 8B model, it's a good goal. That said it wasn't clear to me from the improvements in Table 2 on 8B how much of a difference the model makes in practice. What do the higher scores mean? How do I interpret them? This is coupled with what I would call an extremely expensive training time of 480 + 192 H200 days. That is not cheap, and it may be misleading for Table 2 to show 2c per 100 samples when the actual model training consumed significant energy cost and GPU purchasing cost. 

**Specific Questions**
1. How do I understand the results in Table 2 in terms of their practical impact?
2. What is the true cost of MetaCaptioner after accounting for training cost?

### Soundness
3

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
4

### Summary
The paper introduces CapFlow, a novel multi-agent framework for generalist visual captioning using open-source models, and presents MetaCaptioner, a fine-tuned captioner trained on high-quality data synthesized by CapFlow. The work demonstrates competitive performance with commercial models (e.g., GPT-4.1) at significantly lower cost.

### Strengths
1.  MetaCaptioner achieves state-of-the-art results among open-source captioners and approaches GPT-4.1 in quality, especially in reasoning-intensive tasks.
2.	CapFlow demonstrates a substantial reduction in inference and data generation costs compared to commercial models.
3.	A well-motivated hierarchical, domain-specialized multi-agent design that directly tackles domain diversity in visual captioning.

### Weaknesses
1. Lack of Direct Human Evaluation. Caption quality is judged by GPT-5; potential model-based evaluation bias is unaddressed.
2. Missing Gemini caption or GPT5 caption comparison. No caption-quality comparison to a closed-source baseline like Gemini, weakening “near-commercial quality” claims.
3. There are hallucinations in reject sampling process. The MLLM (qwen2.5vl-7B) used in the reject-sampling pipeline may exhibit notable hallucinations.
4. It is unclear how specialist and reasoning agents are instantiated and validated (e.g., structural perception, infographic perception, OCR, coder, texture, video perception; general/medical/knowledge/video reasoning). Are these just prompt-specialized variants? How is accuracy ensured relative to visual specialists? Is there cross-agent verification?
5. The workflow appears relatively simple, so tightly binding the approach to a multi-agent framing can feel overstated. A leaner design—e.g., a single controller agent that orchestrates tools and drives self-evolving caption refinement (plan–act–reflect loops)—might achieve similar quality with less complexity and overhead.

### Questions
Please see Weakness part.

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
This paper introduces CapFlow, a multi-agent, hierarchical workflow for generalist visual captioning across nine domains. A domain router selects a domain-specific workflow; functional agents produce evidence; a summary agent aggregates it into a dense caption. Using CapFlow as a data engine, the authors synthesize MetaCaption-4.1M and fine-tune MetaCaptioner-8B. They report caption quality comparable to GPT-4.1 while cutting average inference cost to 10.5%, with further gains when scaling functional agents to 72B.

### Strengths
1. Concrete workflows for nine visual domains with explicit agent roles. 
2. A multi-agent pipeline used to synthesize 4.1M high-quality captions. 
3. Ablations show gains from hierarchical workflows and domain routing.
4. ~89.5% cost reduction vs. GPT-4.1.
5. Implementation details for data and training are provided.

### Weaknesses
1. Evaluation leans heavily on LLM-as-a-judge. Most headline claims rely on model-based scoring. This makes the results vulnerable to prompt/model bias. I would expect at least a small but representative human preference study per domain and a few task-grounded metrics to triangulate quality.
2. Router robustness is unclear. The domain router is main contribution, yet there’s no analysis on mixed-domain inputs.

### Questions
see weakness

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
4

### Summary
This paper introduces CapFlow, a multi-agent workflow for generating visual captions using open-source multimodal large language models (MLLMs). CapFlow employs domain routing and specialized agents to produce multi-domain visual captions, which are filtered via a reject sampling pipeline. The synthesized data is then used to fine-tune MetaCaptioner, a general-purpose captioning model.

### Strengths
* The manuscript is clearly written, with a logical flow that makes the overall pipeline easy to follow. The problem statement is well-motivated and practically relevant, making the work engaging and interesting to read.
* If the dataset and models are publicly released as claimed, they would provide a valuable resource for the open-source multimodal learning community, enabling further research on generalist captioning and multimodal alignment.

### Weaknesses
* Technical novelty is limited. Although the agent composition and hierarchical workflow are carefully designed, the core components are incremental integrations of existing methods rather than fundamentally new algorithmic contributions. The work introduces no novel learning architecture or objective formulation.

* Although the paper claims that MetaCaptioner achieves performance comparable to GPT-4.1, the presented benchmark results do not fully substantiate this claim.

* Qualitative or diagnostic discussion of where prior models fall short and how the proposed method specifically addresses those limitations is needed.

### Questions
* The paper attributes notable performance improvements to the proposed domain routing mechanism. Could the authors provide more analysis or ablations demonstrating its robustness? Specifically, I wonder how sensitive is the system to routing errors or misclassification of domains. Is there a significant drop in caption quality when the routing assigns incorrect workflows?

* While Table 5 shows that captions generated by MetaCaptioner substantially enhance downstream reasoning when used as inputs to external LLMs, Table 6 indicates relatively modest gains in its direct multimodal evaluation. Does this imply that MetaCaptioner mainly improves as a captioning front-end rather than as a genuinely stronger multimodal learner? If so, how does this affect its practical utility in end-to-end multimodal understanding tasks?

### Soundness
3

### Presentation
3

### Contribution
2
