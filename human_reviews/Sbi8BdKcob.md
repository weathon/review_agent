# From Indeterminacy to Determinacy: Augmenting Logical Reasoning Capabilities with Large Language Models

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 3

## Abstract
Recent advances in large language models (LLMs) have revolutionized the landscape of reasoning tasks. To enhance the capabilities of LLMs to emulate human reasoning, many prior works have focused on modeling intermediate reasoning steps using specific thought structures like chains, trees, or graphs. However, LLM-based reasoning continues to encounter challenges in three key aspects: 1) Selecting appropriate reasoning structures for various tasks; 2) Sufficiently and efficiently exploiting known conditions to deduce new insights; 3) Considering the impact of historical reasoning experience on future reasoning steps. To address these challenges, we propose DetermLR, a novel reasoning framework that formulates the reasoning process as a transformational journey from indeterminate premises to determinate ones. This process is marked by the incremental accumulation of determinate premises, making the conclusion progressively closer to clarity. DetermLR includes three essential components: 1) Premise identification: We systematically categorize premises into two distinct types: determinate and indeterminate. This empowers LLMs to flexibly customize reasoning structures to match the specific task complexities. 2) Premise prioritization and exploration: We leverage quantitative measurements to assess the relevance of each premise to the target, prioritizing more relevant premises for exploring new insights. 3) Iterative process with reasoning memory: We introduce a reasoning memory module to automate storage and extraction of available premises and reasoning paths, preserving historical reasoning details for more accurate premise prioritization and exploration during iterative reasoning. Comprehensive experimental results demonstrate that DetermLR outperforms all baselines on four challenging logical reasoning tasks: LogiQA, ProofWriter, FOLIO, and LogicalDeduction. Compared to previous multi-step reasoning methods, DetermLR can achieve better reasoning performance while requiring fewer visited states, highlighting its superior efficiency and effectiveness in tackling logical reasoning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents DetermLR, a CoT-style prompting strategy that elicits stronger reasoning capabilities from LLMs. Specifically, DetermLR iteratively identifies the most promising premises, prioritises and "executes" them, and then stores useful premises in a memory. 

Experiments are performed on 4 complex logical reasoning datasets, and DetermLR outperforms the 5 compared baselines, sometimes by a large margin.

### Strengths
* CoT-style reasoning is an active and important research area for LLMs as they allow strong reasoning capabilities to be elicited. 

* Logical reasoning is an important problem that LLMs are traditionally not strong at. Further investigation in this area is certainly welcome. 

* The proposed method achieves good performance on the 5 challenging datasets, outperforming the compared CoT-style prompting strategies.

### Weaknesses
* The proposed method is quite simple. Thus, the technical contribution is light. For instance, the "systematic premise identification module" described in Sec. 3.1 is really quite simple, and I don't know whether I'd call it "systematic". 

Besides, it closely follows the Cumulative Reasoning (Zhang et al., 2023) technique, with the addition of a memory. Thus, the novelty is limited. 

* Some important details have been omitted in the paper, making it harder to understand the technical contributions of the paper. I'll detail it below.

### Questions
* In Eq. (1), (2) and (3), what are the definitions of \texttt{relevance}, \texttt{supplement} and \texttt{verify}? If you follow Cumulative Reasoning (CR), are all these functions realised by the LLM?

* In Sec. 3.3, what exactly is the memory? 

* What is the definition of "state" in this paper? Is it the number of "invoked" premises? 

* The results on CR on FOLIO in your paper are different from that in the original paper (on GPT-4), which is much higher (87.45 vs 69.11). Why such a large discrepancy?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces a new reasoning framework, DetermLR, aimed at enhancing the logical reasoning capabilities of large language models. The study addresses challenges of LLMs facing in emulating human-like reasoning, including selecting appropriate reasoning structures, efficiently using known information, and incorporating past reasoning into future decisions. DetermLR use premise identification, premise prioritization and exploration, and an iterative process with reasoning memory. Experimental results on logical reasoning tasks show that DetermLR outperforms baselines in terms of reasoning performance and efficiency.

### Strengths
- This paper is well-motivated and proposes a novel framework that tackles the challenges in emulating human-like logical reasoning.
- The method incorporates a prioritized strategy to direct the reasoning process; history reasoning information including valid and invalid intermediate results to continue reasoning, which are key components for effective reasoning.
- The experimental results demonstrate the effectiveness and efficiency of the proposed framework compared to baseline methods.

### Weaknesses
1. The paper lacks much necessary information about framework. 
- How do you score relevance and supplement? A transparent LM or directly instruct GPT-4? 
- From Fig.1 first step, the authors seem to filter out determinate premises by words matching (eg, the sentence including Gary). However, in some cases, the relationship between premises and conclusion is only logic-level. eg., If Erin is round, then Gary is quiet.
- How do you exploit history reasoning paths information? 


2. [1] paper use a similar idea, alternating between premises selection and inference, despite of lackness of history reasoning paths. But the authors do not emphasis the point in details. In facts, I think those failure cases can help reduce search space.

[1] Creswell, Antonia, Murray Shanahan, and Irina Higgins. "Selection-inference: Exploiting large language models for interpretable logical reasoning." arXiv preprint arXiv:2205.09712 (2022).

### Questions
See weakness above. If the authors can perfectly solve my issues, I will consider improving my score.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a new prompting technique for logic reasoning, DetermLR, which first classifies the given premises into determinate and indeterminate, and then sorts them by priority. Then DetermLR will first explore the premises with high priority and store the new conclusions into the memory for future reference. The proposed method shows superior results on LogiQA, ProofWriter, FOLIO, and LogicalDeduction, over multiple baselines.

### Strengths
1. The proposed method seems to be quite effective on four logic reasoning datasets, compared to multiple baselines.
2. The paper is mostly clear and well-written.

### Weaknesses
1. It is not clear how each module in the proposed framework is implemented. Such information is completely missing in the method section while judging from the experiment section, it seems that all of them are implemented by prompting GPT4. However, it is still unclear how the scorers are implemented and what is the threshold $\theta$ for supplementary premises filtering.
2. I'm not quite sure why the proposed method can select a reasoning structure. The main technique of the proposed method seems to be classifying the given premises into determinate and indeterminate, and sorting the premises by priority. It seems to adopt a mostly linear reasoning structure with memory reference.

I'm willing to increase my score if my concerns are properly addressed.

### Questions
1. How is the time/compute efficiency of the proposed method compared to the baselines? I understand that the proposed method visited fewer states during inference, but I'm not quite sure if the compute/number of GPT4 prompting for each state visiting is the same as the baselines.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
They propose an approach for improving the logical reasoning of LLMs. Their approach is to structure the prompt with three main components including their so-called, premise identification, premise prioritization, and iterative process with reasoning memory.  They conduct experiments on four logical reasoning datasets using their prompting strategy. Their approach outperforms other recent strategies for prompting LLMs such as chain-of-though and some other variations.

### Strengths
-The authors propose a new strategy for structuring the prompt for LLM to make them perform logical reasoning with a higher accuracy.
-The experiments show the effectiveness of the proposed approach compared to the existing structured prompting strategies.

### Weaknesses
-The terminology, notations, and in general the explanation of the proposed approach was not very clear to me. 
-The results focused on the selected subsets of datasets -selected by authors. No comparison with other results on these datasets [outside this work] was made. Or this was not made explicit at least in the paper as far as I understood.   
-The results were reported only on GPT4.  

See some details in the Questions section.

### Questions
-From the provided examples, it was not made clear to me why the term indeterminacy was chosen for some parts of the information.
-in section 3.1., the authors explained the premise identification, and then only in section 3.2 they started introducing formal notations.
-The formalization and notation are somewhat superficial and not really used to help understanding. 
-The flow of information and how these modules exactly work is not clear, are you expecting the LLM to do these steps with a few shots of in-context learning? For example, how the model was asked to identify the premises in the first step? These are just very hard to read from the current presentation of the paper. 
--Not clear what the authors mean by verification check? how the verification is performed, and what is the kind of computation used here for verification?
--What do you mean by states and visited states? I did not see this defined in the paper.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
