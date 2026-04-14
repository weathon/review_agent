# Certifiably Robust RAG against Retrieval Corruption Attacks

- Decision: Reject
- Scores: 6, 3, 5

## Abstract
Retrieval-augmented generation (RAG) has been shown vulnerable to retrieval corruption attacks: an attacker can inject malicious passages into retrieval results to induce inaccurate responses. In this paper, we propose RobustRAG as the first defense framework against retrieval corruption attacks. The key insight of RobustRAG is an isolate-then-aggregate strategy: we isolate passages into disjoint groups, generate LLM responses based on the concatenated passages from each isolated group, and then securely aggregate these responses for a robust output. To instantiate RobustRAG, we design keyword-based and decoding-based algorithms for securely aggregating unstructured text responses. Notably, RobustRAG can achieve certifiable robustness: we can formally prove and certify that, for certain queries, RobustRAG can always return accurate responses, even when an adaptive attacker has full knowledge of our defense and can arbitrarily inject a small number of malicious passages. We evaluate RobustRAG on open-domain QA and long-form text generation datasets and demonstrate its effectiveness and generalizability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the author propose a novel framework called RobustRAG to defend retrieval corruption attack. This method can ensure the accuracy of the response from Retrieval-Augmented Generation. RobustRAG isolates passages, then gets a response for each of these passages and aggregates them by secure keyword and secure decoding aggregation. The authors conduct experiments on multiple LLMs and datasets and prove that RobustRAG has better certifiable accuracy than vanilla RAG.

### Strengths
### 1.Good writing
The writing logic of this paper is clear and very easy for readers to understand. The text is concise and accurate in introducing the preliminary and method. In addition, the paper has no grammatical errors and typos, and is highly complete.


### 2. Comprehensive discussions
The discussions in the appendix of this paper answered many of my questions and are very comprehensive.

### Weaknesses
### 1. Insufficient introduction to experiments against attack
The paper claims that their approach can defend against retrieval corruption attacks, and the experiment in section 5.3 simulates RobustRAG against attacks. Even including the appendix, I suggest that the authors add the settings of the attack experiments in the appendix.

### 2. Lacks comparison experiments
How much impact will RobustRAG have on performance compared to RAG without any attack risk? Since it is necessary to prevent retrieval corruption attacks, it is also necessary to apply RobustRAG all the time. But for benign setups, will RobustRAG affect performance? I suggest the authors add this experiment.

### Questions
In line 134-139, the authors mention that RobustRAG can achieve robustness under the condition: "when k' is smaller than the number of relevant benign passages; when the corrupted passages outnumber relevant passages." I can understand the first condition, but for the second condition "when the corrupted passages outnumber relevant passages", why can corrupted passages outnumber relevant passages? I think the worst case should be equal, not outnumber.
In line 141-142, "the attacker can only manipulate a fraction of retrieved passages", how much does a fraction refer to specifically?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper studies the safety and security issues associated with using RAG by proposing a new defense framework, RobustRAG, against retrieval corruption attacks. The main idea of the proposed method is majority vote, where the LLM generates responses based on each of the retrieved documents, and the final output is the majority vote result. The proposed RobustRAG is evaluated on open-domain QA and long-form text generation datasets, demonstrating its effectiveness and generalizability.

### Strengths
- The topic of the safety of RAG is trending and important.
- The paper is well presented, and the main points are easy to follow.
- The experimental results look OK.

### Weaknesses
There is one major weakness and several minor weaknesses.

**Major:**
- First, the inference cost of the proposed framework will significantly increase compared to the vanilla RAG. Given a query, the proposed method requires generating multiple responses to determine the final majority vote. If the system employing the proposed method cannot perform parallel generation, the latency in obtaining the final answer will also increase drastically. I believe this limitation will significantly impact the real-world use of the proposed method.

**Minor:**
- The overall idea of the work is not novel to me. Essentially, it is a randomized smoothing/majority vote technique applied in a new context, RAG.
- Given the majority vote nature, the proposed scenario cannot handle cases where the number of poisoned responses exceeds the number of clean ones. I acknowledge that this limitation is not only associated with this particular work but also with methods that use the majority vote concept in general. However, I feel that in RAG, there should be better alternatives to address this security risk, such as controlling access to the RAG database.

### Questions
Please see my comments in the Weakness section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces a framework to defend RAG systems against retrieval corruption attacks by isolating passages into groups and securely aggregating responses to maintain robustness.

### Strengths
1. It offers formal guarantees that responses remain accurate despite partial corruption of retrieved passages.

2. RobustRAG’s keyword and decoding aggregation methods securely handle unstructured text, making responses more resilient to manipulation​

### Weaknesses
1. Limited Practicality of Retrieval Corruption Attacks: The assumption that attackers can inject malicious passages into a retrieval corpus may not align with real-world scenarios, especially when the documents are private or secured, limiting practical applicability of the defense.

2. The paper lacks comparisons with other established defense methods.

3.Time Cost: RobustRAG’s isolate-then-aggregate strategy adds computational overhead and the paper should report the increased time cost in expertment.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
2
