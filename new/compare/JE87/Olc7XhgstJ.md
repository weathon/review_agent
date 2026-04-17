# Review

## Summary
This paper proposes a new method to solve the underthinking problem of LLMs. The method consists of three steps: (1) segment the thinking part of the response at a thought level based on changes in entropy, (2) guide the model to continue writing for each thought without any thoughts switching by reducing the logits of words such as "wait" and "alternatively", (3) construct preference optimization data pairs based on the correctness of the completion, and optimize the model using the STPO algorithm. The paper shows that this method can reduce the output length and improve the accuracy of LLMs on math and code tasks.

## Soundness
2

## Presentation
3

## Contribution
2

## Strengths
1. The paper is well-written and easy to follow.
2. The proposed method is simple and effective in reducing the output length of LLMs and improving their accuracy.

## Weaknesses
1. The proposed method is only evaluated on math and code tasks. It's unclear whether it can also improve the performance of LLMs on other types of tasks, such as reasoning tasks (e.g., commonsense reasoning, Big-bench hard).
2. The proposed method is only compared with two baselines: NoThink and NOWAIT. I suggest the authors compare the proposed method with more baselines, such as [1,2,3,4].
3. The proposed method has several hyperparameters (e.g., the entropy threshold, the number of segmented thoughts) that need to be tuned. It's unclear how to tune these hyperparameters on a new task.

[1] https://arxiv.org/abs/2406.04244

[2] https://arxiv.org/abs/2406.10774

[3] https://arxiv.org/abs/2406.11439

[4] https://arxiv.org/abs/2402.19173

## Questions
1. How to apply the proposed method to LLMs that don't generate intermediate thoughts (e.g., GPT-4, GPT-4o, DeepSeek-R1)?
2. Can the proposed method improve the performance of LLMs on commonsense reasoning tasks and Big-bench hard?
3. How does the proposed method compare with other baselines?
4. How to tune the hyperparameters of the proposed method on a new task?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
4