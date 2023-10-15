# Rubrics for scoring the prompt generation (reading ability) 
**Correctness** @ [arxiv_number.yaml]()
- This criterion assesses the level of accuracy in the LLM's answers based on the ground truth.
  - **Score 2**: The answers are entirely correct or equivalent. By “equivalent,” it means the LLM can use any convention as long as it doesn’t involve a physical error.
    - Example: “$c_{l,s}(k)$” (ground truth) vs. “$c_{k,s,l}$” (LLM’s answer).
  - **Score 1**: For partially correct answers.
    - Example: “degrees of freedom are spin and valley index” (ground truth) vs. “degrees of freedom are spin, valley index, and momentum” (LLM’s answer).
  - **Score 0**: Answers that are completely incorrect or missing.
    - Example: “real space” (ground truth) vs. “momentum space” (LLM’s answer).

---

**Placeholder ambiguity** @ [prompt_classification.yaml](prompt_classification.yaml)
- This criterion classifies all task+placeholder combinations that appear in the prompt annotations, essential for filtering during analysis. 
  - **A (Ambiguous)**: Denotes combinations that a human annotator might find ambiguous or challenging to infer.
    - Note: If a combination is marked as ambiguous, it indicates that it may be reasonable for evaluators to leave it blank.
  - **Unlabeled**: Combinations considered reasonable for evaluation without any ambiguity.
    - Note: For certain task+placeholder combinations, comments may be provided for additional clarification.

# Rubrics for scoring the task execution (solving ability)
**0. Whether the result appears explicitly in the paper (`in_paper`)** @ [arxiv_number.yaml]()
- This would allow us to classify at the level of individual prompts whether the calculation has been seen before or not, and we can track accuracy separately in the multiple classes.
  - **2**: The results are explicitly in the paper.
  - **1**: Part of results are explicitly in the paper or the paper provides a combined result (e.g., "A+B"), while the task requires extracting a specific part (e.g., "A").
  - **0**: The results are not in the paper. 

---

**1. Prompt Quality (`prompt_quality`)** @ [arxiv_number.yaml]()
- Assesses the clarity and completeness of the provided prompt.
  - **Score 2**: Unambiguous and complete. Easily understandable and straightforward for a human.
  - **Score 1**: Has missing information or minor typos, but a human can reasonably infer the intent.
  - **Score 0**: Misleading or very unclear. A human would likely not provide a correct answer based on the prompt.

---

**2. Following Instructions (`follow_instructions`)** @ [arxiv_number.yaml]()
- This criterion assesses the LLM's attentiveness to the user's request. It ensures the model remains on track and doesn't produce extraneous or irrelevant information.
  - **Score 2**: The LLM faithfully follows the given prompt and generates relevant content. 
  - **Score 1**: The LLM produces irrelevant information, or misunderstand the instruction. [Example: Prompt states that shift of momentum on "top" layer, while LLM generates the shift on both layers]
  - **Score 0**: The LLM deviates significantly from the prompt, producing mostly or entirely irrelevant content.

---

**3. Physics Logic & Reasoning (`physics_logic`)** @ [arxiv_number.yaml]()
- This criterion evaluates the LLM's physics reasoning capabilities. We want to ensure the solution isn't just formula-driven but adheres to standard physics understanding and principles.
  - **Score 2**: The LLM's logic and methodology are consistent with standard physics reasoning.
  - **Score 1**: The LLM employs some reasoning that is either unnecessary or not straightforward.
  - **Score 0**: The logic does not conform to standard physics reasoning processes.

---

**4. Mathematical Derivation (`math_derivation`)** @ [arxiv_number.yaml]()
- This evaluates the mathematical accuracy and rigor in the LLM's problem-solving process. Physics often requires precise mathematical reasoning, and this criterion ensures just that.
  - **Score 2**: All mathematical derivations and simplifications are accurate.
  - **Score 1**: The LLM makes minor mathematical mistakes, but they account for less than 50% of the overall solution.
  - **Score 0**: Significant mathematical errors are present, accounting for more than 50% of the solution.

---

**5. Final Answer Accuracy (`final_answer_acccuracy`)** @ [arxiv_number.yaml]()
- The correctness of the final answer is crucial. This criterion differentiates between nuanced mistakes and outright incorrect answers.
  - **Score 2**: The LLM provides the correct final answer.
  - **Score 1**: The LLM's final answer is mostly correct, but there are nuances or minor errors, such as incorrect indices.
  - **Score 0**: The final answer is incorrect.