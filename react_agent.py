from pathlib import Path
from shutil import copyfile, rmtree

import os
import re
import subprocess
import traceback
from llm_engine import LLMEngine


SYSTEM_PROMPT = """You are an expert computational biologist assistant. Your goal is to help scientists test hypotheses using Python code.
You will be given:
1. A scientific hypothesis.
2. Information about a dataset (path, structure, previews).
3. Optionally, relevant domain knowledge.

Your tasks are:
1. Write a *complete* and *executable* Python script to analyze the provided data to test the given hypothesis.
   - The script should load the data, perform necessary analysis (e.g., statistical tests, data visualization generation, calculations), and save key results or plots needed to evaluate the hypothesis.
   - Ensure the script saves outputs to predictable locations (e.g., derived from input paths or specified output names).
   - Wrap the Python code in a single ```python ... ``` block.
2. If the script fails during execution, debug it based on the error message provided and generate a corrected, complete script.
3. After the script executes successfully, you may be asked to interpret the results and state a conclusion regarding the hypothesis based on the generated output.

Please adhere to these guidelines:
- Write robust, well-commented code.
- Use standard libraries (like pandas, numpy, scipy, matplotlib, seaborn) that can be installed via pip.
- Do NOT use interactive commands (like `!pip install` or `input()`).
- Ensure all file paths used in the code are correct based on the provided dataset information and expected output locations.
- Be concise in your explanations outside the code block.
"""

REQUEST_PROMPT = """Please analyze the data to test the following hypothesis:

**Hypothesis:** {hypothesis}

**Task Instructions:** {task_inst}

{domain_knowledge_prompt}
{data_info_prompt}

Generate the Python code to perform this analysis.
"""

SELF_DEBUG_PROMPT = """The Python script you previously generated resulted in an error during execution.

**Error Message:**
```
{error_message}
```

Please analyze the error, fix the script, and provide the complete, corrected Python code below. Ensure the corrected script addresses the issue and can run successfully to test the original hypothesis.
"""

CONCLUSION_PROMPT = """The Python script for analyzing the data and testing the hypothesis has executed successfully.

**Original Hypothesis:** {hypothesis}

**Analysis Output Summary/Preview:**
```
{output_preview}
```
*(Note: This might be a file path, contents of a results file, or a description of generated plots.)*

Based on the original hypothesis and the results obtained from the script execution, please provide a concise conclusion regarding the validity of the hypothesis. Explain your reasoning based on the output.
"""

DATA_INFO_PROMPT_TEMPLATE = """
**Dataset Information:**
- You can access the dataset at the base path: `{dataset_path}`
- Dataset Directory Structure:
```
{dataset_folder_tree}
```
- Dataset File Previews:
{dataset_preview}
"""

DOMAIN_KNOWLEDGE_PROMPT_TEMPLATE = """
**Relevant Domain Knowledge:**
{domain_knowledge}
"""

class BioAgent():
    def __init__(self, llm_engine_name, context_cutoff=30000, use_knowledge=False, use_self_debug=True, conda_env_name="bioagent-eval"):
        self.llm_engine = LLMEngine(llm_engine_name)
        self.llm_engine_name = llm_engine_name 
        self.context_cutoff = context_cutoff
        self.use_knowledge = use_knowledge
        self.use_self_debug = use_self_debug
        self.conda_env_name = conda_env_name

        self.sys_msg = ""
        self.history = []
        self.first_user_msg_content = "" 


    def get_initial_prompts(self, task):
        """Constructs the system message and the first user message."""
        self.sys_msg = SYSTEM_PROMPT 
        domain_knowledge_prompt = ""
        if self.use_knowledge and task.get("domain_knowledge"):
            domain_knowledge_prompt = DOMAIN_KNOWLEDGE_PROMPT_TEMPLATE.format(
                domain_knowledge=task["domain_knowledge"]
            )

        data_info_prompt = DATA_INFO_PROMPT_TEMPLATE.format(
            dataset_path=task.get('dataset_path', 'N/A'),
            dataset_folder_tree=task.get('dataset_folder_tree', 'N/A'),
            dataset_preview=task.get("dataset_preview", 'N/A')
        )

        
        self.first_user_msg_content = REQUEST_PROMPT.format(
            hypothesis=task.get("hypothesis", "No hypothesis provided."),
            task_inst=task.get("task_inst", "Analyze the data provided."), 
            domain_knowledge_prompt=domain_knowledge_prompt,
            data_info_prompt=data_info_prompt
        )

        return self.sys_msg, self.first_user_msg_content

    def write_program(self, assistant_output, out_fname):
        """Extracts python code from LLM response and writes to file. Returns True on success."""
        Path(out_fname).parent.mkdir(parents=True, exist_ok=True)

        match = re.search(r"```python(.*?)```", assistant_output, re.DOTALL)
        if match:
            code_content = match.group(1).strip()
            if not code_content:
                print("Warning: LLM generated an empty Python code block.")
                return False 
            try:
                with open(out_fname, "w", encoding="utf-8") as f:
                    f.write(code_content)
                return True 
            except Exception as e:
                print(f"Error writing program to {out_fname}: {e}")
                return False 
        else:
            print("Warning: Could not find Python code block (```python...```) in LLM response.")
            return False 

    def install(self, out_fname):
        """Installs dependencies for the generated program. Returns (success, error_message)."""
        err_msg = ""
        success = False
        eval_dir = Path("program_to_eval/") # Define path for evaluation

        try:
            # 1. Setup eval directory
            if eval_dir.exists():
                rmtree(eval_dir)
            os.makedirs(eval_dir)
            script_basename = Path(out_fname).name
            eval_script_path = eval_dir / script_basename
            copyfile(out_fname, eval_script_path)

            # 2. Detect dependencies (pipreqs)
            # Ensure pipreqs runs in the context of the target env's python if possible,
            # although pipreqs often works globally. Here we run it directly.
            reqs_in_path = Path("requirements.in") # Place requirements.in in cwd
            pipreqs_cmd = [
                "pipreqs", str(eval_dir), f"--savepath={str(reqs_in_path)}", "--mode=no-pin", "--force"
            ]
            print(f"Running: {' '.join(pipreqs_cmd)}")
            exec_res = subprocess.run(pipreqs_cmd, capture_output=True, text=True)

            if exec_res.returncode != 0:
                err_msg = f"pipreqs failed (exit code {exec_res.returncode}).\nStderr: {exec_res.stderr}\nStdout: {exec_res.stdout}\nPlease ensure imports are standard and detectable."
                return False, err_msg
            if not reqs_in_path.exists() or reqs_in_path.stat().st_size == 0:
                print("Warning: requirements.in is empty or missing after pipreqs run. Assuming no external dependencies.")
                # Create empty file if needed for pip-compile/sync
                reqs_in_path.touch()


            # 3. Resolve dependencies (pip-compile)
            reqs_out_path = Path("eval_requirements.txt")
            # Base command
            pip_compile_cmd_base = [
                "conda", "run", "-n", self.conda_env_name, "--no-capture-output", # Show output for debugging
                "pip-compile", str(reqs_in_path),
                "--upgrade-package", "numpy<2.0",
                "--output-file", str(reqs_out_path)
            ]

            # Try legacy resolver first
            print(f"Running pip-compile with legacy resolver...")
            exec_res_legacy = subprocess.run(
                pip_compile_cmd_base + ["--resolver=legacy"], capture_output=True, text=True
            )

            if exec_res_legacy.returncode != 0:
                print('Legacy resolver failed. Trying backtracking resolver...')
                print(f"Stderr (Legacy): {exec_res_legacy.stderr}")
                # Try default (backtracking) resolver
                exec_res_backtrack = subprocess.run(
                    pip_compile_cmd_base, capture_output=True, text=True
                )
                if exec_res_backtrack.returncode != 0:
                    err_msg = f"pip-compile failed with both legacy and backtracking resolvers (exit code {exec_res_backtrack.returncode}).\nStderr (Backtrack): {exec_res_backtrack.stderr}\nPlease check for package conflicts."
                    return False, err_msg
            else:
                print("Legacy resolver succeeded.")


            # 4. Install dependencies (pip-sync)
            pip_sync_cmd = [
                "conda", "run", "-n", self.conda_env_name, "--no-capture-output",
                "pip-sync", str(reqs_out_path)
            ]
            print(f"Running: {' '.join(pip_sync_cmd)}")
            exec_res = subprocess.run(pip_sync_cmd, capture_output=True, text=True)

            if exec_res.returncode != 0:
                err_msg = f"pip-sync failed (exit code {exec_res.returncode}).\nStderr: {exec_res.stderr}\nStdout: {exec_res.stdout}\nFailed to synchronize environment."
                return False, err_msg

            print("Installation successful.")
            success = True

        except Exception as e:
            err_msg = f"An unexpected error occurred during installation setup: {traceback.format_exc()}"
            return False, err_msg
        
        finally:
            if reqs_in_path.exists(): 
                os.remove(reqs_in_path)
            if reqs_out_path.exists(): 
                os.remove(reqs_out_path)
            
        return success, err_msg

    def step(self, out_fname, output_fname, hypothesis):
        """Installs, runs the script, and handles errors. Returns (needs_debug, status_message)."""
        install_ok, install_msg = self.install(out_fname)

        if not install_ok:
            trimmed_msg = ' '.join(install_msg.split()[:1000])
            start_msg = "The code you generated encountered issues during the installation of Python packages."
            return True, start_msg + trimmed_msg # Needs debug, return error message

        script_basename = Path(out_fname).name
        run_script_path = Path("program_to_eval") / script_basename # Path inside conda run

        run_cmd = ["conda", "run", "-n", self.conda_env_name, "python", str(run_script_path)]
        print(f"Running: {' '.join(run_cmd)}")
        
        err_msg = ""
        special_err = False
        exec_success = False

        try:
            exec_res = subprocess.run(run_cmd, capture_output=True, text=True, timeout=900) 
            if exec_res.returncode == 0:
                if output_fname and not Path(output_fname).exists():
                    special_err = True
                    err_msg = f"Script finished successfully (exit code 0), but the expected output file '{output_fname}' was not found. Please ensure the script saves the output to the correct path."
                else:
                    exec_success = True 
                    print(f"Script execution successful. Output file '{output_fname}' checked (if specified).")
            else:
                special_err = True
                err_msg = f"Script execution failed (exit code {exec_res.returncode}).\nStderr:\n{exec_res.stderr}\nStdout:\n{exec_res.stdout}"

        except subprocess.TimeoutExpired:
            special_err = True
            err_msg = "Script execution timed out after 900 seconds. Please optimize the code for efficiency or simplify the analysis."
        except Exception as e:
            special_err = True
            err_msg = f"An unexpected error occurred during script execution: {traceback.format_exc()}"

        if exec_success:
            return False, "Execution successful." 
        else:
            # An error occurred (install, timeout, runtime, missing output)
            trimmed_msg = ' '.join(err_msg.split()[:1000])
            return True, trimmed_msg 


    def generate_conclusion(self, hypothesis, output_fname, task):
        """Generates a conclusion based on the script output."""
        print("Generating conclusion...")
        output_preview = f"Analysis results are expected in: {output_fname}"
        try:
            output_path = Path(output_fname)
            if output_path.exists():
                # Simple preview: first few lines for text files, existence for others
                if output_path.suffix in ['.csv', '.txt', '.log']:
                    with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
                        preview_lines = [next(f) for _ in range(10)]
                    output_preview = f"Results saved to '{output_fname}'. Preview:\n{''.join(preview_lines)}..."
                else:
                    output_preview = f"Results saved to '{output_fname}' (binary or other format)."
            else:
                output_preview = f"Expected output file '{output_fname}' not found after execution."

        except Exception as e:
            output_preview += f"\n(Could not read preview: {e})"

        conclusion_request_content = CONCLUSION_PROMPT.format(
            hypothesis=hypothesis,
            output_preview=output_preview
        )
        
        # Build context for conclusion: System + First User Request + Last Code + Conclusion Request
        conclusion_context = [
            {'role': 'system', 'content': self.sys_msg},
            {'role': 'user', 'content': self.first_user_msg_content},
            self.history[-1], # Last assistant message (the successful code)
            {'role': 'user', 'content': conclusion_request_content}
        ]

        assistant_conclusion = self.llm_engine.respond(conclusion_context, temperature=0.3, top_p=0.95) 
        
        self.history.append({'role': 'user', 'content': conclusion_request_content})
        self.history.append({'role': 'assistant', 'content': assistant_conclusion})

        return assistant_conclusion

    def solve_task(self, task, out_fname):
        """Main function to solve a task: generate code, debug, and conclude."""
        self.history = []
        conclusion = "Not generated."

        # 1. Get Initial Prompts
        self.sys_msg, self.first_user_msg_content = self.get_initial_prompts(task)

        # 2. Initial Code Generation
        initial_user_input = [
            {'role': 'system', 'content': self.sys_msg},
            {'role': 'user', 'content': self.first_user_msg_content}
        ]
        print("Generating initial code...")
        assistant_output = self.llm_engine.respond(initial_user_input, temperature=0.2, top_p=0.95)

        # Write initial program
        write_ok = self.write_program(assistant_output, out_fname)
        if not write_ok:
            self.history = initial_user_input + [{'role': 'assistant', 'content': assistant_output}]
            return {"history": self.history, "conclusion": "Code extraction failed.", "success": False}

        self.history = initial_user_input + [{'role': 'assistant', 'content': assistant_output}]
        last_code = assistant_output # Keep track of the code for comparison

        # 3. Execution and Self-Debugging Loop
        max_debug_attempts = 5 # Limit debugging attempts
        execution_successful = False
        for attempt in range(max_debug_attempts + 1): # +1 for initial attempt run
            print(f"\n--- Attempt {attempt + 1}/{max_debug_attempts + 1} ---")
            needs_debug, status_message = self.step(out_fname, task.get("output_fname"), task.get("hypothesis"))

            if not needs_debug:
                print(f"Execution successful on attempt {attempt + 1}.")
                execution_successful = True
                break # Exit loop on success

            # Execution failed, prepare for debugging (if enabled and attempts remain)
            print(f"Execution failed: {status_message}")
            if not self.use_self_debug or attempt >= max_debug_attempts:
                print("Debugging disabled or max attempts reached. Stopping.")
                self.history.append({'role': 'user', 'content': f"Execution failed on final attempt: {status_message}"})
                break # Exit loop

            print("Requesting debug from LLM...")
            # Formulate debug request
            debug_request_content = SELF_DEBUG_PROMPT.format(error_message=status_message)
            debug_input = [
                {'role': 'system', 'content': self.sys_msg},
                {'role': 'user', 'content': self.first_user_msg_content},
                self.history[-1], # Previous assistant code that failed
                {'role': 'user', 'content': debug_request_content}
            ]

            assistant_output = self.llm_engine.respond(debug_input, temperature=0.2, top_p=0.95)

            write_ok = self.write_program(assistant_output, out_fname)
            if not write_ok:
                print("Failed to write corrected program. Stopping debug loop.")
                self.history += [
                    {'role': 'user', 'content': debug_request_content},
                    {'role': 'assistant', 'content': assistant_output + "\n\n[Agent Error: Failed to save corrected code]"}
                ]
                break

            # Add debug interaction to history
            self.history += [
                {'role': 'user', 'content': debug_request_content},
                {'role': 'assistant', 'content': assistant_output}
            ]

            # Early stopping check: If LLM provides the exact same code again, stop.
            if assistant_output == last_code:
                print("LLM returned the same code after error. Stopping debug loop.")
                break
            last_code = assistant_output


        # 4. Generate Conclusion if execution was successful
        if execution_successful:
            try:
                conclusion = self.generate_conclusion(task.get("hypothesis"), task.get("output_fname"), task)
            except Exception as e:
                conclusion = f"Failed to generate conclusion: {traceback.format_exc()}"
        else:
            conclusion = "Conclusion not generated because script execution failed."

        # 5. Return results
        return {"history": self.history, "conclusion": conclusion, "success": execution_successful}



if __name__ == "__main__":
    
    Path("pred_programs").mkdir(exist_ok=True)
    Path("pred_results").mkdir(exist_ok=True)
    Path("program_to_eval").mkdir(exist_ok=True) # For install step

    agent = BioAgent(
        llm_engine_name="gpt-4o-mini", 
        context_cutoff=28000, 
        use_knowledge=False,  
        use_self_debug=True, 
        conda_env_name="bioagent-eval" 
                                    
    )

    example_task = {
        "hypothesis": "Gene X expression level is significantly higher in tumor samples compared to normal samples.",
        "task_inst": "Perform a t-test to compare Gene X expression between tumor and normal groups using the provided dataset. Save the p-value and t-statistic to a CSV file.",
        "dataset_path": "data/expression_data", 
        "dataset_folder_tree": """
data/expression_data/
└── gene_expression.csv
""",
        "dataset_preview": """
Preview of data/expression_data/gene_expression.csv:
```csv
SampleID,Group,GeneX_Expression,GeneY_Expression
Sample001,Tumor,10.5,5.2
Sample002,Normal,5.1,6.1
Sample003,Tumor,12.1,4.8
Sample004,Normal,4.8,5.9
... (more rows) ...
```
""",
        "domain_knowledge": "Student's t-test is appropriate for comparing means between two independent groups assuming approximate normality and equal variances (or use Welch's t-test if variances are unequal). A p-value < 0.05 is typically considered statistically significant.", # Optional
        "output_fname": "pred_results/geneX_ttest_results.csv" # Expected output file path
    }

    output_script_filename = "pred_programs/analyze_geneX_expression.py"

    result = agent.solve_task(example_task, out_fname=output_script_filename)

    print("\n--- Agent Task Result ---")
    print(f"Execution Successful: {result['success']}")
    print(f"\nConclusion:\n{result['conclusion']}")
    print("\n--- Full Conversation History ---")
    for msg in result['history']:
        print(f"[{msg['role'].upper()}]\n{msg['content']}\n---")

