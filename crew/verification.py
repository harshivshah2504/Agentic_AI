from backtesting_agent.extract_python_code import extract_python_code
from backtesting_agent.openai import get_llm_response
from runner import run_code
import contextlib
import io
import os
import re
import pickle
import time

def checker(python_code, ticker, start_date, MAX_RETRIES=10):
    attempt = 0
    strategy_id = 1
    stdout_output = ""

    # Ensure temp directory exists
    temp_dir = "/home/R4/Harshiv/temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Unique filenames per run
    timestamp = int(time.time())
    bt_file = os.path.join(temp_dir, f"temp_bt_{timestamp}.html")
    stats_file = os.path.join(temp_dir, f"temp_stats_{timestamp}.pkl")

    # Main block for execution
    main_block = f'''
if __name__ == '__main__':
    start_date = "{start_date}"
    try:
        bt = MultiBacktest(Strategy, cash=100000, commission=0.00005, margin=1/100, fail_fast=False, bt_file = r"{bt_file}", stats_file = r"{stats_file}")
        ticker_name = f"{ticker}"
        stats = bt.backtest_stock(ticker_name, start_date)
        print("--- Backtest Statistics ---")
        print(stats)
        print("-------------------------")
    except NameError as e:
        print(f"Execution Error: A required class (like MultiBacktest or Strategy) was not defined by the LLM. Details: {{e}}")
    except Exception as e:
        print(f"An unexpected error occurred during final backtest execution: {{e}}")
'''

    # Clean previous main blocks and format code
    python_code = re.sub(r'if __name__ == .__main__.:.*', '', python_code, flags=re.DOTALL).strip()
    python_code = python_code + "\n\n" + main_block
    python_code = re.sub(r"```python\s*([\s\S]*?)\s*```", r"\1", python_code)

    error_message = ""

    while attempt < MAX_RETRIES:
        stdout_output, stderr_output = run_code(python_code)
        current_error = ""
        if "Traceback (most recent call last):" in stderr_output or "Traceback (most recent call last):" in stdout_output or "An unexpected error occurred" in stderr_output or "An unexpected error occurred" in stdout_output :
            current_error = stderr_output if "Traceback" in stderr_output else stdout_output

        if not current_error:
            print(f"Strategy {strategy_id} executed successfully on attempt {attempt + 1}.")
            return {
                "status": "success",
                "strategy_id": strategy_id,
                "output": stdout_output,
                "final_code": python_code,
                "stats_file": stats_file,
                "fig_file": bt_file,
                "attempts_taken": attempt + 1
            }

        
        attempt += 1
        error_message = current_error
        print(f"Error in Strategy {strategy_id}, Attempt {attempt}: \n{error_message}")

        if attempt >= MAX_RETRIES:
            break

        correction_prompt = f"""
        The following Python strategy code has an error:
        **Code:**
        ```python
        {python_code}
        ```
        **Error Message:**
        {error_message}
        Please fix the error and return only the corrected and complete Python code and do not repeat the earlier error in any of the code.
        """
        print("\n--- Sending request for code correction ---")
        corrected_response = get_llm_response(correction_prompt)
        python_code = extract_python_code(corrected_response)
        python_code = re.sub(r'if __name__ == .__main__.:.*', '', python_code, flags=re.DOTALL).strip()
        python_code = python_code + "\n\n" + main_block
        print(f"\nReceived Corrected Code (Attempt {attempt+1}):\n")

    return {
        "status": "failed",
        "strategy_id": strategy_id,
        "error": error_message,
        "final_code": python_code,
        "attempts_taken": attempt
    }