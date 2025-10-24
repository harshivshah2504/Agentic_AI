# main_crew.py
import os
from tests import *
from crewai import Agent, Task, Crew, Process
from backtesting_agent.prompts import *
from tools.code_execution_tool import CodeExecutionTool
from crewai import LLM
import google.generativeai as genai
import json
from verification import checker
from dotenv import load_dotenv
load_dotenv()


    # Develop an advanced long-short, purely technical algorithmic trading strategy for Microsoft (MSFT), starting from July 1st, 2025.

    # 1.  **Trend and Momentum Core (Quant Task):**
    #     -   Use a triple moving average system (9 EMA, 21 EMA, 55 EMA). 
    #         -   Go long when 9 > 21 > 55 and price closes above all three.
    #         -   Exit when 9 crosses below 21 or price closes below 55 EMA.
    #     -   Confirm momentum using RSI(14) > 60 and Stochastic %K(14,3,3) > %D, ensuring the crossover occurs above 50.

    # 3.  **Trade Management:**
    #     -   Once conditions are met, enter on a breakout above the previous day’s high.
    #     -   If the breakout candle closes with volume > 1.5× its 20-day average, increase position size by 25%.
    #     -   No short positions; this strategy trades long-only.

    # 4.  **Risk Management:**
    #     -   Use fixed fractional risk: risk 1.25% of current equity per trade.
    #     -   Initial stop-loss: 2×ATR below entry.
    #     -   Take-profit target: 3×ATR.
    #     -   Introduce a trailing stop that activates after profit > 1.5×ATR, tightening to 1×ATR distance.


llm = LLM(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)


code_verifier_tool = CodeExecutionTool()

def load_template(filepath):
    with open(filepath, 'r') as f:
        return f.read()

risk_template = load_template('/home/R4/Harshiv/templates/risk_management.py')
trade_template = load_template('/home/R4/Harshiv/templates/trade_management.py')


decomposer = Agent(
  role='Strategy Task Decomposer',
  goal=DECOMPOSER_SYSTEM_MESSAGE,
  backstory='You are an expert project manager at a quantitative hedge fund. Your strength is translating high-level, sometimes ambiguous, strategy ideas from portfolio managers into structured, technical specifications for your development team.',
  llm=llm,
  verbose=True
)

quant = Agent(
  role='Quantitative Developer',
  goal=GENERATE_BACKTEST_CODE_PROMPT_MAIN,
  backstory=GENERATE_STRATEGY_CODE_SYSTEM_MESSAGE,
  llm=llm,
  verbose=True,
  allow_delegation=False
)

risk_analyst = Agent(
    role='Risk Management Specialist',
    goal='Create a Python class for risk management based on the user\'s request and provided templates. The class must be standalone and syntactically correct.',
    backstory=f'You are a financial engineer specializing in risk models. You are provided with examples of risk management classes and must adapt them to the specific task. Ensure the final size is always >=1 .Your template for reference is:\n{risk_template}',
    llm=llm,
    verbose=True
)


trade_analyst = Agent(
    role='Trade Management Specialist',
    goal='Create a Python class for trade management (stop-loss and take-profit) based on the user\'s request and provided templates. The class must be standalone and syntactically correct.',
    backstory=f'You are a trade execution expert who designs rules for entering and exiting positions. Your template for reference is:\n{trade_template}',
    llm=llm,
    verbose=True
)

assembler = Agent(
    role='Lead Software Engineer & Code Assembler',
    goal=f'Combine the individual Python code snippets (strategy logic, risk class, trade class) into a single, complete, and syntactically correct code following the instructions mentioned in {GENERATE_BACKTEST_CODE_PROMPT_MAIN}. Follow the import in the template and the variables in set_params() should be as it is in the template. Used the backtesting library mentioned in the template nothing else',
    backstory='You are the lead engineer responsible for final code integration. You are a stickler for detail, ensuring that all parts fit together perfectly, all imports are correct, and the final script is clean and readable.',
    llm=llm,
    verbose=True
)

verifier = Agent(
  role='Quality Assurance & Code Verification Engineer',
  goal='Rigorously test the fully assembled Python backtesting script by executing it. You must use your Code Execution Tool to run the script and report back the exact outcome: either success with the backtest results or failure with the detailed error message.',
  backstory='You are an automated testing bot. Your world is binary: code either runs perfectly, or it fails. You do not write or fix code; you only execute and report. Your reports are precise and literal.',
  tools=[code_verifier_tool],
  llm=llm,
  verbose=True
)




def run_crew(nlp_query: str):
    print("Starting the Multi-Agent Backtesting Strategy Generation Crew...")

    # --- Define tasks here so they can use the query ---
    decomposition_task = Task(
        description=f"Decompose the following user query into a structured JSON object. The JSON must contain keys for 'ticker', 'start_date', 'strategy_task', 'risk_task', and 'trade_task'.\n\nQuery:\n---\n{nlp_query}",
        expected_output="A single, valid JSON object containing the decomposed tasks and metadata as string values.",
        agent=decomposer
    )


    # Task 2: Generate the core strategy code
    quant_task = Task(
        description=f"Based on the 'strategy_task' from the decomposition plan, write the Python code following the instruction given {GENERATE_BACKTEST_CODE_PROMPT_MAIN}",
        expected_output="A Python code snippet containing the `calculate_indicators` and `next` methods.",
        agent=quant,
        context=[decomposition_task]
    )

    # Task 3: Generate the risk management code
    risk_task = Task(
        description=(
            "Write a complete, standalone Python class named `RiskManagement` that encapsulates all logic "
            "for position sizing and dynamic risk control based on performance feedback. "
            "The class should be self-contained and not depend on external code. "
            "It must expose only three main public methods used by the trading strategy: "
            "`get_risk_per_trade()`, `update_after_loss()`, and `update_after_win()`. The parameters for these methods should not be anything other than self."
            "All other helper methods or internal variables should remain private and act as supporters to these. "
            "Ensure the class supports flexible configuration (like initial risk, max risk, drawdown limits, etc.), "
            "maintains state across updates, and handles risk scaling after a sequence of wins or losses. "
            "The code should be idiomatic, production-grade Python with type hints and docstrings."
        ),
        expected_output=(
            "A single Python code block containing the complete implementation of the `RiskManagement` class."
        ),
        agent=risk_analyst,
        context=[decomposition_task],
    )

    # Task 4: Generate the trade management code
    trade_task = Task(
        description=(
            "Write a complete, standalone Python class named `TradeManagement` that encapsulates all logic "
            "for calculating take-profit (TP) and stop-loss (SL) levels for trades. "
            "The main trading strategy will only interact with the method `calculate_tp_sl(direction)`. The parameter 'direction' indicates whether the trade is 'buy' or 'short', it should be the only parameter apart from self."
            "Other internal functions or state variables should support this method and remain private. "
            "handle position direction (long/short), and ensure numerical stability. "
            "Implement type hints, docstrings, and follow clean, modular Python design."
        ),
        expected_output=(
            "A single Python code block containing the complete implementation of the `TradeManagement` class."
        ),
        agent=trade_analyst,
        context=[decomposition_task],
    )


    # Task 5: Assemble the final script
    assembly_task = Task(
        description=f"Assemble the code snippets from the quant, risk, and trade tasks into a final, runnable script which should support the backtesting framework with methods same as of backtesting.py. Use the provided master template {GENERATE_BACKTEST_CODE_PROMPT_MAIN}. Ensure all parts are correctly placed, and the final output is a single block of Python code where the risk management and trade management classes are the output from risk_task and trade_task",
        expected_output="A single string containing the full, complete, and ready-to-run Python backtesting script with the given format.",
        agent=assembler,
        context=[quant_task, risk_task, trade_task, decomposition_task] 
    )

    # --- Crew setup ---
    strategy_crew = Crew(
        agents=[decomposer, quant, risk_analyst, trade_analyst, assembler],
        tasks=[decomposition_task, quant_task, risk_task, trade_task, assembly_task],
        process=Process.sequential,
        verbose=1
    )

    # --- Run Crew ---
    result = strategy_crew.kickoff()
    raw_output = json.loads(decomposition_task.output.raw)
    ticker = raw_output.get("ticker", "MSFT")
    start_date = raw_output.get("start_date", "20250701")

    result = checker(assembly_task.output.raw, ticker, start_date)
    return result
