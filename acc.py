# Import python packages
import streamlit as st
import pandas as pd
import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col, lit, call_builtin, uniform, random, current_timestamp, datediff
from snowflake.snowpark import Session
import json
import hashlib
import uuid
import random as py_random
import re
from snowflake.snowpark.window import Window
from snowflake.snowpark.functions import row_number


def ensure_system_initialized(session: Session) -> None:
    """Initialize CURRENT_BEST if it's empty. ARM_STATS can start blank."""
    current_best_count = session.sql("SELECT COUNT(*) FROM CURRENT_BEST").collect()[0][0]
    if current_best_count == 0:
        print("Initializing CURRENT_BEST with random model...")
        session.sql("""
            INSERT INTO CURRENT_BEST (MODEL_ID, SCORE, UPDATED_AT)
            SELECT MODEL_ID, 0.5000, CURRENT_TIMESTAMP()
            FROM ARMS 
            WHERE IS_ACTIVE
            ORDER BY RANDOM()
            LIMIT 1
        """).collect()

def _escape_sql_literal(s: str) -> str:
    # double any single quotes for a SQL string literal
    return s.replace("'", "''") if isinstance(s, str) else s


def get_config_values(session: Session) -> tuple[float, float]:
    """Get epsilon and performance weight from config table."""
    config_df = session.table("CONFIG")
    eps = float(config_df.filter(col("KEY") == "epsilon").select("VALUE").collect()[0][0])
    perf_weight = float(config_df.filter(col("KEY") == "perf_weight").select("VALUE").collect()[0][0])
    return eps, perf_weight

def get_current_best_model(session: Session) -> str:
    """Get the current best performing model."""
    return session.table("CURRENT_BEST").select("MODEL_ID").collect()[0][0]

def select_exploration_model(session: Session, best_model: str) -> str:
    """Select a random model for exploration, excluding the current best."""
    available_models = session.table("ARMS").filter(
        (col("IS_ACTIVE") == True) & (col("MODEL_ID") != best_model)
    ).select("MODEL_ID").collect()
    
    if available_models:
        return py_random.choice(available_models)[0]
    else:
        return best_model

def choose_model(session: Session, epsilon: float) -> tuple[str, bool]:
    """Choose model using epsilon-greedy strategy."""
    best_model = get_current_best_model(session)
    rnd = py_random.random()
    
    if rnd < epsilon:
        # Exploration
        model = select_exploration_model(session, best_model)
        is_explore = model != best_model
    else:
        # Exploitation
        model = best_model
        is_explore = False
    
    return model, is_explore


def execute_model(session: Session, model: str, prompt: str) -> tuple[str, int, str]:
    """Execute the chosen model and measure latency."""
    model_lit  = _escape_sql_literal(model)
    prompt_lit = _escape_sql_literal(prompt)

    start_time = session.sql("SELECT CURRENT_TIMESTAMP()").collect()[0][0]

    # USE THE ESCAPED LITERALS HERE ↓↓↓
    cortex_result = session.sql(
        f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model_lit}', '{prompt_lit}')"
    ).collect()
    response = cortex_result[0][0]

    end_time = session.sql("SELECT CURRENT_TIMESTAMP()").collect()[0][0]
    latency_ms = session.sql(
        f"SELECT DATEDIFF('millisecond', '{start_time}', '{end_time}')"
    ).collect()[0][0]

    return response, latency_ms, start_time
def log_event(session: Session, event_id: str, prompt: str, model: str, 
              response: str, latency_ms: int, is_explore: bool) -> None:
    """Log the event to the EVENTS table."""
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    escaped_prompt = prompt.replace("'", "''")
    escaped_response = response.replace("'", "''")
    
    session.sql(f"""
        INSERT INTO EVENTS (EVENT_ID, TS, PROMPT_HASH, PROMPT, CHOSEN_MODEL, RESPONSE, LATENCY_MS, IS_EXPLORATION, REWARD, REWARD_READY)
        VALUES ('{event_id}', CURRENT_TIMESTAMP(), '{prompt_hash}', '{escaped_prompt}', '{model}', '{escaped_response}', {latency_ms}, {is_explore}, NULL, FALSE)
    """).collect()

def parse_judge_score(raw_output: str) -> float:
    """Parse judge score from raw output with fallback strategies."""
    score = 0.0
    
    try:
        # First try: clean JSON parsing
        json_text = raw_output.strip()
        if json_text.startswith('{"score":') and '}' in json_text:
            json_end = json_text.find('}') + 1
            json_text = json_text[:json_end]
        
        parsed = json.loads(json_text)
        score = float(parsed.get("score", 0.0))
        
    except (json.JSONDecodeError, ValueError, TypeError):
        # Second try: regex pattern matching
        json_pattern = r'\{\s*["\']?score["\']?\s*:\s*([0-9]*\.?[0-9]+)\s*\}'
        match = re.search(json_pattern, raw_output)
        if match:
            score = float(match.group(1))
        else:
            # Final fallback: look for score field
            score_pattern = r'["\']?score["\']?\s*:\s*([0-9]*\.?[0-9]+)'
            match = re.search(score_pattern, raw_output)
            if match:
                score = float(match.group(1))
    
    return max(0.0, min(1.0, score))

def get_judge_models(session: Session, exclude_model: str) -> list:
    """Get all active judge models excluding the chosen model."""
    return session.table("ARMS").filter(
        (col("IS_ACTIVE") == True) & (col("MODEL_ID") != exclude_model)
    ).select("MODEL_ID").collect()

def evaluate_response_with_judges(session: Session, event_id: str, model: str, response: str) -> float:
    """Evaluate response using judge models and return average score."""
    sum_score = 0.0
    judge_count = 0
    
    judge_models = get_judge_models(session, model)
    
    for judge_row in judge_models:
        judge_model = judge_row[0]
        
        # Create judge prompt
        judge_prompt = f"Rate the helpfulness of this answer 0-1. Respond ONLY JSON: {{\"score\": number}}\\n\\nAnswer:\\n{response}"
        escaped_judge_prompt = judge_prompt.replace("'", "''")
        
        # Get judge response
        judge_result = session.sql(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{judge_model}', '{escaped_judge_prompt}')").collect()
        raw_output = judge_result[0][0]
        
        # Parse score
        score = parse_judge_score(raw_output)
        
        # Log judge output
        escaped_raw_output = raw_output.replace("'", "''")
        session.sql(f"""
            INSERT INTO JUDGE_OUTPUTS (EVENT_ID, JUDGE_MODEL, RAW_OUTPUT, PARSED_SCORE)
            VALUES ('{event_id}', '{judge_model}', '{escaped_raw_output}', {score})
        """).collect()
        
        sum_score += score
        judge_count += 1
    
    return sum_score / judge_count if judge_count > 0 else 0.0

def get_model_cost(session: Session, model: str) -> float:
    """Get the cost per million tokens for a model."""
    model_cost_result = session.table("ARMS").filter(col("MODEL_ID") == model).select("CREDITS_PER_MTOK").collect()[0][0]
    return float(model_cost_result)

def calculate_reward(avg_performance: float, model_cost: float, perf_weight: float) -> float:
    """Calculate reward based on performance and cost efficiency."""
    cost_efficiency = 1.0 / model_cost if model_cost > 0 else 0.0
    return (perf_weight * avg_performance) + ((1 - perf_weight) * cost_efficiency)

def update_event_reward(session: Session, event_id: str, reward: float) -> None:
    """Update event with calculated reward."""
    session.sql(f"""
        UPDATE EVENTS
        SET REWARD = {reward}, REWARD_READY = TRUE
        WHERE EVENT_ID = '{event_id}'
    """).collect()

def update_arm_stats(session: Session, model: str, avg_performance: float, model_cost: float, reward: float) -> None:
    """Update ARM_STATS using EWMA."""
    session.sql(f"""
        MERGE INTO ARM_STATS t
        USING (SELECT '{model}' AS MODEL_ID) s
        ON t.MODEL_ID = s.MODEL_ID
        WHEN MATCHED THEN
            UPDATE SET
                PULLS = PULLS + 1,
                EWMA_PERF = (0.8 * EWMA_PERF + 0.2 * {avg_performance}),
                EWMA_COST = (0.8 * EWMA_COST + 0.2 * {model_cost}),
                SCORE = (0.8 * SCORE + 0.2 * {reward}),
                LAST_UPDATE = CURRENT_TIMESTAMP()
        WHEN NOT MATCHED THEN
            INSERT (MODEL_ID, PULLS, EWMA_PERF, EWMA_COST, SCORE, LAST_UPDATE)
            VALUES ('{model}', 1, {avg_performance}, {model_cost}, {reward}, CURRENT_TIMESTAMP())
    """).collect()

def update_current_best(session: Session) -> None:
    """Update cached best model based on latest scores."""
    session.sql("DELETE FROM CURRENT_BEST").collect()
    session.sql("""
        INSERT INTO CURRENT_BEST (MODEL_ID, SCORE, UPDATED_AT)
        SELECT MODEL_ID, SCORE, CURRENT_TIMESTAMP()
        FROM ARM_STATS
        ORDER BY SCORE DESC
        LIMIT 1
    """).collect()

def process_exploration_feedback(session: Session, event_id: str, model: str, response: str, perf_weight: float) -> None:
    """Process feedback for exploration attempts."""
    # Evaluate with judges
    avg_performance = evaluate_response_with_judges(session, event_id, model, response)
    
    # Get model cost and calculate reward
    model_cost = get_model_cost(session, model)
    reward = calculate_reward(avg_performance, model_cost, perf_weight)
    
    # Update event, stats, and best model
    update_event_reward(session, event_id, reward)
    update_arm_stats(session, model, avg_performance, model_cost, reward)
    update_current_best(session)

def adaptive_cortex_complete(session: Session, prompt: str) -> str:
    """
    Multi-armed bandit LLM router using epsilon-greedy exploration
    """
    # Initialize system if needed
    ensure_system_initialized(session)
    
    # Get configuration
    epsilon, perf_weight = get_config_values(session)
    
    # Generate event ID
    event_id = str(uuid.uuid4())
    
    # Choose model using epsilon-greedy strategy
    model, is_explore = choose_model(session, epsilon)
    
    # Execute chosen model
    response, latency_ms, start_time = execute_model(session, model, prompt)
    
    # Log the event
    log_event(session, event_id, prompt, model, response, latency_ms, is_explore)
    
    # Process exploration feedback if needed
    if is_explore:
        process_exploration_feedback(session, event_id, model, response, perf_weight)
    
    # Return result as JSON
    result = {
        "model": model,
        "response": response
    }
    return json.dumps(result)

def adaptive_cortex_complete_over_table(
    session: Session,
    base_prompt: str,
    table_name: str,
    col_name: str,
    id_col: str = None,
    limit: int = None,
    write_to_table: str = None,
):
    """
    Run adaptive_cortex_complete() over each row of a Snowflake table column.
    """
    # Load table
    df = session.table(table_name)

    # Add a row number if no id column is given
    if id_col is None:
        w = Window.order_by(lit(1))
        df = df.with_column("ROW_NUM", row_number().over(w))
        id_col = "ROW_NUM"

    # Optionally limit rows
    if limit:
        df = df.limit(limit)

    results = []

    for row in df.collect():
        row_id = row[id_col]
        text = row[col_name]

        print(f"Processing row {row}, (ID={row_id})...")

        # Run your existing adaptive_cortex_complete on the text
        out_json = adaptive_cortex_complete(session, f"{base_prompt}\n\n{text}")
        results.append({"ID": row_id, "OUTPUT": out_json})

    # Optionally persist results to a Snowflake table
    if write_to_table:
        result_df = session.create_dataframe(results)
        result_df.write.mode("overwrite").save_as_table(write_to_table)

    return results
