import os
import json
import requests
from typing import List, Optional
from openai import OpenAI

# ─── Required environment variables ──────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")
BASE_URL     = os.getenv("NERC_ENV_URL", "http://localhost:7860")

# ─── Validate HF_TOKEN (mandatory per guidelines) ────────────────────────────
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK = "nerc-env"
MAX_STEPS = 50
TEMPERATURE = 0.2
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.5

# ─── OpenAI client ────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL
)

SYSTEM_PROMPT = """You are an emergency response coordinator AI.
You will receive the current state of an emergency environment and must decide the best action.

Always respond with a single JSON object with this exact structure:
{
    "action_type": "assign_doctor|dispatch_ambulance|transfer_patient|dispatch_rescue|prioritize_patient|wait",
    "patient_id": "p1",
    "doctor_id": "d1",
    "ambulance_id": "a1",
    "hospital_id": "h1",
    "rescue_team_id": "r1",
    "location": "zone_a"
}

Only include fields relevant to your chosen action_type.
Prioritize critical patients. Avoid ICU overflow. Save as many lives as possible."""

# ─── Mandatory log functions ──────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ─── LLM agent ────────────────────────────────────────────────────────────────
def get_action_from_llm(obs: dict) -> dict:
    try:
        prompt = f"Current environment state:\n{json.dumps(obs, indent=2)}\n\nWhat action should be taken? Respond with a single JSON action object only."
        import time
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt}
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                raw = response.choices[0].message.content.strip()
                if "```" in raw:
                    raw = raw.split("```")[1].replace("json", "").strip()
                return json.loads(raw)
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    time.sleep(45 * (attempt + 1))
                else:
                    raise e
    except Exception as e:
        import sys; print(f"[DEBUG] LLM error: {e} — using rule-based agent", file=sys.stderr, flush=True)
    return get_rule_based_action(obs)

def get_rule_based_action(obs: dict) -> dict:
    patients   = obs.get("patients",    [])
    doctors    = obs.get("doctors",     [])
    ambulances = obs.get("ambulances",  [])
    hospitals  = obs.get("hospitals",   [])
    rescue     = obs.get("rescue_teams", [])

    available_doctors    = [d for d in doctors    if d["status"] == "available"]
    available_ambulances = [a for a in ambulances if a["status"] == "available"]
    available_rescue     = [r for r in rescue      if r["status"] == "available"]
    alive                = [p for p in patients    if p["alive"]]
    hospitals_with_space = [h for h in hospitals   if h["current_patients"] < h["icu_capacity"]]
    unallocated          = [p for p in alive        if p["assigned_hospital"] is None]

    for p in alive:
        if p["severity"] == "critical" and p["assigned_doctor"] is None and available_doctors:
            return {"action_type": "assign_doctor", "patient_id": p["id"], "doctor_id": available_doctors[0]["id"]}

    if available_rescue:
        unrescued = [p for p in alive if not p["rescued"]]
        if unrescued:
            return {"action_type": "dispatch_rescue", "rescue_team_id": available_rescue[0]["id"], "location": "disaster_zone"}

    if unallocated and available_ambulances and hospitals_with_space:
        priority     = {"critical": 0, "moderate": 1, "mild": 2}
        best_patient = sorted(unallocated, key=lambda p: priority.get(p["severity"], 3))[0]
        best_hosp    = max(hospitals_with_space, key=lambda h: h["icu_capacity"] - h["current_patients"])
        return {"action_type": "dispatch_ambulance", "patient_id": best_patient["id"], "ambulance_id": available_ambulances[0]["id"], "hospital_id": best_hosp["id"]}

    for p in alive:
        if p["severity"] == "moderate" and p["assigned_doctor"] is None and available_doctors:
            return {"action_type": "assign_doctor", "patient_id": p["id"], "doctor_id": available_doctors[0]["id"]}

    for p in alive:
        if p["severity"] == "mild" and p["assigned_doctor"] is None and available_doctors:
            return {"action_type": "assign_doctor", "patient_id": p["id"], "doctor_id": available_doctors[0]["id"]}

    return {"action_type": "wait"}

# ─── Episode runner ───────────────────────────────────────────────────────────
def run_episode(task_id: str) -> dict:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    try:
        res  = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
        obs  = res.json()
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action     = get_action_from_llm(obs)
            action_str = json.dumps(action)
            error      = None

            try:
                step_res = requests.post(
                    f"{BASE_URL}/step",
                    json={"action": action}
                ).json()

                if "observation" in step_res:
                    obs    = step_res["observation"]
                    reward = float(step_res.get("reward", 0))
                    done   = bool(step_res.get("done", False))
                elif "detail" in step_res:
                    error  = step_res["detail"]
                    reward = 0.0
                else:
                    reward = 0.0

            except Exception as e:
                error  = str(e)
                reward = 0.0

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        try:
            grade_res = requests.post(f"{BASE_URL}/grader").json()
            score     = float(grade_res.get("score", 0.0))
        except Exception:
            score = 0.0

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score, "steps": steps_taken, "success": success}

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    tasks   = ["task_1", "task_2", "task_3", "task_4", "task_5"]
    results = []
    for task_id in tasks:
        result = run_episode(task_id)
        results.append(result)
    avg = sum(r["score"] for r in results) / len(results)
    print(f"\n[SUMMARY] average_score={avg:.4f}", flush=True)

if __name__ == "__main__":
    main()
