import pytest
from env import NERCEnv
from models import Action
from grader import grade_episode
from tasks import TASKS

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    return NERCEnv()

# ─── Test reset() ─────────────────────────────────────────────────────────────

def test_reset_task1(env):
    obs = env.reset("task_1")
    assert obs.mode == "hospital"
    assert len(obs.patients) == 3
    assert obs.time_left == 15
    assert obs.done == False

def test_reset_task2(env):
    obs = env.reset("task_2")
    assert obs.mode == "hospital"
    assert len(obs.patients) == 6
    assert obs.time_left == 20

def test_reset_task3(env):
    obs = env.reset("task_3")
    assert obs.mode == "disaster"
    assert len(obs.patients) == 10
    assert len(obs.rescue_teams) >= 3

def test_reset_task4(env):
    obs = env.reset("task_4")
    assert obs.mode == "disaster"
    assert len(obs.patients) == 12

def test_reset_task5(env):
    obs = env.reset("task_5")
    assert obs.mode == "disaster"
    assert len(obs.patients) == 15

def test_reset_invalid_task(env):
    with pytest.raises(ValueError):
        env.reset("task_999")

def test_reset_returns_clean_state(env):
    env.reset("task_1")
    env.step(Action(action_type="assign_doctor", patient_id="p1", doctor_id="d1"))
    obs = env.reset("task_1")
    for p in obs.patients:
        assert p.assigned_doctor is None
        assert p.assigned_hospital is None
        assert p.steps_without_care == 0
        assert p.alive == True

# ─── Test step() ──────────────────────────────────────────────────────────────

def test_step_requires_reset(env):
    with pytest.raises(RuntimeError):
        env.step(Action(action_type="wait"))

def test_step_assign_doctor(env):
    env.reset("task_1")
    obs, reward, done, info = env.step(Action(
        action_type="assign_doctor",
        patient_id="p1",
        doctor_id="d1"
    ))
    assert reward > 0
    assert obs.time_left == 14
    assigned = next(p for p in obs.patients if p.id == "p1")
    assert assigned.assigned_doctor == "d1"

def test_step_dispatch_ambulance(env):
    env.reset("task_1")
    obs, reward, done, info = env.step(Action(
        action_type="dispatch_ambulance",
        patient_id="p1",
        ambulance_id="a1",
        hospital_id="h1"
    ))
    assert reward > 0
    patient = next(p for p in obs.patients if p.id == "p1")
    assert patient.assigned_hospital == "h1"

def test_step_wait_penalizes_with_critical_patients(env):
    env.reset("task_1")
    _, reward, _, _ = env.step(Action(action_type="wait"))
    assert reward < 0

def test_step_time_decrements(env):
    obs = env.reset("task_1")
    initial_time = obs.time_left
    obs, _, _, _ = env.step(Action(action_type="wait"))
    assert obs.time_left == initial_time - 1

def test_step_done_when_time_runs_out(env):
    env.reset("task_1")
    done = False
    for _ in range(20):
        _, _, done, _ = env.step(Action(action_type="wait"))
        if done:
            break
    assert done == True

def test_step_after_done_raises(env):
    env.reset("task_1")
    for _ in range(20):
        _, _, done, _ = env.step(Action(action_type="wait"))
        if done:
            break
    with pytest.raises(RuntimeError):
        env.step(Action(action_type="wait"))

# ─── Test reward signals ──────────────────────────────────────────────────────

def test_reward_critical_doctor_higher_than_mild(env):
    env.reset("task_1")
    _, r_critical, _, _ = env.step(Action(
        action_type="assign_doctor",
        patient_id="p1",
        doctor_id="d1"
    ))
    env.reset("task_1")
    _, r_mild, _, _ = env.step(Action(
        action_type="assign_doctor",
        patient_id="p3",
        doctor_id="d1"
    ))
    assert r_critical > r_mild

def test_reward_icu_overflow_negative(env):
    env.reset("task_2")
    # fill hospital h1 to capacity
    for i in range(3):
        env.step(Action(
            action_type="dispatch_ambulance",
            patient_id=f"p{i+1}",
            ambulance_id="a1",
            hospital_id="h1"
        ))
    # try to overflow
    _, reward, _, _ = env.step(Action(
        action_type="dispatch_ambulance",
        patient_id="p4",
        ambulance_id="a1",
        hospital_id="h1"
    ))
    assert reward < 0

def test_reward_invalid_action_negative(env):
    env.reset("task_1")
    _, reward, _, _ = env.step(Action(
        action_type="assign_doctor",
        patient_id="p999",
        doctor_id="d1"
    ))
    assert reward < 0

# ─── Test state() ─────────────────────────────────────────────────────────────

def test_state_requires_reset(env):
    with pytest.raises(RuntimeError):
        env.state_snapshot()

def test_state_returns_dict(env):
    env.reset("task_1")
    state = env.state_snapshot()
    assert isinstance(state, dict)
    assert "patients" in state
    assert "hospitals" in state
    assert "mode" in state

def test_state_matches_observation(env):
    obs = env.reset("task_1")
    state = env.state_snapshot()
    assert len(state["patients"]) == len(obs.patients)
    assert state["mode"] == obs.mode

# ─── Test grader ──────────────────────────────────────────────────────────────

def test_grader_score_range(env):
    env.reset("task_1")
    for _ in range(5):
        env.step(Action(action_type="wait"))
    result = grade_episode(env.state)
    assert 0.0 <= result["score"] <= 1.0

def test_grader_perfect_score(env):
    env.reset("task_1")
    env.step(Action(action_type="assign_doctor", patient_id="p1", doctor_id="d1"))
    env.step(Action(action_type="dispatch_ambulance", patient_id="p1", ambulance_id="a1", hospital_id="h1"))
    env.step(Action(action_type="assign_doctor", patient_id="p2", doctor_id="d2"))
    env.step(Action(action_type="dispatch_ambulance", patient_id="p2", ambulance_id="a2", hospital_id="h1"))
    env.step(Action(action_type="dispatch_ambulance", patient_id="p3", ambulance_id="a1", hospital_id="h1"))
    result = grade_episode(env.state)
    assert result["score"] > 0.8

def test_grader_has_breakdown(env):
    env.reset("task_1")
    env.step(Action(action_type="wait"))
    result = grade_episode(env.state)
    assert "breakdown" in result
    assert "summary" in result
    assert "survival_score" in result["breakdown"]

# ─── Test tasks ───────────────────────────────────────────────────────────────

def test_all_tasks_exist():
    assert "task_1" in TASKS
    assert "task_2" in TASKS
    assert "task_3" in TASKS
    assert "task_4" in TASKS
    assert "task_5" in TASKS

def test_tasks_have_required_fields():
    required = ["mode", "time_limit", "description", "difficulty",
                "patients", "hospitals", "ambulances", "doctors", "rescue_teams"]
    for task_id, task in TASKS.items():
        for field in required:
            assert field in task, f"Task {task_id} missing field: {field}"

def test_difficulty_progression():
    difficulties = [TASKS[f"task_{i}"]["difficulty"] for i in range(1, 6)]
    assert difficulties[0] == "easy"
    assert difficulties[1] == "medium"
    assert difficulties[4] == "expert"

def test_disaster_tasks_have_rescue_teams():
    for task_id, task in TASKS.items():
        if task["mode"] == "disaster":
            assert len(task["rescue_teams"]) > 0, f"{task_id} disaster task has no rescue teams"