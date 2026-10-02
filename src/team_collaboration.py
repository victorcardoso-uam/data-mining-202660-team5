"""
Team Collaboration & Engineering Contribution Registry
Data Mining & Modern AI Systems (IIND4417) — Session 13
"""
import datetime

TEAM_REGISTRY = {
    "cohort": "Team 5",
    "repository": "data-mining-202660-team5",
    "members": [
        {
            "name": "Farid Isaac Alfaro Omana",
            "student_id": "00432439",
            "role": "Lead Data Engineer",
            "assigned_reviewer": "Andres Correa Solis",
            "git_feature_branch": "feature/activity-10-farid-alfaro",
            "preferred_ai_assistant": "Claude",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        # Teammates will append their dictionary blocks via their respective branches!
    ],
}


def display_team_roster():
    print(f"\n{'=' * 20} {TEAM_REGISTRY['cohort']} ACTIVE ROSTER {'=' * 20}")
    for m in TEAM_REGISTRY["members"]:
        print(
            f"* {m['name']} ({m['student_id']}) | Role: {m['role']} "
            f"| Branch: {m['git_feature_branch']}"
        )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    display_team_roster()
