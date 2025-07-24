# tasks.py
from datetime import datetime
import random
from config import Session
from app.models import User


def reset_weekly_goals():
    """
    Resets each user's weekly progress and assigns a new goal.
    Penalizes users who didn't meet last week's goal by deducting Nurbits.
    """
    session = Session()
    try:
        users = session.query(User).all()
        for user in users:
            # Penalize if last week's goal not met
            if (user.WeeklyProgress or 0) < (user.CurrentWeekGoal or 5):
                user.Nurbits = max((user.Nurbits or 0) - 5, 0)

            # Reset progress and assign a new random goal between 3 and 6
            user.WeeklyProgress = 0
            user.CurrentWeekGoal = random.randint(3, 6)

        session.commit()
        print(f"[{datetime.utcnow()}] ✅ Weekly goals reset completed.")
    except Exception as e:
        session.rollback()
        print(f"[{datetime.utcnow()}] ❌ Failed to reset goals: {e}")
    finally:
        session.close()


def carry_forward_unset_intentions():
    session = Session()
    try:
        from app.models import Intention  # 🔁 Ensure correct import
        users = session.query(User).all()
        current_week = datetime.utcnow().date()  # Or use get_week_start() if defined

        for user in users:
            # Skip if already set this week
            already_set = session.query(Intention).filter_by(UserId=user.Id, WeekStart=current_week).first()
            if already_set:
                continue

            # Get the most recent past intention
            last_intent = session.query(Intention) \
                .filter(Intention.UserId == user.Id, Intention.WeekStart < current_week) \
                .order_by(Intention.WeekStart.desc()).first()

            if last_intent:
                new_intention = Intention(
                    UserId=user.Id,
                    WeekStart=current_week,
                    Intention=last_intent.Intention,
                    Level=last_intent.Level,
                    CreatedAt=datetime.utcnow(),         # 👈
                     UpdatedAt=datetime.utcnow()
                )
                session.add(new_intention)
                user.CurrentWeekGoal = int(last_intent.Intention.split()[0])
                user.LastGoalSet = datetime.utcnow()

            else:
                # 🔁 Fallback for users with no past intention — set to 1 reflection
                new_intention = Intention(
                    UserId=user.Id,
                    WeekStart=current_week,
                    Intention="1 reflections",
                    Level=1
                )
                session.add(new_intention)
                user.CurrentWeekGoal = 1
                user.LastGoalSet = datetime.utcnow()

        session.commit()
        print(f"[{datetime.utcnow()}] ✅ Carried forward unset intentions (including default for new users).")
    except Exception as e:
        session.rollback()
        print(f"[{datetime.utcnow()}] ❌ Failed to carry forward intentions: {e}")
    finally:
        session.close()
