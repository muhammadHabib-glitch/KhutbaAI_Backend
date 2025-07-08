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
