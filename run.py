# run.py

from app import create_app
from config import engine, Base
from flask_cors import CORS
from app.tasks import reset_weekly_goals, carry_forward_unset_intentions
from apscheduler.schedulers.background import BackgroundScheduler

# ✅ Optional: create DB tables if not already created
with engine.begin() as conn:
    Base.metadata.create_all(conn)

# 🔥 Create Flask app
app = create_app()
CORS(app)

# ✅ APScheduler setup for weekly tasks
scheduler = BackgroundScheduler()

# 🔁 Reset weekly goals every Saturday at 12:00 AM
scheduler.add_job(
    reset_weekly_goals,
    trigger='cron',
    day_of_week='sat',
    hour=0,
    minute=0
)

# 🔁 Carry forward intentions on Sunday night at 11:59 PM
scheduler.add_job(
    carry_forward_unset_intentions,
    trigger='cron',
    day_of_week='sun',
    hour=23,
    minute=59
)

scheduler.start()

# 🚀 Launch the server
if __name__ == '__main__':
    app.run(debug=True)
