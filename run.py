# run.py

from app import create_app
from config import engine, Base
from flask_cors import CORS

# ✅ Optional: create DB tables if not already created
with engine.begin() as conn:
    Base.metadata.create_all(conn)

# 🔥 Create Flask app
app = create_app()
CORS(app)

# ✅ APScheduler setup for weekly goal reset
from apscheduler.schedulers.background import BackgroundScheduler
from app.tasks import reset_weekly_goals

scheduler = BackgroundScheduler()
scheduler.add_job(
    reset_weekly_goals,
    trigger='cron',
    day_of_week='sat',
    hour=0,
    minute=0
)
scheduler.start()

# 🚀 Launch the server
if __name__ == '__main__':
    app.run(debug=True)
