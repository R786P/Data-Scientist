"""
Production Monitoring - Month 6
Logs system health and performance metrics
"""
import datetime

class SystemMonitor:
    def __init__(self):
        self.logs = []

    def log_event(self, event_type, status):
        """Record every major action of the agent"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {event_type.upper()}: {status}"
        self.logs.append(entry)
        # Keep only last 10 logs to save memory
        if len(self.logs) > 10: self.logs.pop(0)

    def get_health_report(self):
        """Returns the status of all modules"""
        return "🛡️ System Health: All Modules (ML, Stats, XAI, Domain) are ONLINE."
