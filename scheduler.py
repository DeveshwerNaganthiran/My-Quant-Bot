import subprocess
import time
import sys
import datetime

# Configuration
DURATION_SECONDS = 3 * 60 * 60  # 3 hours (in seconds)
# DURATION_SECONDS = 60  # <-- Uncomment this to test it for 1 minute first!
SCRIPTS = ["main_live.py", "main_live_inverse.py"]

def get_current_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def run_alternating_scheduler():
    print(f"[{get_current_time()}] Starting Alternating Bot Scheduler...")
    print(f"[{get_current_time()}] Mode: Switch every {DURATION_SECONDS / 3600} hours.")
    
    current_script_index = 0
    
    while True:
        script_to_run = SCRIPTS[current_script_index]
        print(f"\n{'='*50}")
        print(f"[{get_current_time()}] 🚀 LAUNCHING: {script_to_run}")
        print(f"{'='*50}\n")
        
        # Start the python script as a subprocess
        process = subprocess.Popen([sys.executable, script_to_run])
        
        try:
            # Let the process run for the specified duration
            process.wait(timeout=DURATION_SECONDS)
            print(f"[{get_current_time()}] ⚠️ {script_to_run} exited early on its own.")
            
        except subprocess.TimeoutExpired:
            # 3 hours have passed, time to swap
            print(f"\n[{get_current_time()}] ⏳ Time is up! Stopping {script_to_run}...")
            process.terminate()  # Send graceful termination signal
            
            try:
                # Give the bot 10 seconds to close connections (MT5, DB, etc.) gracefully
                process.wait(timeout=10)
                print(f"[{get_current_time()}] ✅ {script_to_run} stopped gracefully.")
            except subprocess.TimeoutExpired:
                print(f"[{get_current_time()}] ❌ Process didn't stop in time. Force killing...")
                process.kill()
        
        # Toggle to the other script (0 becomes 1, 1 becomes 0)
        current_script_index = (current_script_index + 1) % len(SCRIPTS)
        
        # Pause for 10 seconds before starting the next bot
        # This is CRITICAL to ensure MT5 terminal connections are fully released
        print(f"[{get_current_time()}] 💤 Waiting 10 seconds for MT5 terminal to clear connections...")
        time.sleep(10)

if __name__ == "__main__":
    try:
        run_alternating_scheduler()
    except KeyboardInterrupt:
        print("\nScheduler manually stopped by user.")
        sys.exit(0) 