import time
import firebase_admin
from firebase_admin import credentials, firestore
from multiprocessing import Process, Queue
import file_analysis
import analyst_problem_feasibility_xai

# Firebase init
SERVICE_ACCOUNT_KEY = "startupiq-c1fe1-firebase-adminsdk-fbsvc-8b23336991.json"
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Ordered workflow
step_map = {
    "file_analysis": "team_agent",
    "team_agent":"business_analysis",
    "business_analysis": "financial_stats_agent",
    "financial_stats_agent": "final_agent",
    "final_agent": None
}

step_function_map = {
    "file_analysis": file_analysis.main,
    "problem_feasibility" : analyst_problem_feasibility_xai.main
}

# Shared queue
job_queue = Queue()

def worker(job_queue):
    """Worker that consumes jobs from the queue"""
    while True:
        doc_id, step_name = job_queue.get()
        print(f"[Worker] Processing {step_name} for {doc_id}")
        if step_function_map.get(step_name):
            try:
                step_func = step_function_map[step_name]
                step_func(doc_id,db)
            except Exception as e:
                print(f"Error processing {step_name} for {doc_id}: {e}")
                pass
        else:
            time.sleep(5)  # Simulate work for unimplemented steps
        

        doc_ref = db.collection("sync-job").document(doc_id)
        next_step = step_map.get(step_name)

        if next_step:
            doc_ref.update({
                "current_step": next_step,
                "current_step_status": "not_started",
                "status": "in_progress",
                "percentage": firestore.Increment(20)
            })
            print(f"[Worker] {step_name} done, moved to {next_step}")
        else:
            doc_ref.update({
                "current_step_status": "completed",
                "status": "done",
                "percentage": 100
            })
            print(f"[Worker] All steps done for {doc_id}")

def listener(col_snapshot, changes, read_time):
    for change in changes:
        cur_doc = change.document.to_dict()
        doc_id = change.document.id

        cur_step = cur_doc.get("current_step")
        cur_status = cur_doc.get("current_step_status")

        if cur_status == "not_started":
            # Reserve the step before enqueuing
            db.collection("sync-job").document(doc_id).update({
                "current_step_status": "queued"
            })
            job_queue.put((doc_id, cur_step))
            print(f"[Listener] Enqueued {doc_id} for {cur_step}")

# Start worker in separate process
p = Process(target=worker, args=(job_queue,))
p.start()

# Start listener
col_watch = db.collection("sync-job").on_snapshot(listener)
print("Listening for changes...")

while True:
    time.sleep(10)
