from flask import Flask, render_template_string, request, redirect, url_for, render_template
import threading
import time
import os
import sys
import queue

sys.path.append(os.getcwd())
from modules import test_parallel, test_sequential

app = Flask(__name__)

config = {
    "N": 5,
    "M": 132,
    "t1": 10,
    "t2": 30,
    "PK": 1.0,
    "PM": 1.0,
    "K": 10,
    "Z": 10,
    "num_iters": 1,
    "elite_number": 1
}

result_queue = queue.Queue()
running = False
results_data = {}
@app.route("/", methods=["GET", "POST"])
def index():
    global running
    if request.method == "POST":
    
        config["N"] = int(request.form.get("N", 5))
        config["M"] = int(request.form.get("M", 132))
        config["t1"] = int(request.form.get("t1", 10))
        config["t2"] = int(request.form.get("t2", 30))
        config["PK"] = float(request.form.get("PK", 1.0))
        config["PM"] = float(request.form.get("PM", 1.0))
        config["K"] = int(request.form.get("K", 10))
        config["Z"] = int(request.form.get("Z", 10))
        config["num_iters"] = int(request.form.get("num_iters", 1))
        config["elite_number"] = int(request.form.get("elite_number", 1))
        
        use_saved = request.form.get("use_saved") == "on"
        if not use_saved:
    
            filename = "saved_matrices.pkl"
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                    print(f"Файл {filename} удален")
                except Exception as e:
                    print(f"Ошибка при удалении файла {filename}: {e}")

        mode = request.form.get("mode")
        if not running:
            running = True
            threading.Thread(target=run_task, args=(mode,), daemon=True).start()

        return redirect(url_for('index'))
    
    return render_template('index.html', config=config, running=running)

def run_task(mode):
    global running, results_data
    try:
        print(f"Running {mode}...")
        if mode == "parallel":
            test_parallel(config["num_iters"], config)
        else:
            test_sequential(config["num_iters"], config)
        
        formatted = time.strftime("%d.%m.%Y", time.localtime())
        filename = f"{'p' if mode == 'parallel' else 's'} {formatted} elites({config['elite_number']}) Z({config['Z']}) K({config['K']}) M({config['M']}) N({config['N']}) iterations({config['num_iters']}).txt"
        
        with open(filename, 'r', encoding='utf-8') as f:
            results_data[mode] = f.read()
        
        result_queue.put(("success", f"{mode.capitalize()} выполнено успешно."))
    except Exception as e:
        result_queue.put(("error", str(e)))
    finally:
        running = False

@app.route("/progress")
def progress():
    messages = []
    while not result_queue.empty():
        msg_type, content = result_queue.get()
        messages.append({"type": msg_type, "content": content})
    return {"messages": messages, "running": running, "results": results_data}


if __name__ == "__main__":
    app.run(debug=True)