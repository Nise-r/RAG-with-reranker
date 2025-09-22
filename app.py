from flask import Flask, render_template, request, jsonify
import os
import time
from langgraph.types import Command, interrupt
from werkzeug.utils import secure_filename
from qna import agent, log_messages, ret_chunks  

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf'}

#global var
# pdf_path = None
pdf_path = []
is_interrupted = False
chat_history = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global pdf_path
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid or unsupported file'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(save_path)
        
        print("="*10)
        print(f"[UPLOAD] File saved at: {save_path}")
        print("="*10)
        
        pdf_path.append(save_path)
        
        return jsonify({'filepath': save_path})
    except Exception as e:
        return jsonify({'error': f'File save failed: {str(e)}'}), 500
    
@app.route('/logs', methods=['GET'])
def get_logs():
    global log_messages
    logs_to_send = log_messages.copy()
    log_messages.clear()  # Optional: clear after sending
    return jsonify({'logs': logs_to_send})

@app.route('/chunks', methods=['GET'])
def get_chunks():
    global ret_chunks
    chunks_to_send = ret_chunks.copy()
    ret_chunks.clear()  # Optional: clear after sending
    return jsonify({'chunks': chunks_to_send})

@app.route('/ask', methods=['POST'])
def ask():
    global pdf_path, is_interrupted, chat_history
    
    data = request.get_json()
    question = data.get('question', '').strip()
#     pdf_path = data.get('filepath', None)
    path = pdf_path
    pdf_path = []
    
    if not question:
        return jsonify({'answer': '❗ Please enter a question.'}), 400
    
    try:
        thread_config = {"configurable": {"thread_id": "some_id"}}
        
        print("="*10)
        print(f"[ASK] Question: {question}")
        print(f"[ASK] PDF Path: {path}")
        print("="*10)
        
        if not is_interrupted:
            result = agent.invoke({
                "query": question,
                "k":6,
                "pdf_path": path, 
                "result": "",
                "imgs": [],
                "paper_url": None,
                "next_node": None,
                "chat_history":chat_history
            }, config=thread_config)
            state = agent.get_state(thread_config)
            
            #delay for log messages
            time.sleep(1)
            if state and state.tasks and state.tasks[0].interrupts:
                is_interrupted = True
                return jsonify({'answer':f"{result['result']}\n\n{state.tasks[0].interrupts[0].value['query']}"})
            
            chat_history = result['chat_history']
            return jsonify({'answer': result['result']})
        else:
            result = agent.invoke(Command(resume=question),config=thread_config)
            is_interrupted = False
            
            chat_history = result['chat_history']
            return jsonify({'answer': result['result']})
    except Exception as e:

        return jsonify({'answer': f"❌ Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
