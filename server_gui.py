import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import subprocess
import time
import os
import socket

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

class FlowerServerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lung Cancer Detection Server - Ensemble")
        self.geometry("800x600")
        
        self.server_process = None
        self.server_running = False
        self.start_time = None
        
        control_frame = ttk.LabelFrame(self, text="Server Control")
        control_frame.pack(pady=10, padx=10, fill=tk.X)
        ttk.Label(control_frame, text="Port:").pack(side=tk.LEFT, padx=5)
        self.port_entry = ttk.Entry(control_frame, width=10)
        self.port_entry.insert(0, "8080")
        self.port_entry.pack(side=tk.LEFT, padx=5)
        self.start_button = ttk.Button(control_frame, text="Start Server", command=self.start_server)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(control_frame, text="Stop Server", command=self.stop_server, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        status_frame = ttk.LabelFrame(self, text="Server Status")
        status_frame.pack(pady=5, padx=10, fill=tk.X)
        self.status_label = ttk.Label(status_frame, text="Status: Not running")
        self.status_label.pack()
        self.time_label = ttk.Label(status_frame, text="Uptime: 0s | Rounds: 0")
        self.time_label.pack()
        
        log_frame = ttk.LabelFrame(self, text="Server Logs")
        log_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def start_server(self):
        if not self.server_running:
            port = self.port_entry.get()
            try:
                port = int(port)
                if not (1 <= port <= 65535):
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Invalid port number (1-65535)")
                return
                
            self.server_running = True
            self.start_time = time.time()
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            local_ip = get_local_ip()
            server_address = f"0.0.0.0:{port}"
            self.status_label.config(text=f"Status: Starting on {server_address} (Local IP: {local_ip})")
            self.log_text.insert(tk.END, f"Attempting to bind to port {port}...\n")
            
            threading.Thread(target=self.run_server, args=(port,), daemon=True).start()
            self.update_uptime()
    
    def stop_server(self):
        if self.server_running:
            self.server_running = False
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Stopping...")
            if self.server_process:
                self.server_process.terminate()
                self.server_process = None
            self.status_label.config(text="Status: Stopped")
            self.start_button.config(state=tk.NORMAL)
    
    def run_server(self, port):
        try:
            local_ip = get_local_ip()
            server_address = f"0.0.0.0:{port}"
            self.log_text.insert(tk.END, f"Starting server on {server_address} (Local IP: {local_ip})...\n")
            self.log_text.insert(tk.END, f"Clients should connect to {local_ip}:{port} with ensemble models\n")
            self.log_text.insert(tk.END, "Waiting for at least one client to connect (timeout: 60s)...\n")
            self.log_text.see(tk.END)
            self.server_process = subprocess.Popen(
                ["python", "flower_server.py", "--port", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            self.status_label.config(text=f"Status: Running on {server_address} (Local IP: {local_ip})")
            while self.server_running:
                output = self.server_process.stdout.readline()
                if output:
                    self.log_text.insert(tk.END, output)
                    self.log_text.see(tk.END)
                elif self.server_process.poll() is not None:
                    break
        except Exception as e:
            self.log_text.insert(tk.END, f"Error: {str(e)}. Check if port {port} is available or in use.\n")
            self.status_label.config(text="Status: Error")
        finally:
            self.server_running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Stopped")
    
    def update_uptime(self):
        if self.server_running:
            uptime = int(time.time() - self.start_time)
            self.time_label.config(text=f"Uptime: {uptime}s")
            self.after(1000, self.update_uptime)

if __name__ == "__main__":
    app = FlowerServerApp()
    app.mainloop()