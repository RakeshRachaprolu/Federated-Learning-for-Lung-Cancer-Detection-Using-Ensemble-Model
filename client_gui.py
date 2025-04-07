import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from client import start_client
import threading
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class LungCancerClientApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lung Cancer Client - Ensemble")
        self.geometry("600x500")
        
        conn_frame = ttk.LabelFrame(self, text="Server Connection")
        conn_frame.pack(pady=10, padx=10, fill=tk.X)
        ttk.Label(conn_frame, text="Server (IP:Port):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.server_entry = ttk.Entry(conn_frame)
        self.server_entry.insert(0, "192.168.29.108:8080")
        self.server_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Label(conn_frame, text="Use the server's IP (e.g., 192.168.29.108:8080)").grid(row=1, column=0, columnspan=2, padx=5, pady=2, sticky=tk.W)
        
        client_frame = ttk.LabelFrame(self, text="Client Configuration")
        client_frame.pack(pady=10, padx=10, fill=tk.X)
        ttk.Label(client_frame, text="Client ID:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.id_entry = ttk.Entry(client_frame)
        self.id_entry.insert(0, "1")
        self.id_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Label(client_frame, text="Data Path:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.data_entry = ttk.Entry(client_frame)
        self.data_entry.insert(0, "Client_Data_1")
        self.data_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10, padx=10, fill=tk.X)
        self.start_btn = ttk.Button(btn_frame, text="Start Client", command=self.start_client)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="Stop Client", command=self.stop_client, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        log_frame = ttk.LabelFrame(self, text="Client Log")
        log_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        conn_frame.columnconfigure(1, weight=1)
        client_frame.columnconfigure(1, weight=1)
    
    def start_client(self):
        server = self.server_entry.get()
        client_id = self.id_entry.get()
        data_path = self.data_entry.get()
        
        if not all([server, client_id, data_path]):
            messagebox.showerror("Error", "All fields are required")
            return
        if ':' not in server:
            messagebox.showerror("Error", "Server address must be in IP:Port format")
            return
            
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"Starting client {client_id} with ensemble models...\n")
        
        threading.Thread(
            target=self.run_client,
            args=(server, data_path, client_id),
            daemon=True
        ).start()
    
    def stop_client(self):
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.log_text.insert(tk.END, "Client stopped\n")
    
    def run_client(self, server, data_path, client_id):
        try:
            self.log_text.insert(tk.END, f"Connecting to {server} with retry attempts...\n")
            self.log_text.insert(tk.END, f"Data path: {data_path}, exists: {os.path.exists(data_path)}\n")
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            start_client(server, data_path, client_id)
            
            output = sys.stdout.getvalue()
            self.log_text.insert(tk.END, output)
            self.log_text.insert(tk.END, f"Client {client_id} completed successfully\n")
            self.log_text.see(tk.END)
            
        except Exception as e:
            self.log_text.insert(tk.END, f"Error in client {client_id}: {str(e)}. Check data path, server address, and network.\n")
        finally:
            sys.stdout = old_stdout
            self.stop_client()

if __name__ == "__main__":
    app = LungCancerClientApp()
    app.mainloop()