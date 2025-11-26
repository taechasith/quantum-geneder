import sys      # lets us access command-line args and exit the program cleanly
import os       # used to walk through folders to find dataset files
import io       # used to store images in memory as if they were files
import random   # used to randomly choose one label file from many
import numpy as np  # used for numerical arrays, math, and data processing
import kagglehub    # used to download the dataset from Kaggle by its handle

# Scientific & Quantum Libraries      
from qiskit import QuantumCircuit, transpile     # QuantumCircuit builds the quantum logic; transpile adapts it to a backend
from qiskit_aer import AerSimulator             # AerSimulator runs the quantum circuit on a classical simulator

# Visualization Libraries      
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel  
# QApplication = main app, QMainWindow = main window frame, QWidget = basic container,
# QHBoxLayout/QVBoxLayout = layout managers, QLabel = text/image display.

from PyQt6.QtGui import QPixmap, QImage, QFont   # QImage/QPixmap to show images, QFont to style text
from PyQt6.QtCore import Qt                      # Qt core flags (alignment, text mode, etc.)
import pyqtgraph.opengl as gl                    # OpenGL 3D plotting for the scatter cloud
from pycirclize import Circos                    # used to make the circular Normal/Abnormal diagram
import matplotlib.pyplot as plt                  # plotting backend used by Circos to create a Figure

# --- 1. QUANTUM SIMULATION BACKEND ---      
class SpermQuantumSimulation:      # class that handles data loading and quantum simulation based on that data
    def __init__(self, dataset_handle="orvile/mhsma-sperm-morphology-analysis-dataset"):      
        # __init__ runs when we create SpermQuantumSimulation(); sets default state
        self.dataset_handle = dataset_handle     # remember which Kaggle dataset to download
        self.path = None                         # will later store local path to downloaded dataset
        self.prob_normal = 0.5                   # default P(normal) if dataset fails
        self.prob_abnormal = 0.5                 # default P(abnormal) if dataset fails
        self.selected_file = "Unknown"           # placeholder to show which label file was used

    def fetch_and_process(self):      # method to download data, find labels, and compute probabilities
        print("--- Initializing Biological Data ---")  # console log so user knows what's happening
        try:      
            self.path = kagglehub.dataset_download(self.dataset_handle)  # download dataset and get its local folder path
                  
            # Find all label files      
            label_files = []                     # list to collect all label file paths
            for root, _, files in os.walk(self.path):   # walk through all subdirectories in the dataset folder
                for file in files:      
                    if file.startswith("y_") and file.endswith(".npy"):  # filter only y_*.npy label files
                        label_files.append(os.path.join(root, file))     # add full path to list
                  
            if not label_files:                             # if we found no matching label files
                raise FileNotFoundError("No 'y_*.npy' files found.")  # explicitly fail so we go to except

            # Random selection      
            selected_path = random.choice(label_files)   # pick one label file at random for this run      
            self.selected_file = os.path.basename(selected_path)  # keep only the filename for display
                  
            # Load and Force Binary (0 or 1)      
            raw_labels = np.load(selected_path)          # load label array from disk      
            labels = (raw_labels > 0).astype(int)        # convert any positive value to 1, others to 0 (binary labels)
                  
            total = len(labels)                          # total number of samples in this file
            abnormal = np.sum(labels)                    # since labels are 0/1, sum = count of abnormal (1)
            self.prob_normal = (total - abnormal) / total  # biological probability of normal
            self.prob_abnormal = abnormal / total          # biological probability of abnormal
                  
            print(f"File: {self.selected_file}")         # show which file was used
            print(f"Bio-Probs: Normal={self.prob_normal:.2%}, Abnormal={self.prob_abnormal:.2%}")  # show computed probabilities
      
        except Exception as e:      # any error in download/processing comes here
            print(f"Data Error: {e}. Using default 50/50.")  # tell user and keep default 0.5/0.5

    def run_simulation(self, shots=2000):      # method to run quantum circuit based on computed probabilities
        # Prevent math domain errors      
        p_norm = max(0.0, min(1.0, self.prob_normal))   # ensure probability stays in [0,1] before sqrt/arccos
              
        alpha = np.sqrt(p_norm)                         # amplitude for |0> (normal) in quantum state
        theta = 2 * np.arccos(alpha)                    # rotation angle that encodes this probability via Ry gate
              
        qc = QuantumCircuit(1)                          # create a circuit with 1 qubit
        qc.ry(theta, 0)                                 # rotate qubit 0 around Y-axis by theta (encodes p_norm)
        qc.measure_all()                                # add measurement so we can get classical results
      
        simulator = AerSimulator()                      # choose AerSimulator as backend
        compiled_qc = transpile(qc, simulator)          # compile the circuit into a form optimized for this simulator
        result = simulator.run(compiled_qc, shots=shots).result()  # run the circuit many times and get result object
        return result.get_counts()                      # return dict like {'0': count, '1': count}
      
      
# --- 2. ADVANCED VISUALIZATION FRONTEND ---      
class ModernVisualizer(QMainWindow):      # main GUI window that shows 3D cloud + stats + circular diagram
    def __init__(self, simulation_results, filename, probs):      
        # __init__ sets up the window layout and visual elements
        super().__init__()      
        self.setWindowTitle(f"Quantum Sperm Genomics - {filename}")  # set window title including dataset file
        self.resize(1200, 700)                                       # set initial window size
        self.setStyleSheet("background-color: #1e1e1e; color: white;")  # global dark theme for the window
      
        # Data Parsing      
        self.total_shots = sum(simulation_results.values())  # total number of simulated measurements
        self.count_normal = (      
            simulation_results.get('0', 0) +        # count normal outcomes from string key '0'
            simulation_results.get('0x0', 0) +      # add weird variant key '0x0' if present
            simulation_results.get(0, 0)            # add numeric key 0 if present
        )      
        self.count_abnormal = self.total_shots - self.count_normal  # remaining shots treated as abnormal
        self.probs = probs                                          # store biological probabilities (normal, abnormal)
        self.filename = filename                                    # store dataset file name for display
      
        # --- Layout ---      
        central_widget = QWidget()                     # central content container for the main window
        self.setCentralWidget(central_widget)          # register it as the main central widget
        layout = QHBoxLayout()                         # layout that arranges widgets horizontally (left/right)
        central_widget.setLayout(layout)               # apply that layout to the central widget
      
        # --- LEFT PANEL: 3D View + Stats ---      
        left_panel = QWidget()                         # container for left side: 3D + stats
        left_layout = QVBoxLayout()                    # vertical layout for stacking 3D view on top of stats
        left_panel.setLayout(left_layout)              # attach vertical layout to the left panel
              
        # 1. OpenGL View      
        self.gl_view = gl.GLViewWidget()               # create a 3D OpenGL view widget
        self.gl_view.setBackgroundColor('#101010')     # set dark background for the 3D area
        self.gl_view.setCameraPosition(distance=45, elevation=30)  # set camera distance and angle for better view
              
        g = gl.GLGridItem()                            # create a 3D grid reference plane
        g.setSize(x=40, y=40)                          # set overall grid size
        g.setSpacing(x=2, y=2)                         # set spacing between grid lines
        self.gl_view.addItem(g)                        # add the grid to the 3D view
              
        self.create_gl_cloud()                         # generate and add the scatter of samples to the 3D scene
        left_layout.addWidget(self.gl_view, stretch=3) # add 3D view to left panel, taking more vertical space
      
        # 2. UPDATED Stats Label      
        # Calculations      
        sim_n_pct = (self.count_normal / self.total_shots) * 100   # simulated percentage normal
        sim_a_pct = (self.count_abnormal / self.total_shots) * 100 # simulated percentage abnormal
        bio_n_pct = self.probs[0] * 100                            # biological percentage normal
        bio_a_pct = self.probs[1] * 100                            # biological percentage abnormal
              
        # Deviation (Quantum Noise)      
        diff_n = sim_n_pct - bio_n_pct             # difference between simulated and biological normal %
      
        stats_text = (                             # build an HTML string for the stats report
            f"<h3 style='color: white;'>QUANTUM ANALYSIS REPORT</h3>"      
            f"<b>Source Dataset:</b> {self.filename}<br>"      
            f"<b>Total Samples:</b> {self.total_shots}<br>"      
            f"--------------------------------------------------<br>"      
                  
            f"<b style='color: #00FFFF; font-size: 14px;'>■ NORMAL (Cyan)</b><br>"      
            f"&nbsp;&nbsp;Measured Count: <b>{self.count_normal}</b><br>"      
            f"&nbsp;&nbsp;Simulated Ratio: {sim_n_pct:.2f}%<br>"      
            f"&nbsp;&nbsp;Biological Input: {bio_n_pct:.2f}%<br>"      
            f"&nbsp;&nbsp;<i>Quantum Deviation: {diff_n:+.2f}%</i><br><br>"      
                  
            f"<b style='color: #FF4444; font-size: 14px;'>■ ABNORMAL (Red)</b><br>"      
            f"&nbsp;&nbsp;Measured Count: <b>{self.count_abnormal}</b><br>"      
            f"&nbsp;&nbsp;Simulated Ratio: {sim_a_pct:.2f}%<br>"      
            f"&nbsp;&nbsp;Biological Input: {bio_a_pct:.2f}%"      
        )      
      
        stats_label = QLabel(stats_text)              # QLabel to display the HTML stats text
        stats_label.setFont(QFont("Segoe UI", 10))    # set font family and size
        stats_label.setStyleSheet("background-color: #252526; padding: 15px; border-top: 2px solid #333;")  
        # above line: style the stats block with padding and a top border

        stats_label.setTextFormat(Qt.TextFormat.RichText)  # tell QLabel to treat text as HTML, not plain text
              
        left_layout.addWidget(stats_label, stretch=1) # place stats under 3D view in left panel
        layout.addWidget(left_panel, stretch=2)       # add left panel to main horizontal layout (wider side)
      
        # --- RIGHT PANEL: Circular Plot ---      
        self.image_label = QLabel()                   # label to display the circular diagram image
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # center image within the label
        self.create_circlize_plot()                   # generate circular diagram and set it on image_label
        layout.addWidget(self.image_label, stretch=1) # add right panel to main layout (narrower side)
      
    def create_gl_cloud(self):      # helper to build and add the 3D scatter point cloud
        """Renders the 3D scatter plot."""      
        pos = np.random.normal(size=(self.total_shots, 3))  # generate random (x,y,z) positions for each shot
        pos[:, 0] *= 5        # widen cloud along x-axis
        pos[:, 1] *= 5        # widen cloud along y-axis
        pos[:, 2] *= 0.5      # flatten cloud along z-axis
              
        colors = np.zeros((self.total_shots, 4))      # RGBA color array for each point, initially all zeros
              
        if self.count_normal > 0:      
            colors[:self.count_normal] = (0, 1, 1, 0.8)  # first normal points in array: cyan
        if self.count_abnormal > 0:      
            colors[self.count_normal:] = (1, 0.3, 0.3, 0.8)  # remaining points: red
      
        np.random.shuffle(pos)                        # shuffle positions so the cloud looks more mixed visually
      
        sp = gl.GLScatterPlotItem(pos=pos, color=colors, size=6, pxMode=True)  
        # GLScatterPlotItem: 3D points using our positions and colors

        sp.setGLOptions('translucent')                # enable transparency rendering
        self.gl_view.addItem(sp)                      # add scatter plot to the 3D view
      
    def create_circlize_plot(self):      # helper to create the circular Normal/Abnormal diagram
        """Generates Circular Diagram."""      
        n_pct = (self.count_normal / self.total_shots) * 100  # normal percentage
        a_pct = (self.count_abnormal / self.total_shots) * 100  # abnormal percentage
      
        sectors = {f"Normal\n{n_pct:.1f}%": n_pct, f"Abnormal\n{a_pct:.1f}%": a_pct}  
        # sectors dict: label text -> sector size value

        circos = Circos(sectors, space=5)                       # build circular layout with small gaps
      
        for sector in circos.sectors:      
            color = "#00FFFF" if "Normal" in sector.name else "#FF4444"  # choose cyan for Normal, red for Abnormal
            track1 = sector.add_track((90, 100))               # outer track ring
            track1.axis(fc=color, alpha=0.6)                   # fill that ring with semi-transparent color
            track1.text(sector.name, color="white", size=10)   # draw sector label text on ring
      
            track2 = sector.add_track((60, 85))                # inner track for decorative bars
            x = np.linspace(sector.start, sector.end, 40)      # angles for bar positions
            y = np.random.randint(20, 100, 40)                 # random bar heights (decorative, not data-driven)
            track2.bar(x, y, color=color, alpha=0.8)           # draw the bars within this sector
      
        fig = circos.plotfig()                                 # render Circos diagram to a Matplotlib figure
        fig.patch.set_facecolor('#1e1e1e')                     # set figure background to dark
      
        buf = io.BytesIO()                                     # create in-memory buffer for image bytes
        fig.savefig(buf, format='png', facecolor='#1e1e1e', dpi=100, bbox_inches='tight')  
        # save figure as PNG into the buffer

        buf.seek(0)                                            # move buffer cursor back to the start
        plt.close(fig)                                         # close Matplotlib figure to free memory
      
        qimg = QImage.fromData(buf.getvalue())                 # load PNG bytes into a QImage
        pixmap = QPixmap.fromImage(qimg)                       # convert QImage to a QPixmap for display
        self.image_label.setPixmap(pixmap)                     # put pixmap into the label on the right panel
      
# --- 3. EXECUTION ---      
if __name__ == "__main__":      # this block runs only when this file is executed directly, not imported
    app = QApplication(sys.argv)                               # create Qt application with command-line arguments
    sim = SpermQuantumSimulation()                             # create simulation backend object
    sim.fetch_and_process()                                    # download dataset and compute bio probabilities
    counts = sim.run_simulation(shots=2000)                    # run quantum simulation to get measurement counts
    window = ModernVisualizer(counts, sim.selected_file, (sim.prob_normal, sim.prob_abnormal))  
    # create GUI window with simulation results and probabilities

    window.show()                                              # show the main window
    sys.exit(app.exec())                                       # start Qt event loop, exit program when window closes
