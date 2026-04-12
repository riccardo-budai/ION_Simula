import threading
import time
import numpy as np
import zmq
import json
import signal
import sys
from multiprocessing import Process, Value, Queue, Manager
from queue import Empty
from pylsl import StreamInfo, StreamOutlet, local_clock

# Global flag for clean exit
RUNNING = True


def signal_handler(sig, frame):
    """Handles CTRL+C by setting the global flag to False"""
    global RUNNING
    print("\n[SYSTEM] Detected CTRL+C. Starting shutdown procedure...")
    RUNNING = False


# --- ZMQ LISTENER PROCESS ---
def zmq_listener_process(main_queue, evoked_queues, port=5555, controller_ip="localhost"):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    try:
        socket.connect(f"tcp://{controller_ip}:{port}")
        socket.subscribe("GEN/")
        socket.subscribe("SYS/")
        socket.subscribe("STIM/")
        print(f"[ZMQ LISTENER] Listening on {controller_ip}:{port}...")
    except Exception as e:
        print(f"[ZMQ ERROR] Connection failed: {e}")
        return

    while True:
        try:
            if socket.poll(500):
                msg_string = socket.recv_string()
                try:
                    topic, json_str = msg_string.split(' ', 1)
                    payload = json.loads(json_str)['data']
                except ValueError:
                    continue

                if topic == "SYS/LOAD_CONFIG":
                    print(f"[CMD] Config Received: {len(payload.get('active_modules', []))} modules.")
                    main_queue.put(("LOAD_CONFIG", payload))

                elif topic == "SYS/SHUTDOWN":
                    print("[CMD] Remote shutdown requested.")
                    main_queue.put(("SHUTDOWN", None))
                    break

                elif topic == "GEN/VITALS":
                    main_queue.put(("UPDATE_VITALS", payload))

                elif topic.startswith("STIM/"):
                    stim_type = topic.split("/")[1]  # Es. "SEP" da "STIM/SEP"

                    # 1. Controllo diretto
                    if stim_type in evoked_queues:
                        evoked_queues[stim_type].put(payload)
                        print(f"[TRIG] Stimulus queued: {stim_type}")

                    # 2. Controllo di compatibilità (es. se arriva "SEP_UL" ma la coda è "SEP")
                    elif "SEP" in stim_type and "SEP" in evoked_queues:
                        evoked_queues["SEP"].put(payload)
                        print(f"[TRIG] Stimulus queued (Generic): SEP")

                    elif "MEP" in stim_type and "MEP" in evoked_queues:
                        evoked_queues["MEP"].put(payload)
                        print(f"[TRIG] Stimulus queued (Generic): MEP")

        except zmq.ContextTerminated:
            break
        except Exception:
            pass


# --- ZMQ TELEMETRY SENDER ---
def telemetry_sender(active_streams_list, tutor_ip="192.168.50.100", port=5556):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    try:
        socket.connect(f"tcp://{tutor_ip}:{port}")
    except:
        pass

    while RUNNING:
        try:
            current_streams = list(active_streams_list)
            report = {
                "sender": "VIRTUAL_PATIENT",
                "type": "STATUS_REPORT",
                "timestamp": time.time(),
                "data": {
                    "status": "RUNNING" if current_streams else "IDLE",
                    "active_lsl": current_streams
                }
            }
            socket.send_json(report)
        except Exception:
            pass
        time.sleep(2.0)


# --- LSL GENERATOR PROCESS ---
class LSLGenerator(Process):
    def __init__(self, name, type_stream, channels, srate, shared_anesthesia, is_continuous=True, input_queue=None):
        super().__init__()
        self.name_stream = name
        self.type_stream = type_stream
        self.channels = channels
        self.n_channels = len(channels)
        self.srate = srate
        self.shared_anesthesia = shared_anesthesia
        self.is_continuous = is_continuous
        self.input_queue = input_queue

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            fmt = 'string' if self.type_stream == 'Markers' else 'float32'
            info = StreamInfo(self.name_stream, self.type_stream, self.n_channels, self.srate, fmt,
                              f'uid_{self.name_stream}')
            chns = info.desc().append_child("channels")
            for ch in self.channels:
                chns.append_child("channel").append_child_value("label", ch)

            outlet = StreamOutlet(info)
            print(f"   >>> Stream STARTED: {self.name_stream} [{self.type_stream}]")

            if self.is_continuous:
                self._run_continuous(outlet)
            else:
                self._run_event_based(outlet)
        except Exception as e:
            print(f"Process {self.name_stream} error: {e}")

    def _run_continuous(self, outlet):
        chunk_size = int(self.srate * 0.1) or 1
        next_time = local_clock()
        while True:
            depth = self.shared_anesthesia.value
            amp = 1.0 - (depth * 0.5)
            data = np.random.randn(chunk_size, self.n_channels) * amp
            outlet.push_chunk(data.tolist())

            next_time += (chunk_size / self.srate)
            sleep_dur = next_time - local_clock()
            if sleep_dur > 0: time.sleep(sleep_dur)

    def _run_event_based(self, outlet):
        """Attende trigger dalla coda e spara uno sweep."""
        print(f"   [{self.name_stream}] Waiting for triggers...")
        while True:
            if self.name_stream == "IOM_Vitals":
                hr = 60 + np.random.normal(0, 2)
                sample = [hr, 98, 120, 80, 36.5, 45, 2.5, 3.0, 0.0, 0.0]
                outlet.push_sample(sample)
                time.sleep(1.0)

            elif self.type_stream in ['BAEP', 'SEP', 'MEP', 'VEP']:
                if self.input_queue:
                    try:
                        # Attesa bloccante con timeout per permettere al ciclo di girare e controllare kill signals
                        _ = self.input_queue.get(timeout=1.0)

                        # Trigger Ricevuto! Genera Sweep
                        n_samples = int(0.05 * self.srate)  # 50ms
                        chunk = np.random.randn(n_samples, self.n_channels).tolist()
                        outlet.push_chunk(chunk)
                        print(f"      -> {self.name_stream} FIRED! ({n_samples} samples)")
                    except Empty:
                        pass  # Nessun trigger, continua

            elif self.type_stream == 'Markers':
                time.sleep(1)


# --- HELPER: CONFIGURATION MAPPER ---
def create_processes_from_config(config_data, shared_anesthesia, queues):
    modules = config_data.get("active_modules", [])
    procs = []

    # 1. EEG
    if "EEG" in modules:
        procs.append(LSLGenerator("BioSemi_EEG", "EEG", ["Fp1", "Fp2", "C3", "C4", "O1", "O2"], 512, shared_anesthesia))

    # 2. ECOG
    if "ECOG" in modules:
        procs.append(
            LSLGenerator("Cortical_Strip", "ECOG", ["Strip1", "Strip2", "Strip3", "Strip4"], 512, shared_anesthesia))

    # 3. EMG
    if any(m in modules for m in ["EMG_FREE", "EMG_TRIGGERED", "EMG_CRANIAL"]):
        emg_labels = ["Tib_L", "Tib_R", "AbdHall_L", "AbdHall_R", "OrbOc_L", "OrbOc_R"]
        procs.append(LSLGenerator("Spine_Box_EMG", "EMG", emg_labels, 2048, shared_anesthesia))

    # 4. DWAVE
    if "DWAVE" in modules:
        procs.append(LSLGenerator("D_Wave_Catheter", "DWAVE", ["D-Wave"], 2000, shared_anesthesia))

    # 5. VITALS
    if "VIRTUAL_PATIENT" in modules or "ANESTHESIA" in modules:
        vitals_ch = ["HR", "SpO2", "NIBP_Sys", "NIBP_Dia", "Temp", "BIS", "Prop", "Remi", "Ket", "Sevo"]
        procs.append(LSLGenerator("IOM_Vitals", "VITALS", vitals_ch, 1, shared_anesthesia, is_continuous=False))
        procs.append(LSLGenerator("Patient_Monitor_Wave", "EKG_RESP", ["ECG_II", "Resp"], 500, shared_anesthesia))

    # 6. EVOKED POTENTIALS
    # Passiamo le CODE GESTITE DAL MANAGER
    if "BAEP" in modules:
        procs.append(LSLGenerator("Evoked_BAEP", "BAEP", ["A1", "A2"], 10000, shared_anesthesia, False, queues["BAEP"]))

    if "VEP" in modules:
        procs.append(LSLGenerator("Evoked_VEP", "VEP", ["O1", "O2"], 1000, shared_anesthesia, False, queues["VEP"]))

    if "SEP_UpperLimb" in modules or "SEP_UL" in modules:
        procs.append(
            LSLGenerator("Evoked_SEP_UL", "SEP", ["C3'", "C4'", "Erb_L", "Erb_R"], 5000, shared_anesthesia, False,
                         queues["SEP"]))

    if "SEP_LowerLimb" in modules or "SEP_LL" in modules:
        procs.append(
            LSLGenerator("Evoked_SEP_LL", "SEP", ["Cz'", "Fz", "Pop_L", "Pop_R"], 2000, shared_anesthesia, False,
                         queues["SEP"]))

    if any(x in modules for x in ["MEP_UpperLimb", "MEP_LowerLimb", "MEP_CONTRA", "MEP_FOUR_LIMB"]):
        procs.append(
            LSLGenerator("Evoked_MEP", "MEP", ["Thenar_L", "Thenar_R", "Tib_L", "Tib_R"], 5000, shared_anesthesia,
                         False, queues["MEP"]))

    # 7. MARKERS
    procs.append(LSLGenerator("IOM_Markers", "Markers", ["Events"], 0, shared_anesthesia, is_continuous=False))

    return procs


# --- MAIN ORCHESTRATOR ---
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    manager = Manager()
    anesthesia_level = Value('d', 0.0)

    # [FIX CRITICO] Uso manager.Queue() invece di Queue() normale.
    # Le Queue normali si corrompono se il processo lettore viene terminato (terminate()).
    # Le Manager Queue sono residenti in un processo server separato e sopravvivono ai kill.
    queues = {
        "BAEP": manager.Queue(),
        "SEP": manager.Queue(),
        "MEP": manager.Queue(),
        "VEP": manager.Queue()
    }

    # Anche la coda principale può essere managed per sicurezza, ma qui va bene normale
    # perché il lettore (Main) non viene mai killato.
    main_command_queue = Queue()

    active_stream_names = manager.list()
    active_procs = []

    TUTOR_IP = "localhost"

    p_listener = Process(target=zmq_listener_process, args=(main_command_queue, queues, 5555, TUTOR_IP))
    p_listener.daemon = True
    p_listener.start()

    t_telemetry = threading.Thread(target=telemetry_sender, args=(active_stream_names, TUTOR_IP, 5556))
    t_telemetry.daemon = True
    t_telemetry.start()

    print("--- VIRTUAL PATIENT GENERATOR READY ---")
    print(f">>> Listening on {TUTOR_IP}:5555 for CONFIG...")

    while RUNNING:
        try:
            cmd_type, payload = main_command_queue.get(timeout=0.1)

            if cmd_type == "LOAD_CONFIG":
                print("\n[MAIN] Analyzing new configuration...")

                potential_procs = create_processes_from_config(payload, anesthesia_level, queues)
                desired_stream_names = set(p.name_stream for p in potential_procs)
                current_stream_names = set(active_stream_names)

                if desired_stream_names == current_stream_names and len(active_procs) > 0:
                    print(f"   -> Configuration identical to active streams.")
                    print("   -> SKIPPING RESTART.")
                    for p in potential_procs: p.terminate()
                else:
                    print("   -> Configuration CHANGED. Reconfiguring...")

                    if active_procs:
                        print("   -> Stopping active streams...")
                        for p in active_procs:
                            p.terminate()
                            p.join()
                        active_procs = []
                        del active_stream_names[:]

                    if potential_procs:
                        for p in potential_procs:
                            p.start()
                            active_procs.append(p)
                            active_stream_names.append(p.name_stream)
                        print(f"   -> {len(active_procs)} new streams started.")
                    else:
                        print("   -> WARNING: No streams matched configuration.")

            elif cmd_type == "UPDATE_VITALS":
                hr = payload.get('hr', 70)
                anesthesia_level.value = 0.8 if hr < 50 else 0.0

            elif cmd_type == "SHUTDOWN":
                print("[MAIN] Shutdown requested via ZMQ.")
                RUNNING = False
                break

        except Empty:
            continue
        except KeyboardInterrupt:
            RUNNING = False
            break

    print("\n[SYSTEM] Cleaning up processes...")
    for p in active_procs:
        if p.is_alive():
            p.terminate()
            p.join()

    if p_listener.is_alive():
        p_listener.terminate()
        p_listener.join()

    print("Bye.")