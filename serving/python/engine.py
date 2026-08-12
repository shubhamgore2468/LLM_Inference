import queue
import threading
import time
from typing import Any, Callable, Dict, List, Tuple
from runner import Engine

EventCallback = Callable[[Dict[str, Any]], None]

class EngineThread:

    def __init__(self, model_path, **kwargs):

        """
        Manages a stateful ML engine in a dedicated worker thread.
        Bridges the gap between concurrent HTTP reqs and a sync single threaded engine loop using non blocking queues
        """

        self.engine = Engine(model_path, **kwargs)
        self.cfg = self.engine.cfg
        self.inbox_queue = queue.Queue()
        self.callbacks = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.iteration_count = 0
        # Background worker lifecycle
        self.worker_thread = threading.Thread(
                target = self._run,
                daemon=True,
                name="engine_worker"
                )
        self.worker_thread.start()

    def submit_request(
            self,
            token_ids: List[int],
            max_tokens: int,
            ignore_eos: bool,
            on_event: EventCallback
            ):
        """ Submits a new token generation req to engine.
            Blocks the HTTP thread briefly until the engine registers the seq and returns a unique seq ID.
        """
        response_box: Dict[str, int] = {}
        registration_ready_event = threading.Event()

        # Safely pass parameters and synchronization primitives to the worker thread
        self.inbox_queue.put((
            token_ids,
            max_tokens,
            ignore_eos,
            on_event,
            response_box,
            registration_ready_event
            ))
        # Wait until engine thread assigns a seq ID 

        registration_ready_event.wait()

        return response_box["sequence_id"]


    def get_metrics(self) -> Dict[str, Any]:
        """Fetches a snapshot of current engine and operational stats"""

        stats_snapshot = self.engine.stats()

        return {
            "waiting_sequences": stats_snapshot.waiting,
            "running_sequences": stats_snapshot.running,
            "in_flight_tokens": stats_snapshot.in_flight_tokens,
            "kv_cache_utilization": stats_snapshot.kv_util,
            "total_finished": stats_snapshot.total_finished,
            "total_preempted": stats_snapshot.total_preempted,
            "prefix_hit_blocks": stats_snapshot.prefix_hit_blocks,
            "prefill_blocks": stats_snapshot.prefill_blocks,
            "loop_iterations": self.iteration_count,
                }

    def stop(self) -> None:
        """ Signals the background loop to stop and waits for the thread to exit"""

        self.stop_event.set()
        self.worker_thread.join(timeout=5)

    def _run(self) -> None:
        """ Main execution loop for processing and advancing model reqs"""

        while not self.stop_event.is_set():
            self._drain_incoming_requests()

            # Avoid high CPU utilization when there are not active seqs 

            if not self.engine.sched.has_work():
                time.sleep(0.0005)
                continue

            # Advance the model by one step and increment iteration counter

            generated_tokens = self.engine.step(verbose=False)

            self.iteration_count += 1

            self._emit_events(generated_tokens)


    def _drain_incoming_requests(self) -> None:

        """ Empties the inbox queue and registers all new requests into the engine"""

        while True:

            try:
                (
                    token_ids, 
                    max_tokens, 
                    ignore_eos, 
                    callback, 
                    response_box, 
                    registration_ready_event
                ) = self.inbox_queue.get_nowait()
            except queue.Empty:
                return

            # Register request into the synchronous engine scheduler
            sequence_id = self.engine.add(token_ids, max_tokens=max_tokens, ignore_eos=ignore_eos)
            
            # Map the seq ID to its callback func 
            with self.lock:
                self.callbacks[sequence_id] = callback

            # Hand the ID back to waiting HTTP thread

            response_box["sequence_id"] = sequence_id
            registration_ready_event.set()


    
    def _emit_events(self, generated_tokens: List[Tuple[int, Any]]) -> None:
        """Dispatches generated tokens and cleanup events to their respective callbacks."""
        # 1. Stream newly generated tokens
        for sequence_id, token in generated_tokens:
            with self.lock:
                callback = self.callbacks.get(sequence_id)
            if callback:
                callback({"token": token})

        # 2. Clean up sequences that finished during this step
        for sequence_id in self.engine.sched.take_finished():
            with self.lock:
                callback = self.callbacks.pop(sequence_id, None)
            if callback:
                callback({"done": True})

