"""Own and clean up the learner/actor process groups for one experiment."""
import os
from pathlib import Path
import signal
import subprocess
import time


class StopRequested(Exception):
    def __init__(self, signum):
        self.signum = signum


def stop_handler(signum, frame):
    signal.signal(signum, signal.SIG_IGN)
    raise StopRequested(signum)


def stop_workers(workers):
    for process in workers.values():
        try:
            os.killpg(process.pid, signal.SIGCONT)
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if all(process.poll() is not None for process in workers.values()):
            break
        time.sleep(0.05)
    for process in workers.values():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def supervise(commands, log_dir, environment, cwd, timeout=None):
    """Return when learner succeeds; actor exits and timeouts are failures."""
    workers, handles = {}, []
    previous = {sig: signal.signal(sig, stop_handler) for sig in
                (signal.SIGTERM, signal.SIGHUP)}
    try:
        for name, command in commands.items():
            handle = (Path(log_dir) / (name + ".log")).open("w")
            handles.append(handle)
            workers[name] = subprocess.Popen(
                command, env=environment, cwd=str(cwd), stdout=handle,
                stderr=subprocess.STDOUT, start_new_session=True)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            status = workers["learner"].poll()
            if status is not None:
                if status:
                    raise RuntimeError("Learner exited with code %d; see learner.log" % status)
                return
            for name, process in workers.items():
                if name != "learner" and process.poll() is not None:
                    raise RuntimeError("%s exited before learner completion; see %s.log" % (name, name))
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Run exceeded timeout of %s seconds" % timeout)
            time.sleep(0.2)
    finally:
        stop_workers(workers)
        for handle in handles:
            handle.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
