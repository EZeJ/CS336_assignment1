import debugpy

def wait_for_debugger(port: int = 5678, host: str = "localhost") -> None:
    """
    Wait for a debugger to attach on the given host and port using debugpy.

    Args:
        port (int): Port to listen on. Default is 5678.
        host (str): Host to bind to. Default is "localhost".
    """
    print(f"Waiting for debugger to attach on {host}:{port}...")
    debugpy.listen((host, port))
    debugpy.wait_for_client()
    print("Debugger attached.")