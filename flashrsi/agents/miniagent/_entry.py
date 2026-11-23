"""Private process bridge. This adapter is deliberately outside Snapshot rules."""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path


async def _main():
    # The adapter gives native tools a chance to reap their own subprocesses.
    if os.name == "posix":
        asyncio.get_running_loop().add_signal_handler(
            signal.SIGTERM, asyncio.current_task().cancel
        )
    request = json.load(sys.stdin)
    # Reserve stdout for the result; harness/tool output goes to stderr.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8") as result_stream:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        # -I ignores parent PYTHONPATH. Expose only the trusted bridge package
        # and the selected Snapshot's runtime, even in the lean container.
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        sys.path.insert(0, request["workspace_dir"])
        from runtime.agent import run

        result = await run(**request)
        print(json.dumps(result, ensure_ascii=False), file=result_stream)


if __name__ == "__main__":
    asyncio.run(_main())
