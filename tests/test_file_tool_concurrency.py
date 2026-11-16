"""File tools exercised through independent Session bindings and real file I/O."""

import asyncio
import multiprocessing
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_tools import approve, bound, call, diff

from daphnis.tools import ApprovalDecision, files

NAME = "shared.txt"
OLD = "old\n"
FIRST, SECOND = "version A\n", "version B\n"
TIMEOUT = 10


def wait(event):
    if not event.wait(TIMEOUT):
        raise TimeoutError("Timed out waiting for the other file operation")


async def execute(workspace, name, *, approval=approve, **arguments):
    return (
        await bound(workspace, approval=approval).execute(call(name, **arguments))
    ).value


def write(workspace, name, text, *, approval=approve, path=NAME):
    if name == "write_file":
        arguments = {"path": path, "content": text}
    elif name == "replace_text":
        arguments = {"path": path, "old_text": OLD, "new_text": text}
    else:
        arguments = {"patch": diff(path, OLD, text)}
    return asyncio.run(execute(workspace, name, approval=approval, **arguments))


def os_hooks(**hooks):
    return patch.object(files, "os", SimpleNamespace(**(vars(os) | hooks)))


def watch_contention(event, *, shared=False):
    real_flock = files.fcntl.flock
    flag = files.fcntl.LOCK_SH if shared else files.fcntl.LOCK_EX

    def flock(fd, operation):
        try:
            real_flock(fd, operation)
        except BlockingIOError:
            if operation & flag:
                event.set()
            raise

    return patch.object(
        files, "fcntl", SimpleNamespace(**(vars(files.fcntl) | {"flock": flock}))
    )


@contextmanager
def pause_first_writer(staged, contended):
    def fsync(fd):
        os.fsync(fd)
        staged.set()
        wait(contended)

    with os_hooks(fsync=fsync), watch_contention(contended):
        yield


def race_writer(workspace, tool, index, ready, staged):
    async def approve_together(_request):
        ready.wait(TIMEOUT)  # Both requests have captured the same initial version.
        if index == 1:
            wait(staged)
        return ApprovalDecision.ALLOW_ONCE

    return write(workspace, tool, (FIRST, SECOND)[index], approval=approve_together)


def race_process(workspace, locks, tool, index, ready, staged, contended, results):
    with (
        patch.object(files, "_lock_directory", return_value=locks),
        pause_first_writer(staged, contended),
    ):
        result = race_writer(workspace, tool, index, ready, staged)
    results.put((index, os.getpid(), result))


def interrupted_process(workspace, locks, staged, release):
    def fsync(fd):
        os.fsync(fd)
        staged.set()
        wait(release)

    with (
        patch.object(files, "_lock_directory", return_value=locks),
        os_hooks(fsync=fsync),
    ):
        write(workspace, "replace_text", FIRST)


@unittest.skipUnless(os.name == "posix", "File tools require POSIX")
class FileToolConcurrencyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        self.locks = root / "locks"
        self.enterContext(
            patch.object(files, "_lock_directory", return_value=self.locks)
        )
        self.target = self.workspace / NAME

    def assert_conflict(self, results, tool):
        first, second = results
        self.assertTrue(first["ok"], first)
        self.assertFalse(second["ok"], second)
        code = "file_exists" if tool == "write_file" else "target_changed"
        self.assertEqual(second["error"]["code"], code, second)
        self.assertEqual(self.target.read_bytes(), FIRST.encode())
        self.assertEqual(list(self.workspace.iterdir()), [self.target])
        self.assertEqual(len(list(self.locks.iterdir())), 1)

    def check_thread_race(self, tool):
        if tool != "write_file":
            self.target.write_text(OLD)
        ready, staged, contended = (
            threading.Barrier(2),
            threading.Event(),
            threading.Event(),
        )
        with pause_first_writer(staged, contended), ThreadPoolExecutor(2) as pool:
            tasks = [
                pool.submit(race_writer, self.workspace, tool, index, ready, staged)
                for index in range(2)
            ]
            self.assert_conflict([task.result(2 * TIMEOUT) for task in tasks], tool)

    def test_replace_text_two_threads(self):
        self.check_thread_race("replace_text")

    def test_apply_patch_two_threads(self):
        self.check_thread_race("apply_patch")

    def test_write_file_two_threads_no_overwrite(self):
        self.check_thread_race("write_file")

    def test_replace_text_two_processes(self):
        self.target.write_text(OLD)
        ctx = multiprocessing.get_context("spawn")
        ready, staged, contended, results = (
            ctx.Barrier(2),
            ctx.Event(),
            ctx.Event(),
            ctx.Queue(),
        )
        self.addCleanup(results.close)
        processes = [
            ctx.Process(
                target=race_process,
                args=(
                    self.workspace,
                    self.locks,
                    "replace_text",
                    index,
                    ready,
                    staged,
                    contended,
                    results,
                ),
            )
            for index in range(2)
        ]
        for process in processes:
            self.start_process(process)
        reports = {}
        for _ in processes:
            index, pid, result = results.get(timeout=2 * TIMEOUT)
            reports[index] = (pid, result)
        self.assertNotEqual(reports[0][0], reports[1][0])
        for process in processes:
            process.join(TIMEOUT)
            self.assertEqual(process.exitcode, 0)
        self.assert_conflict([reports[0][1], reports[1][1]], "replace_text")

    def start_process(self, process):
        def stop():
            if process.is_alive():
                process.kill()
            process.join(TIMEOUT)
            process.close()

        process.start()
        self.addCleanup(stop)

    def check_reader_visibility(self, *, create):
        new = "甲乙" * 100_000 + "\n"
        expected = new.encode()
        half_written, publish, published, release, contended = (
            threading.Event() for _ in range(5)
        )

        @contextmanager
        def fdopen(fd, mode):
            with os.fdopen(fd, mode) as stream:
                if mode != "wb":
                    yield stream
                    return

                def write_in_halves(data):
                    half = len(data) // 2
                    count = stream.write(data[:half])
                    stream.flush()
                    half_written.set()
                    wait(publish)
                    return count + stream.write(data[half:])

                yield SimpleNamespace(
                    write=write_in_halves, flush=stream.flush, fileno=stream.fileno
                )

        def replace(*args, **kwargs):
            os.replace(*args, **kwargs)
            published.set()
            wait(release)

        def link(*args, **kwargs):
            os.link(*args, **kwargs)
            published.set()
            wait(release)

        if not create:
            self.target.write_text(OLD)
        tool = "write_file" if create else "replace_text"
        # Force the POSIX link fallback for creation and pause while both links exist.
        no_native_rename = SimpleNamespace(
            CDLL=lambda *_args, **_kwargs: SimpleNamespace()
        )
        with (
            os_hooks(fdopen=fdopen, replace=replace, link=link),
            patch.object(files, "ctypes", no_native_rename),
            watch_contention(contended, shared=True),
            ThreadPoolExecutor(2) as pool,
        ):
            writer = pool.submit(write, self.workspace, tool, new)
            try:
                wait(half_written)
                if create:
                    self.assertFalse(self.target.exists())
                else:
                    for _ in range(16):
                        self.assertEqual(self.target.read_bytes(), OLD.encode())
                publish.set()
                wait(published)
                for _ in range(16):
                    self.assertEqual(self.target.read_bytes(), expected)
                if create:
                    self.assertEqual(self.target.stat().st_nlink, 2)
                reader = pool.submit(
                    asyncio.run,
                    execute(
                        self.workspace,
                        "read_file",
                        path=NAME,
                        max_bytes=files.MAX_FILE_BYTES,
                    ),
                )
                wait(contended)
                self.assertFalse(reader.done())
            finally:
                publish.set()
                release.set()
            self.assertTrue(writer.result(TIMEOUT)["ok"])
            result = reader.result(TIMEOUT)
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["content"], new)
        self.assertEqual(self.target.stat().st_nlink, 1)
        self.assertEqual(list(self.workspace.iterdir()), [self.target])

    def test_readers_see_a_complete_version_during_replace(self):
        self.check_reader_visibility(create=False)

    def test_readers_see_complete_data_once_create_is_published(self):
        self.check_reader_visibility(create=True)

    def test_failures_before_and_after_publication(self):
        self.target.write_text(OLD)
        self.target.chmod(0o640)

        def fail_before_write(fd, *_args):
            os.close(fd)
            raise OSError("injected open error")

        def fail_to_publish(*_args, **_kwargs):
            raise OSError("injected publish error")

        def fail_after_publish(*args, **kwargs):
            os.rmdir(*args, **kwargs)
            raise OSError("injected cleanup error")

        for hook, fail, published in (
            ("fdopen", fail_before_write, False),
            ("replace", fail_to_publish, False),
            ("rmdir", fail_after_publish, True),
        ):
            with self.subTest(hook=hook):
                with os_hooks(**{hook: fail}):
                    result = write(self.workspace, "replace_text", FIRST)
                self.assertEqual(result["error"]["code"], "io_error", result)
                self.assertEqual(result.get("partial", False), published)
                self.assertEqual(self.target.read_text(), FIRST if published else OLD)
                self.assertEqual(self.target.stat().st_mode & 0o777, 0o640)
                self.assertEqual(list(self.workspace.iterdir()), [self.target])

    def test_killed_writer_keeps_original_and_releases_lock(self):
        self.target.write_text(OLD)
        ctx = multiprocessing.get_context("spawn")
        staged, release = ctx.Event(), ctx.Event()
        process = ctx.Process(
            target=interrupted_process,
            args=(self.workspace, self.locks, staged, release),
        )
        self.start_process(process)
        wait(staged)
        self.assertEqual(self.target.read_text(), OLD)
        process.kill()
        process.join(TIMEOUT)
        self.assertLess(process.exitcode, 0)
        self.assertEqual(self.target.read_text(), OLD)
        result = write(self.workspace, "replace_text", FIRST)
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.target.read_text(), FIRST)
        self.assertEqual(len(list(self.locks.iterdir())), 1)

    def test_different_files_progress_and_cancelled_waiter_releases_resources(self):
        self.target.write_text(OLD)
        independent = self.workspace / "unrelated.txt"
        independent.write_text(OLD)
        contended = threading.Event()

        async def acquire_and_cancel():
            async with files._file_locks((self.target,)):
                waiting = asyncio.create_task(
                    execute(self.workspace, "read_file", path=NAME)
                )
                await asyncio.wait_for(asyncio.to_thread(wait, contended), TIMEOUT)
                other = await asyncio.wait_for(
                    execute(
                        self.workspace,
                        "replace_text",
                        path=independent.name,
                        old_text=OLD,
                        new_text=SECOND,
                    ),
                    TIMEOUT,
                )
                self.assertTrue(other["ok"], other)
                self.assertFalse(waiting.done())
                waiting.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await waiting
            result = await asyncio.wait_for(
                execute(self.workspace, "read_file", path=NAME), TIMEOUT
            )
            self.assertEqual(result["content"], OLD)

        with watch_contention(contended, shared=True):
            asyncio.run(acquire_and_cancel())
        self.assertEqual(independent.read_text(), SECOND)

    def test_cancelled_writer_holds_lock_until_started_io_finishes(self):
        self.target.write_text(OLD)
        staged, release, contended = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def fsync(fd):
            os.fsync(fd)
            staged.set()
            wait(release)

        async def cancel_during_write():
            writer = asyncio.create_task(
                execute(
                    self.workspace,
                    "replace_text",
                    path=NAME,
                    old_text=OLD,
                    new_text=FIRST,
                )
            )
            try:
                await asyncio.to_thread(wait, staged)
                writer.cancel()
                reader = asyncio.create_task(
                    execute(self.workspace, "read_file", path=NAME)
                )
                await asyncio.to_thread(wait, contended)
                self.assertFalse(writer.done())
                self.assertFalse(reader.done())
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await writer
            result = await asyncio.wait_for(reader, TIMEOUT)
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["content"], FIRST)

        with os_hooks(fsync=fsync), watch_contention(contended, shared=True):
            asyncio.run(cancel_during_write())
        self.assertEqual(list(self.workspace.iterdir()), [self.target])
