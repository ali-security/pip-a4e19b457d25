import datetime
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
import sys
from collections.abc import Sequence
from importlib.machinery import ModuleSpec
from optparse import Values
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional
from unittest import mock

import pytest

from pip._internal.cli.base_command import Command
from pip._internal.cli.req_command import (
    IndexGroupCommand,
    RequirementCommand,
    SessionCommandMixin,
)
from pip._internal.cli.status_codes import SUCCESS
from pip._internal.commands import commands_dict, create_command

# These are the expected names of the commands whose classes inherit from
# IndexGroupCommand.
EXPECTED_INDEX_GROUP_COMMANDS = [
    "download",
    "index",
    "install",
    "list",
    "lock",
    "wheel",
]


def check_commands(pred: Callable[[Command], bool], expected: list[str]) -> None:
    """
    Check the commands satisfying a predicate.
    """
    commands = [create_command(name) for name in sorted(commands_dict)]
    actual = [command.name for command in commands if pred(command)]
    assert actual == expected, f"actual: {actual}"


def test_commands_dict__order() -> None:
    """
    Check the ordering of commands_dict.
    """
    names = list(commands_dict)
    # A spot-check is sufficient to check that commands_dict encodes an
    # ordering.
    assert names[0] == "install"
    assert names[-1] == "help"


@pytest.mark.parametrize("name", list(commands_dict))
def test_create_command(name: str) -> None:
    """Test creating an instance of each available command."""
    command = create_command(name)
    assert command.name == name
    assert command.summary == commands_dict[name].summary


def test_session_commands() -> None:
    """
    Test which commands inherit from SessionCommandMixin.
    """

    def is_session_command(command: Command) -> bool:
        return isinstance(command, SessionCommandMixin)

    expected = [
        "download",
        "index",
        "install",
        "list",
        "lock",
        "search",
        "uninstall",
        "wheel",
    ]
    check_commands(is_session_command, expected)


def test_index_group_commands() -> None:
    """
    Test the commands inheriting from IndexGroupCommand.
    """

    def is_index_group_command(command: Command) -> bool:
        return isinstance(command, IndexGroupCommand)

    check_commands(is_index_group_command, EXPECTED_INDEX_GROUP_COMMANDS)

    # Also check that the commands inheriting from IndexGroupCommand are
    # exactly the commands with the --no-index option.
    def has_option_no_index(command: Command) -> bool:
        return command.parser.has_option("--no-index")

    check_commands(has_option_no_index, EXPECTED_INDEX_GROUP_COMMANDS)


@pytest.mark.parametrize("command_name", EXPECTED_INDEX_GROUP_COMMANDS)
@pytest.mark.parametrize(
    "disable_pip_version_check, no_index, expected_called",
    [
        # The fetch phase only runs when both disable_pip_version_check
        # and no_index are False.
        (False, False, True),
        (False, True, False),
        (True, False, False),
        (True, True, False),
    ],
)
@mock.patch("pip._internal.cli.index_command._pip_self_version_check_fetch")
def test_index_group_pip_version_check(
    mock_version_check: mock.Mock,
    command_name: str,
    disable_pip_version_check: bool,
    no_index: bool,
    expected_called: bool,
) -> None:
    """
    Test whether the pre-body fetch runs when ``pip_version_check()`` is
    entered, for each of the IndexGroupCommand classes.
    """
    command = create_command(command_name)
    options = command.parser.get_default_values()
    options.disable_pip_version_check = disable_pip_version_check
    options.no_index = no_index
    # Return None so the emit branch is a no-op.
    mock_version_check.return_value = None

    # See test test_list_pip_version_check() below.
    if command_name == "list":
        expected_called = False

    with command.pip_version_check(options):
        pass
    if expected_called:
        mock_version_check.assert_called_once()
    else:
        mock_version_check.assert_not_called()


def test_requirement_commands() -> None:
    """
    Test which commands inherit from RequirementCommand.
    """

    def is_requirement_command(command: Command) -> bool:
        return isinstance(command, RequirementCommand)

    check_commands(is_requirement_command, ["download", "install", "lock", "wheel"])


@pytest.mark.parametrize("flag", ["", "--outdated", "--uptodate"])
@mock.patch("pip._internal.cli.index_command._pip_self_version_check_fetch")
@mock.patch.dict(os.environ, {"PIP_DISABLE_PIP_VERSION_CHECK": "no"})
def test_list_pip_version_check(version_check_mock: mock.Mock, flag: str) -> None:
    """
    Ensure that pip list doesn't perform a version self-check unless given
    --outdated or --uptodate (as they require hitting the network anyway).
    """
    command = create_command("list")
    command.run = lambda *args, **kwargs: 0  # type: ignore[method-assign]
    command.main([flag])
    if flag != "":
        version_check_mock.assert_called_once()
    else:
        version_check_mock.assert_not_called()


@mock.patch.dict(os.environ, {"PIP_DISABLE_PIP_VERSION_CHECK": "no"})
def test_install_imports_self_version_check_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ``pip install`` must import the self-version check module before the
    command body runs.

    The body may overwrite pip's own files on disk. If the self-version check
    were only imported once the body is done, the still-running pip would
    import and execute that freshly-installed code (CVE-2026-6357).
    """
    module_name = "pip._internal.self_outdated_check"
    events: list[str] = []

    class RecordingFinder(importlib.abc.MetaPathFinder):
        """Record fresh (non-cached) imports of the self-version check."""

        def find_spec(
            self,
            fullname: str,
            path: Optional[Sequence[str]],
            target: Optional[ModuleType] = None,
        ) -> Optional[ModuleSpec]:
            if fullname == module_name:
                events.append("import")
            return None

    class OverwrittenModuleLoader(importlib.abc.Loader):
        """Stands in for a self_outdated_check.py replaced by the install."""

        def exec_module(self, module: ModuleType) -> None:
            events.append("injected code executed")

    class OverwrittenModuleFinder(importlib.abc.MetaPathFinder):
        def find_spec(
            self,
            fullname: str,
            path: Optional[Sequence[str]],
            target: Optional[ModuleType] = None,
        ) -> Optional[ModuleSpec]:
            if fullname != module_name:
                return None
            return importlib.util.spec_from_loader(fullname, OverwrittenModuleLoader())

    def run(options: Values, args: list[str]) -> int:
        # Simulate the install replacing pip's own modules on disk.
        sys.meta_path.insert(0, OverwrittenModuleFinder())
        events.append("body")
        return SUCCESS

    # Seed a fresh self-check state file so the version check never needs
    # to reach the network.
    cache_dir = tmp_path / "cache"
    selfcheck_dir = cache_dir / "selfcheck"
    selfcheck_dir.mkdir(parents=True)
    statefile = selfcheck_dir / hashlib.sha224(sys.prefix.encode()).hexdigest()
    statefile.write_text(
        json.dumps(
            {
                "key": sys.prefix,
                "last_check": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "pypi_version": "1.0",
            }
        ),
        encoding="utf-8",
    )

    # Make the self-version check module "not yet imported", restoring the
    # original module (and the parent package attribute) afterwards.
    original_module = importlib.import_module(module_name)
    monkeypatch.setattr(
        sys.modules["pip._internal"], "self_outdated_check", original_module
    )
    monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.setattr(sys, "meta_path", [RecordingFinder(), *sys.meta_path])

    command = create_command("install")
    command.run = run  # type: ignore[method-assign]
    status = command.main(["--cache-dir", os.fspath(cache_dir), "example"])

    assert status == SUCCESS
    assert events == ["import", "body"]
