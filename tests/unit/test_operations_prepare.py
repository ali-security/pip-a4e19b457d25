import os
import shutil
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from typing import Any
from unittest.mock import Mock, patch

import pytest

from pip._internal.exceptions import HashMismatch
from pip._internal.models.link import Link
from pip._internal.network.download import Downloader
from pip._internal.network.session import PipSession
from pip._internal.operations.prepare import (
    RequirementPreparer,
    _check_download_dir,
    unpack_url,
)
from pip._internal.utils.hashes import Hashes

from tests.lib import TestData
from tests.lib.requests_mocks import MockResponse


def test_unpack_url_with_urllib_response_without_content_type(data: TestData) -> None:
    """
    It should download and unpack files even if no Content-Type header exists
    """
    _real_session = PipSession()

    def _fake_session_get(*args: Any, **kwargs: Any) -> dict[str, str]:
        resp = _real_session.get(*args, **kwargs)
        del resp.headers["Content-Type"]
        return resp

    session = Mock()
    session.resume_retries = 0
    session.get = _fake_session_get
    download = Downloader(session, progress_bar="on")

    uri = data.packages.joinpath("simple-1.0.tar.gz").as_uri()
    link = Link(uri)
    temp_dir = mkdtemp()
    try:
        unpack_url(
            link,
            temp_dir,
            download=download,
            download_dir=None,
            verbosity=0,
        )
        assert set(os.listdir(temp_dir)) == {
            "PKG-INFO",
            "setup.cfg",
            "setup.py",
            "simple",
            "simple.egg-info",
        }
    finally:
        rmtree(temp_dir)


@patch("pip._internal.network.download.raise_for_status")
def test_download_http_url__no_directory_traversal(
    mock_raise_for_status: Mock, tmpdir: Path
) -> None:
    """
    Test that directory traversal doesn't happen on download when the
    Content-Disposition header contains a filename with a ".." path part.
    """
    mock_url = "http://www.example.com/whatever.tgz"
    contents = b"downloaded"
    link = Link(mock_url)

    session = Mock()
    session.resume_retries = 0
    resp = MockResponse(contents)
    resp.url = mock_url
    resp.headers = {
        # Set the content-type to a random value to prevent
        # mimetypes.guess_extension from guessing the extension.
        "content-type": "random",
        "content-disposition": 'attachment;filename="../out_dir_file"',
    }
    session.get.return_value = resp
    download = Downloader(session, progress_bar="on")

    download_dir = os.fspath(tmpdir.joinpath("download"))
    os.mkdir(download_dir)
    file_path, content_type = download(link, download_dir)
    # The file should be downloaded to download_dir.
    actual = os.listdir(download_dir)
    assert actual == ["out_dir_file"]
    mock_raise_for_status.assert_called_once_with(resp)


# A doubly-encoded "../" in the URL path. The path is percent-decoded exactly
# once, so the file name must keep the literal "%2e%2e%2f" rather than turning
# into a "../" path part that escapes the target directory.
_DOUBLE_ENCODED_TRAVERSAL_URL = "http://example.com/%252e%252e%252fout_dir_file.tgz"
_DOUBLE_ENCODED_TRAVERSAL_NAME = "%2e%2e%2fout_dir_file.tgz"


@patch("pip._internal.network.download.raise_for_status")
def test_download_http_url__no_directory_traversal_double_encoded_url(
    mock_raise_for_status: Mock, tmpdir: Path
) -> None:
    """
    Test that directory traversal doesn't happen on download when the URL
    path contains a doubly percent-encoded ".." path part.
    """
    contents = b"downloaded"
    link = Link(_DOUBLE_ENCODED_TRAVERSAL_URL)

    session = Mock()
    session.resume_retries = 0
    resp = MockResponse(contents)
    resp.url = _DOUBLE_ENCODED_TRAVERSAL_URL
    resp.headers = {"content-type": "random"}
    session.get.return_value = resp
    download = Downloader(session, progress_bar="on")

    download_dir = os.fspath(tmpdir.joinpath("download"))
    os.mkdir(download_dir)
    file_path, content_type = download(link, download_dir)
    # The file should be downloaded to download_dir, not its parent.
    assert os.path.dirname(file_path) == download_dir
    assert os.listdir(download_dir) == [_DOUBLE_ENCODED_TRAVERSAL_NAME]
    assert not os.path.exists(tmpdir.joinpath("out_dir_file.tgz"))
    mock_raise_for_status.assert_called_once_with(resp)


def test_check_download_dir__no_directory_traversal_double_encoded_url(
    tmpdir: Path,
) -> None:
    """
    Test that a previously-downloaded file lookup stays inside download_dir
    when the URL path contains a doubly percent-encoded ".." path part, so a
    file outside of it is neither reused nor deleted on a hash mismatch.
    """
    link = Link(_DOUBLE_ENCODED_TRAVERSAL_URL)
    download_dir = os.fspath(tmpdir.joinpath("download"))
    os.mkdir(download_dir)
    outside_file = tmpdir.joinpath("out_dir_file.tgz")
    outside_file.write_bytes(b"outside")

    assert _check_download_dir(link, download_dir, None) is None
    assert (
        _check_download_dir(link, download_dir, Hashes({"sha256": ["bogus"]}))
        is None
    )
    assert outside_file.read_bytes() == b"outside"


def test_save_linked_requirement__no_directory_traversal_double_encoded_url(
    tmpdir: Path,
) -> None:
    """
    Test that saving a downloaded requirement into download_dir doesn't
    escape it when the URL path contains a doubly percent-encoded ".." path
    part.
    """
    download_dir = os.fspath(tmpdir.joinpath("download"))
    os.mkdir(download_dir)
    local_file = tmpdir.joinpath("local.tgz")
    local_file.write_bytes(b"downloaded")

    req = Mock()
    req.link = Link(_DOUBLE_ENCODED_TRAVERSAL_URL)
    req.local_file_path = os.fspath(local_file)
    preparer = Mock(download_dir=download_dir)
    RequirementPreparer.save_linked_requirement(preparer, req)

    assert os.listdir(download_dir) == [_DOUBLE_ENCODED_TRAVERSAL_NAME]
    assert not os.path.exists(tmpdir.joinpath("out_dir_file.tgz"))


@pytest.fixture
def clean_project(tmpdir_factory: pytest.TempPathFactory, data: TestData) -> Path:
    tmpdir = tmpdir_factory.mktemp("clean_project")
    new_project_dir = tmpdir.joinpath("FSPkg")
    path = data.packages.joinpath("FSPkg")
    shutil.copytree(path, new_project_dir)
    return new_project_dir


class Test_unpack_url:
    def prep(self, tmpdir: Path, data: TestData) -> None:
        self.build_dir = os.fspath(tmpdir.joinpath("build"))
        self.download_dir = tmpdir.joinpath("download")
        os.mkdir(self.build_dir)
        os.mkdir(self.download_dir)
        self.dist_file = "simple-1.0.tar.gz"
        self.dist_file2 = "simple-2.0.tar.gz"
        self.dist_path = data.packages.joinpath(self.dist_file)
        self.dist_path2 = data.packages.joinpath(self.dist_file2)
        self.dist_url = Link(self.dist_path.as_uri())
        self.dist_url2 = Link(self.dist_path2.as_uri())
        self.no_download = Mock(side_effect=AssertionError)

    def test_unpack_url_no_download(self, tmpdir: Path, data: TestData) -> None:
        self.prep(tmpdir, data)
        unpack_url(self.dist_url, self.build_dir, self.no_download, verbosity=0)
        assert os.path.isdir(os.path.join(self.build_dir, "simple"))
        assert not os.path.isfile(os.path.join(self.download_dir, self.dist_file))

    def test_unpack_url_bad_hash(self, tmpdir: Path, data: TestData) -> None:
        """
        Test when the file url hash fragment is wrong
        """
        self.prep(tmpdir, data)
        url = f"{self.dist_url.url}#md5=bogus"
        dist_url = Link(url)
        with pytest.raises(HashMismatch):
            unpack_url(
                dist_url,
                self.build_dir,
                download=self.no_download,
                hashes=Hashes({"md5": ["bogus"]}),
                verbosity=0,
            )
