from subprocess import CompletedProcess

from platform_detections.detectors.os import linux


def test_openmp_probe_uses_and_removes_a_unique_temp_directory(monkeypatch):
    compiled_source = None

    def fake_run(command, **kwargs):
        nonlocal compiled_source
        if command[:2] == ["gcc", "--version"]:
            return CompletedProcess(command, 0, stdout="gcc 13.2.0\n", stderr="")
        if command[0] == "gcc" and "-fopenmp" in command:
            compiled_source = linux.Path(command[-1])
            assert compiled_source.name == "openmp_test.c"
            assert compiled_source.exists()
            return CompletedProcess(command, 0, stdout="", stderr="")
        return CompletedProcess(command, 1, stdout="", stderr="")

    monkeypatch.setattr(linux.subprocess, "run", fake_run)

    result = linux.LinuxDetector()._detect_compilers_and_libraries()

    assert result["has_openmp"] is True
    assert compiled_source is not None
    assert not compiled_source.parent.exists()
