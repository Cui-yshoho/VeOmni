from veomni.utils.checkpoint_utils import _validate_dcp_checkpoint_entry


def test_native_dcp_metadata_remains_a_valid_checkpoint(tmp_path):
    checkpoint = tmp_path / "global_step_3"
    checkpoint.mkdir()
    (checkpoint / ".metadata").touch()

    assert _validate_dcp_checkpoint_entry(str(tmp_path), checkpoint.name) == 3


def test_completed_hyper_checkpoint_is_valid_for_auto_resume(tmp_path):
    checkpoint = tmp_path / "global_step_5"
    checkpoint.mkdir()
    (checkpoint / ".hyper_complete").write_text("complete\n", encoding="utf-8")

    assert _validate_dcp_checkpoint_entry(str(tmp_path), checkpoint.name) == 5


def test_incomplete_hyper_checkpoint_is_rejected(tmp_path):
    checkpoint = tmp_path / "global_step_7"
    (checkpoint / "model").mkdir(parents=True)
    (checkpoint / "model" / ".metadata").touch()

    assert _validate_dcp_checkpoint_entry(str(tmp_path), checkpoint.name) is None
