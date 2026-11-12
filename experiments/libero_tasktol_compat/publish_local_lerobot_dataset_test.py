import pytest

from experiments.libero_tasktol_compat.publish_local_lerobot_dataset import validate_audited_parquet_hashes


def test_publication_rejects_changed_or_missing_audited_parquet():
    info = dict(total_episodes=2, chunks_size=1000,
                data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    audit = dict(episode_checks=[dict(episode_index=0, parquet_sha256="first"),
                                 dict(episode_index=1, parquet_sha256="second")])
    files = [dict(path="data/chunk-000/episode_000000.parquet", sha256="first"),
             dict(path="data/chunk-000/episode_000001.parquet", sha256="second")]
    validate_audited_parquet_hashes(info, audit, files)
    with pytest.raises(AssertionError, match="Audited parquet changed"):
        validate_audited_parquet_hashes(info, audit, [files[0], dict(files[1], sha256="modified")])
    with pytest.raises(AssertionError, match="Audited parquet changed"):
        validate_audited_parquet_hashes(info, audit, files[:1])
