"""Validate the two real datasets and approved long-run configurations on CPU."""
import dataclasses
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from openpi.models import tokenizer
from openpi.training import config, data_loader

ROOT = Path('/home/czl/openpi')
RUN = Path('/mnt/SharedData/czl/runs/dual40k_nostate_bs36_20261029')
GROUPS = {
    'nominal': {
        'config': 'pi05_franka_nominal100_action7_nostate_3gpu_bs36_40k',
        'repo': 'caozilin/franka_100_nominal',
        'assets': '/mnt/SharedData/czl/assets/pi05_franka_mujoco_state_action7',
        'norm_sha256': 'e1e3c4e7630f6ee524d99ae99cb0dfe8f61306d83c3feebbfa564edc0c5b7a07',
        'frames': 421102,
        'action_stats_dim': 7,
    },
    'joint': {
        'config': 'pi05_tasktol_mix20_joint32_nostate_4xloss_3gpu_bs36_40k',
        'repo': 'caozilin/franka_nominal80_tolerance20_potpost001_dsp16_20261027',
        'assets': '/mnt/SharedData/czl/assets/pi05_franka_mujoco_joint13',
        'norm_sha256': 'ceefe7d23f307025d6ab06711faeecb6c2a815029f91c814ad0ad5e06c8dc21b',
        'frames': 421029,
        'action_stats_dim': 13,
    },
}


def main():
    logging.basicConfig(level=logging.INFO)
    result = {'passed': False, 'groups': {}}
    started = time.perf_counter()
    long_tokenizer = tokenizer.PaligemmaTokenizer(512)
    common = None
    for group, expected in GROUPS.items():
        base = config.get_config(expected['config'])
        dataset_root = Path('/mnt/SharedData/czl/lerobot') / expected['repo']
        cache = Path('/mnt/SharedData/czl/hf_datasets') / expected['repo']
        assert cache.is_dir()
        cfg = dataclasses.replace(base, data=dataclasses.replace(
            base.data, lerobot_dataset_root=str(dataset_root), hf_datasets_cache_dir=str(cache),
            assets=config.AssetsConfig(assets_dir=expected['assets'])))
        assert cfg.model.pi05 and cfg.model.action_dim == 32 and cfg.model.action_horizon == 16
        assert not cfg.model.discrete_state_input and cfg.model.max_token_len == 48
        assert cfg.model.paligemma_variant == 'gemma_2b_lora'
        assert cfg.model.action_expert_variant == 'gemma_300m_lora'
        assert (cfg.batch_size, cfg.num_train_steps, cfg.log_interval, cfg.save_interval, cfg.keep_period) == (36, 40000, 10, 5000, 5000)
        assert not cfg.resume and not cfg.overwrite and cfg.ema_decay is None
        assert cfg.weight_loader.params_path == 'gs://openpi-assets/checkpoints/pi05_base/params'
        signature = [dataclasses.asdict(cfg.lr_schedule), dataclasses.asdict(cfg.optimizer), str(cfg.freeze_filter)]
        assert common is None or signature == common
        common = signature
        assert signature[0] == {'warmup_steps': 1000, 'peak_lr': 5e-5, 'decay_steps': 40000, 'decay_lr': 5e-6}
        expected_groups = (('action', 0, 7, 1.0),) if group == 'nominal' else (
            ('action', 0, 7, 1.0), ('rotation_tolerance', 7, 13, 0.5))
        assert cfg.model.action_loss_groups == expected_groups
        assert cfg.data.tolerance_trajectory_action_loss_weight == (1.0 if group == 'nominal' else 4.0)
        assert cfg.data.joint_task_tolerance == (group == 'joint')
        stats_path = Path(expected['assets']) / expected['repo'] / 'norm_stats.json'
        assert hashlib.sha256(stats_path.read_bytes()).hexdigest() == expected['norm_sha256']
        dc = cfg.data.create(cfg.assets_dirs, cfg.model)
        for name, dimension in [('state', 7), ('actions', expected['action_stats_dim'])]:
            for field in ('mean', 'std', 'q01', 'q99'):
                value = np.asarray(getattr(dc.norm_stats[name], field))
                assert value.shape == (dimension,) and np.isfinite(value).all(), (group, name, field)
        meta = LeRobotDatasetMetadata(expected['repo'], root=dataset_root)
        assert meta.total_frames == expected['frames'] and meta.total_episodes == 2500
        # LeRobot stores tasks as an integer-keyed mapping in this environment.
        tasks = [str(meta.tasks[i]) for i in range(int(meta.info['total_tasks']))]
        lengths = [int(long_tokenizer.tokenize(task, None)[1].sum()) for task in tasks]
        assert max(lengths) <= 48, (group, max(lengths))
        raw = data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
        dataset = data_loader.transform_dataset(raw, dc)
        candidates = {0, len(dataset)//4, len(dataset)//2, len(dataset)-1}
        if group == 'joint':
            # Ensure both sample classes are inspected even if they are clustered by episode.
            episodes_path = dataset_root/'meta/episodes.jsonl'
            offset = 0
            for line in episodes_path.read_text().splitlines():
                episode = json.loads(line)
                candidates.add(offset)
                offset += episode['length']
                if len(candidates) >= 40:
                    break
        checked, weights = [], set()
        for index in sorted(candidates):
            item = dataset[index]
            assert item['actions'].shape == (16, 32) and item['state'].shape == (32,)
            assert item['tokenized_prompt'].shape == (48,)
            np.testing.assert_array_equal(item['actions'][:, expected['action_stats_dim']:], 0)
            assert np.isfinite(item['actions']).all()
            weight = float(item['action_loss_weight'])
            assert weight in ({1.0} if group == 'nominal' else {1.0, 4.0})
            weights.add(weight)
            checked.append({'index': index, 'tokens': int(np.asarray(item['tokenized_prompt_mask']).sum()), 'action_loss_weight': weight})
        if group == 'joint':
            assert weights == {1.0, 4.0}, ('Did not inspect both trajectory types', weights)
        result['groups'][group] = {'config': expected['config'], 'frames': len(dataset),
            'repo': expected['repo'], 'norm_sha256': expected['norm_sha256'],
            'max_prompt_tokens': max(lengths), 'prompt_count': len(lengths),
            'observed_action_loss_weights': sorted(weights), 'sample_checks': checked,
            'resolved_config': dataclasses.asdict(cfg)}
        print(group, 'PASS', 'max_prompt_tokens', max(lengths), 'weights', sorted(weights), flush=True)
    result.update(passed=True, seconds=time.perf_counter()-started)
    (RUN/'preflight_result.json').write_text(json.dumps(result, indent=2, default=str)+'\n')
    print('All preflight checks passed', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        (RUN/'preflight_failure.json').write_text(json.dumps({'error': repr(error)}))
        raise
