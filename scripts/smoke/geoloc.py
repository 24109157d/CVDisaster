"""读取四对真实 CVIAN 样本，执行 ConvNeXt Base 单 batch 训练和最小检索。"""

import csv
import json
import math
from pathlib import Path
import random
import shutil
import sys
from types import SimpleNamespace

from common import ASSETS, REPO, digest, execute, pairs


def work(data, report, protected):
    with report.stage('imports_and_gpu'):
        import numpy as np
        from PIL import Image
        import torch
        from torch.utils.data import DataLoader
        sys.path.insert(0, str(REPO / 'Author CVDisaster/geolocalization'))
        from sample4geo.dataset.cvdisaster import CVDisasterDatasetTrain, CVDisasterDatasetEval
        from sample4geo.model import TimmModel
        from sample4geo.loss import InfoNCE
        from sample4geo.transforms import get_transforms_train, get_transforms_val
        from sample4geo.trainer import predict

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.set_num_threads(2)
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        probe = torch.ones((16, 16), device='cuda')
        assert (probe @ probe).sum().item() == 4096
        report.state['gpu'] = {'framework': torch.__version__, 'name': torch.cuda.get_device_name(0)}
        torch.cuda.reset_peak_memory_stats()

    with report.stage('prepare_four_sample_sandbox'):
        samples = pairs(data, 4)
        geo = data / '02_Position/CVIAN_position.geojson'
        protected[str(geo)] = digest(geo)
        features = json.loads(geo.read_text())['features']
        angles = {str(f['properties']['id']): f['properties']['compass_an'] for f in features}
        sandbox = report.output / 'sample-data'
        rows = []
        for sample in samples:
            # 仅复制四对图像到本次输出；对街景执行与作者相同的 compass roll。
            for kind in ('svi', 'sat'):
                protected[sample[kind]] = digest(sample[kind])
            angle = float(angles[str(sample['id'])])
            assert math.isfinite(angle)
            svi_rel = Path('00_SVI_aligned') / sample['relative']
            sat_rel = Path('01_Satellite') / sample['relative']
            (sandbox / svi_rel).parent.mkdir(parents=True, exist_ok=True)
            (sandbox / sat_rel).parent.mkdir(parents=True, exist_ok=True)
            with Image.open(sample['svi']) as image:
                pixels = np.asarray(image)
                assert pixels.shape == (512, 1024, 3), pixels.shape
                shifted = np.roll(pixels, int((-angle / 360) * image.width), axis=1)
                Image.fromarray(shifted).save(sandbox / svi_rel)
            with Image.open(sample['sat']) as image:
                assert np.asarray(image).shape == (512, 512, 3)
            shutil.copyfile(sample['sat'], sandbox / sat_rel)
            rows.append((sat_rel.as_posix(), svi_rel.as_posix()))
        # 测试 CSV 只在沙盒生成；训练两对、评估另外两对，不生成正式 subarea 划分。
        for name, subset in [('train_50.csv', rows[:2]), ('val_50.csv', rows[2:])]:
            with (sandbox / name).open('w', newline='') as stream:
                csv.writer(stream).writerows(subset)
        report.state['samples'] = samples

    with report.stage('initialize_pretrained_convnext_base'):
        model = TimmModel('convnext_base.fb_in22k_ft_in1k_384', pretrained=True, img_size=384)
        state = torch.load(ASSETS['cvusa'][0], map_location='cpu')
        # 权重不兼容时直接暴露错误，不静默忽略缺失或额外参数。
        model.load_state_dict(state, strict=True)
        del state
        config = model.get_config()
        model = model.cuda()
        sat_train, svi_train = get_transforms_train((384, 384), (140, 768), config['mean'], config['std'])
        sat_eval, svi_eval = get_transforms_val((384, 384), (140, 768), config['mean'], config['std'])

    with report.stage('one_training_batch'):
        dataset = CVDisasterDatasetTrain(str(sandbox), transforms_query=svi_train,
                    transforms_reference=sat_train, split_key='val50')
        query, reference, ids = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
        assert ids.dtype == torch.int64 and len(set(ids.tolist())) == 2
        assert tuple(query.shape) == (2, 3, 140, 768)
        assert tuple(reference.shape) == (2, 3, 384, 384)
        criterion = InfoNCE(torch.nn.CrossEntropyLoss(label_smoothing=0.1), device='cuda')
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scaler = torch.cuda.amp.GradScaler()
        parameter = next(model.model.parameters())
        before = parameter.detach().clone()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            first, second = model(query.cuda(), reference.cuda())
            loss = criterion(first, second, model.logit_scale.exp())
        assert torch.isfinite(loss) and loss.item() > 0
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g).item() > 0 for g in gradients)
        torch.nn.utils.clip_grad_value_(model.parameters(), 100.)
        scaler.step(optimizer)
        scaler.update()
        assert not torch.equal(before, parameter.detach()), '优化器未实际更新参数'
        report.state['training'] = {'batches': 1, 'batch_size': 2, 'loss': loss.item(),
                                    'ids': ids.tolist(), 'parameters_updated': True}
        del optimizer, gradients, before

    with report.stage('minimal_heldout_retrieval'):
        config = SimpleNamespace(device='cuda', verbose=False, normalize_features=True)
        reference_data = CVDisasterDatasetEval(str(sandbox), split='test', img_type='reference',
                                             transforms=sat_eval, split_key='val50')
        query_data = CVDisasterDatasetEval(str(sandbox), split='test', img_type='query',
                                         transforms=svi_eval, split_key='val50')
        # 复用作者 predict 提取特征；只有两个候选时只计算 Recall@1，不输出失真的 top1%。
        references, rids = predict(config, model, DataLoader(reference_data, batch_size=2, num_workers=0))
        queries, qids = predict(config, model, DataLoader(query_data, batch_size=2, num_workers=0))
        scores = queries @ references.T
        assert scores.shape == (2, 2) and torch.isfinite(scores).all()
        assert set(qids.tolist()) == set(rids.tolist())
        predictions = rids[scores.argmax(dim=1)]
        recall = (predictions == qids).float().mean().item()
        report.state['evaluation'] = {'candidate_count': 2, 'query_count': 2,
            'recall_at_1_percent': recall * 100, 'query_ids': qids.tolist(),
            'predicted_ids': predictions.tolist(), 'scores': scores.cpu().tolist(),
            'meaning': '仅为独立微型样本的流程检查，不是正式定位性能'}
        report.state['peak_gpu_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20


if __name__ == '__main__':
    execute('geoloc', work)
