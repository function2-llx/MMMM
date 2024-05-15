import cytoolz
from tqdm import tqdm

from mmmm.data.defs import PROCESSED_LOCAL_DATA_ROOT
from mmmm.data.sparse import Sparse

def main():
    for case_dir in tqdm(list((PROCESSED_LOCAL_DATA_ROOT / 'LIDC-IDRI/data').iterdir())):
        sparse = Sparse.from_json((case_dir / 'sparse-old.json').read_bytes())
        for target in cytoolz.concat(sparse.targets.values()):
            target.name = 'lung nodule'
        for key, neg_targets in sparse.neg_targets.items():
            if len(neg_targets) == 1:
                sparse.neg_targets[key] = ['lung nodule']
        (case_dir / 'sparse-new.json').write_bytes(sparse.to_jsonb())
        # (case_dir / 'sparse.json').rename(case_dir / 'sparse-old.json')
        (case_dir / 'sparse-new.json').rename(case_dir / 'sparse.json')

if __name__ == '__main__':
    main()
