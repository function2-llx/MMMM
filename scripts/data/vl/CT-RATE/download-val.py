from huggingface_hub import snapshot_download

from mmmm.data.defs import ORIGIN_DATA_ROOT, ORIGIN_VL_DATA_ROOT

def main():
    snapshot_download(
        'ibrahimhamamci/CT-RATE',
        repo_type='dataset',
        allow_patterns='dataset/valid/**',
        resume_download=True,
        cache_dir=ORIGIN_DATA_ROOT / '.hub',
        local_dir=ORIGIN_VL_DATA_ROOT / 'CT-RATE',
        local_dir_use_symlinks=True,
    )

if __name__ == '__main__':
    main()
