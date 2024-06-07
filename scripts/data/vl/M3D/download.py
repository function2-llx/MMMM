from huggingface_hub import snapshot_download

from mmmm.data.defs import ORIGIN_DATA_ROOT, ORIGIN_VL_DATA_ROOT

def main():
    while True:
        try:
            snapshot_download(
                'GoodBaiBai88/M3D-Cap',
                repo_type='dataset',
                resume_download=True,
                cache_dir=ORIGIN_DATA_ROOT / '.hub',
                local_dir=ORIGIN_VL_DATA_ROOT / 'M3D-Cap',
                local_dir_use_symlinks=True,
            )
            break
        except KeyboardInterrupt:
            break
        except Exception:
            pass

if __name__ == '__main__':
    main()
