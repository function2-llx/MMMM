from ._base import Binary3DMaskLoaderMixin, Default3DImageLoaderMixin, MultiLabelMultiFileDataPoint, Processor

class HaNSegProcessor(Default3DImageLoaderMixin, Binary3DMaskLoaderMixin, Processor):
    name = 'HaN-Seg'
    max_workers = 8
    image_reader = 'nrrdreader'
    mask_reader = 'nrrdreader'

    def get_data_points(self):
        # https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.16197, Table 2
        ret = []
        name_mapping = {
            'A_Carotid_L': 'left carotid artery',
            'A_Carotid_R': 'right carotid artery',
            'Arytenoid': 'arytenoid cartilages',
            'Bone_Mandible': 'mandible',
            'Brainstem': 'brainstem',
            'BuccalMucosa': 'buccal mucosa',
            'Cavity_Oral': 'oral cavity',
            'Cochlea_L': 'left cochlea',
            'Cochlea_R': 'right cochlea',
            'Cricopharyngeus': 'cricopharyngeus',  # "cricopharyngeal inlet" seems to be very rarely used
            'Esophagus_S': 'cervical esophagus',
            'Eye_AL': 'anterior segment of left eyeball',
            'Eye_AR': 'anterior segment of right eyeball',
            'Eye_PL': 'posterior segment of left eyeball',
            'Eye_PR': 'posterior segment of right eyeball',
            'Glnd_Lacrimal_L': 'left lacrimal gland',
            'Glnd_Lacrimal_R': 'right lacrimal gland',
            'Glnd_Submand_L': 'left submandibular gland',
            'Glnd_Submand_R': 'right submandibular gland',
            'Glnd_Thyroid': 'thyroid',
            'Glottis': 'glottis',
            'Larynx_SG': 'supraglottis',
            'Lips': 'lip',
        }
        for case_dir in (self.dataset_root / 'set_1').glob('case_*'):
            key = case_dir.name
            ret.append(
                MultiLabelMultiFileDataPoint(
                    key=key,
                    # MR image is not co-registered with CT, thus not used
                    images={'CT': case_dir / f'{key}_IMG_CT.nrrd'},
                    masks=[
                        (tax_name, case_dir / f'{key}_OAR_{file_key}.seg.nrrd')
                        for file_key, tax_name in name_mapping.items()
                    ],
                ),
            )
        return ret
