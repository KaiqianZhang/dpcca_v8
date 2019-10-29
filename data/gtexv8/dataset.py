"""=============================================================================
Represent GTEx dataset of histology images and gene expression levels.
=============   ================================================================"""

from   PIL import Image
import torch
from   torch.utils.data import Dataset
from   torchvision import transforms

# ------------------------------------------------------------------------------

class GTExV8Dataset(Dataset):

    def __repr__(self):
        return '<GTExV8Dataset>'

# ------------------------------------------------------------------------------

    def __init__(self, cfg):
        self.cfg = cfg

        data = torch.load('%s/train.pth' % cfg.ROOT_DIR)

        self.samples    = data['samples']
        # We cannot standardize the images yet; this DF is just filenames.
        self.images_df  = data['images']
        df              = data['genes']
        # This standardizes the columns of this DF, which are samples.
        self.genes_df   = (df - df.min()) / (df.max() - df.min())
        self.tissues_df = data['tissues']

        self.subsample_image = transforms.Compose([
            transforms.RandomRotation((0, 360)),
            transforms.RandomCrop(cfg.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        # For easy indexing of labels.
        labels = []
        for i in range(len(self.samples)):
            sample_id = self.samples[i]
            label = self.tissues_df.loc[sample_id][0]
            labels.append(label)
        self.labels = labels

# ------------------------------------------------------------------------------

    def __len__(self):
        """Return number of samples in dataset.
        """
        return len(self.samples)

# ------------------------------------------------------------------------------

    def __getitem__(self, i):
        """Return the `idx`-th (image, genes)-pair from the dataset.
        """
        sample_id = self.samples[i]

        fname  = self.images_df.loc[sample_id]
        fpath  = '%s/images/%s' % (self.cfg.ROOT_DIR, fname)
        pixels = Image.open(fpath)

        bad_crop = True
        while bad_crop:
            image = self.subsample_image(pixels).numpy()
            # We want to avoid all black crops because it prevents us from
            # feature normalization.
            if image.min() == image.max():
                continue
            # We want to avoid crops that are majority black.
            if (image == 0).sum() / image.size > 0.5:
                continue

            bad_crop = False
            image = (image - image.min()) / (image.max() - image.min())
            image = torch.Tensor(image)

        genes = self.genes_df[sample_id].values
        genes = torch.Tensor(genes)

        assert image.min() == 0
        assert image.max() == 1
        assert genes.min() == 0
        assert genes.max() == 1

        return image, genes
