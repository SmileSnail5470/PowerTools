import os
import time
from PIL import Image
import torch
import torchvision as tv


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root

    def read_img(self, path):
        pic = Image.open(path).convert('RGB')
        transform = tv.transforms.ToTensor()
        return transform(pic)

    def read_mask(self, path):
        pic = Image.open(path).convert('L')
        transform = tv.transforms.ToTensor()
        return transform(pic)

    def get_video(self):
        frame_ids = sorted([int(os.path.splitext(file)[0]) for file in os.listdir(os.path.join(self.root, 'frames'))])
        img_path = os.path.join(self.root, 'frames')
        neg_mask_path = os.path.join(self.root, 'masks')
        pos_mask_path = os.path.join(self.root, 'PosAnnotations')
        imgs = torch.stack([self.read_img(os.path.join(img_path, '{:06d}.png'.format(i))) for i in frame_ids])
        neg_masks = torch.stack([self.read_mask(os.path.join(neg_mask_path, '{:06d}.png'.format(i))) for i in frame_ids])
        if os.path.exists(pos_mask_path):
            pos_masks = torch.stack([self.read_mask(os.path.join(pos_mask_path, '{:06d}.png'.format(i))) for i in frame_ids])
        else:
            pos_masks = None
        files = ['{:06d}.jpg'.format(i) for i in frame_ids]
        return {'imgs': imgs, 'neg_masks': neg_masks, 'pos_masks': pos_masks, 'files': files}


class Evaluator(object):
    def __init__(self, root, res):
        self.res = res
        self.dataset = Dataset(root)

    def evaluate_video(self, model, prompt, vi_data, output_path):
        imgs = vi_data['imgs']
        neg_masks = vi_data['neg_masks']
        pos_masks = vi_data['pos_masks']
        files = vi_data['files']

        # inference
        t0 = time.time()
        pred_imgs = model(imgs, neg_masks, pos_masks, self.res, prompt)
        t1 = time.time()

        # save output
        for i in range(len(files)):
            fpath = os.path.join(output_path, files[i])
            tv.utils.save_image(pred_imgs[i], fpath)
        return t1 - t0, imgs.size(0)

    def evaluate(self, model, prompt, output_path):
        model.cpu()
        vi_data = self.dataset.get_video()
        os.makedirs(output_path, exist_ok=True)
        seconds, frames = self.evaluate_video(model, prompt, vi_data, output_path)
        print('{:.1f} seconds, {} frames\n'.format(seconds, frames))
